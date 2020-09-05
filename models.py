import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import math
import torch.optim as optim
import numpy as np
from os import listdir, mkdir
import shutil
from torch.autograd import grad, set_detect_anomaly
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from math import isnan
from matplotlib import pyplot as plt

try:
    from apex import amp
    apex_available = True
    # amp.register_float_function(torch, 'sigmoid')
except ModuleNotFoundError:
    apex_available = False

class GeneratorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, first=False, last=False):
        super(GeneratorBlock, self).__init__()
        if first:
            self.mainLine = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, 4, 1, 0),
                                          nn.BatchNorm2d(output_channels),
                                          nn.ReLU(True))
        elif last:
            self.mainLine = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1),
                                          nn.Tanh())
        else:
            self.mainLine = nn.Sequential(nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1),
                                          nn.BatchNorm2d(output_channels),
                                          nn.ReLU(True))

    def forward(self, x):
        return self.mainLine(x)



class DiscriminatorBlock(nn.Module):
    def __init__(self, input_channels, output_channels, first=False):
        super(DiscriminatorBlock, self).__init__()
        if first:
            self.mainLine = nn.Sequential(nn.Conv2d(input_channels, output_channels, 4, 2, 1, bias=False),
                                          nn.LeakyReLU(0.2, inplace=True))
        else:
            self.mainLine = nn.Sequential(nn.Conv2d(input_channels, output_channels, 4, 2, 1, bias=False),
                                          nn.BatchNorm2d(output_channels),
                                          nn.LeakyReLU(0.2, inplace=True))
    def forward(self, image):
        return self.mainLine(image)




class Generator(nn.Module):
    def __init__(self, latent_size, generator_filters, num_layers):
        super(Generator, self).__init__()
        blocks = []
        for layer in range(num_layers):
            if layer == 0:
                blocks.append(GeneratorBlock(latent_size, generator_filters * 2**(num_layers - 2), first=True, last=False))
            elif layer == num_layers -1:
                # 3 as output channels because of RGB
                blocks.append(GeneratorBlock(generator_filters, 3,  first=False, last=True))
            else:
                blocks.append(GeneratorBlock(generator_filters * 2**(num_layers - 1 - layer), generator_filters
                                   * 2**(num_layers - 2 - layer), first=False, last=False))
        self.blocks = nn.Sequential(*blocks)

    def forward(self, latent_vector):
        return self.blocks(latent_vector)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.shape[0], -1)

class Discriminator(nn.Module):
    def __init__(self, discriminator_filters, num_layers):
        super(Discriminator, self).__init__()
        blocks = []
        for layer in range(num_layers-1):
            if layer == 0:
                blocks.append(DiscriminatorBlock(3, discriminator_filters, first=True))
            else:
                blocks.append(DiscriminatorBlock(discriminator_filters * 2**(layer - 1), discriminator_filters * 2**(layer), first=False))
        # at this point, the image is 4x4
        blocks.append(Flatten())
        blocks.append(nn.Dropout(p=0.4))
        blocks.append(nn.Linear(discriminator_filters * 2**(num_layers - 2) * 4 * 4, 1))
        self.blocks = nn.Sequential(*blocks)
    def forward(self, image):
        return self.blocks(image)



class DCGan(nn.Module):
    def __init__(self, latent_size, generator_filters, discriminator_filters, num_layers, device):
        super(DCGan, self).__init__()
        self.generator = Generator(latent_size, generator_filters, num_layers).to(device)
        self.generatorOptimizer = optim.Adam(self.generator.parameters(), lr=0.0004, betas=(0.5, 0.999))

        self.discriminator = Discriminator(discriminator_filters, num_layers).to(device)
        self.discriminatorOptimizer = optim.Adam(self.discriminator.parameters(), lr=0.0008, betas=(0.5, 0.999))

        self.generator, self.generatorOptimizer = amp.initialize(self.generator, self.generatorOptimizer, opt_level="O2",
                                                                 keep_batchnorm_fp32=None, loss_scale="dynamic", max_loss_scale=2**13)
        self.discriminator, self.discriminatorOptimizer = amp.initialize(self.discriminator, self.discriminatorOptimizer, opt_level="O2",
                                                                         keep_batchnorm_fp32=None, loss_scale="dynamic", max_loss_scale=2**13)




class Trainer():
    def __init__(self, batch_size, image_size, latent_size, epochs, discriminator_filters, generator_filters, device):
        self.num_layers = int(np.log2(image_size) - 1)
        self.DCGan = DCGan(latent_size, generator_filters, discriminator_filters, self.num_layers, device).to(device)
        self.batch_size = batch_size
        self.latent_size = latent_size

        assert image_size in [2**x for x in range(5, 11)]
        # self.discriminator_loss = torch.tensor(0.).to(device)
        # self.generator_loss = torch.tensor(0.).to(device)
        self.dataLoader = utils.getDataLoader(batch_size, image_size)
        # self.mixed_probability = mixed_probability
        self.epochs = epochs
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.device = device
        self.tensorboard_summary = SummaryWriter('runs5/dcgan')
        self.checkpoint = 0
        self.apex_available = apex_available
        self.constant_input = torch.randn(128, latent_size, 1, 1, device=device)
        self.epoch = 0
        self.flag = False

    def train(self, verbose=True):
        try:
            # set_detect_anomaly(True)
            torch.cuda.empty_cache()

            # load last iteration if training was started but not finished
            if len(listdir("saves5")) > 0:
                # the [4::-3] is because of the file name format, with the number of each checkpoint at these points
                self.loadModel(sorted(listdir('saves5'), key = lambda x: int(x[4: -3]))[-1][4:-3])
                self.epoch = int(sorted(listdir('saves5'), key = lambda x: int(x[4: -3]))[-1][4:-3])
                print("Loading from epoch: ", self.epoch)
                self.epoch = self.epoch + 1
                print("New epoch starts at: ", self.epoch)
                print("New checkpoint starts at: ", self.checkpoint)
            else:
                print("New weights")
                self.DCGan.generator.apply(utils.init_weights)
                self.DCGan.discriminator.apply(utils.init_weights)
                self.epoch = 0
                self.checkpoint = 0

            # training loop
            for epoch in range(0, self.epochs):
                for batch_num, batch in enumerate(self.dataLoader):

                    # if batch_num % 50 == 0:
                    #     # generated_images = self.StyleGan.generator(self.constant_style, self.constant_noise)
                    #     img_grid = make_grid(generated_images)
                    #     self.tensorboard_summary.add_image(f'generated_image{self.checkpoint}', img_grid)
                    #     del generated_images
                    #     del img_grid
                    batch = batch[0].to(self.device)
                    batch_size = batch.shape[0]
                    batch.requires_grad = True

                    # Train Discriminator: maximize log(D(x)) + log(1 - D(G(z)))
                    for _ in range(1):
                        self.DCGan.discriminator.train()
                        self.DCGan.generator.eval()

                        self.DCGan.discriminatorOptimizer.zero_grad()

                        # Train Discriminator on real data
                        real_labels = (torch.ones(batch_size) * 0.9).to(self.device)
                        fake_labels = (torch.ones(batch_size) * 0.1).to(self.device)
                        labels = torch.cat((real_labels, fake_labels), dim=0)
                        del real_labels
                        del fake_labels
                        noise_input = torch.randn(batch_size, self.latent_size, 1, 1, device=self.device)
                        generated_images = self.DCGan.generator(noise_input).detach().to(self.device)
                        del noise_input
                        images = torch.cat((batch, generated_images), dim=0)

                        discriminator_output = self.DCGan.discriminator(images).resize(batch_size * 2)
                        # print(discriminator_output).resize(-1)
                        del images
                        discriminator_loss = self.loss_fn(discriminator_output, labels)

                        del discriminator_output

                        discriminator_loss_mean = discriminator_loss.mean().item()
                        print("DISCRIM LOSS:", discriminator_loss_mean)

                        if self.apex_available:
                            with amp.scale_loss(discriminator_loss, self.DCGan.discriminatorOptimizer) as scaled_loss:
                                scaled_loss.backward()
                        else:
                            discriminator_loss.backward(retain_graph=True)
                        del discriminator_loss

                        for p in self.DCGan.discriminator.parameters():
                            p.data.clamp_(-1, 1)


                        self.DCGan.discriminatorOptimizer.step()

                        # Apply Gradient Penalty every 4 steps
                        # if batch_num % 4 == 0:
                        # discriminator_total_loss = discriminator_total_loss + utils.gradientPenalty(batch, discriminator_real_output, self.device)

                        if isnan(discriminator_loss_mean):
                            print("IS NAN discriminator")
                            break



                        ##################################################################################################################33
                        # # Train Discriminator on real data
                        # real_labels = (torch.ones(batch_size) * 0.9).to(self.device)
                        #
                        # discriminator_real_output = self.DCGan.discriminator(batch).reshape(-1).to(self.device)
                        # discriminator_real_loss = self.loss_fn(discriminator_real_output, real_labels)
                        # discriminator_real_loss_mean = discriminator_real_loss.mean().item()
                        #
                        # # discriminator_gradient_penalty = utils.gradientPenalty(batch, discriminator_real_output, self.device)
                        # # if self.apex_available:
                        # #     with amp.scale_loss(discriminator_gradient_penalty, self.DCGan.discriminatorOptimizer) as scaled_loss:
                        # #         scaled_loss.backward(retain_graph=True)
                        # # else:
                        # #     discriminator_gradient_penalty.backward(retain_graph=True)
                        #
                        # del discriminator_real_output
                        #
                        #
                        # del real_labels
                        #
                        # if self.apex_available:
                        #     with amp.scale_loss(discriminator_real_loss, self.DCGan.discriminatorOptimizer) as scaled_loss:
                        #         scaled_loss.backward()
                        # else:
                        #     discriminator_real_loss.backward(retain_graph=True)
                        # del discriminator_real_loss
                        #
                        #
                        # # Train Discriminator on fake data
                        # fake_labels = (torch.ones(batch_size) * 0.1).to(self.device)
                        # noise_input = torch.randn(batch_size, self.latent_size, 1, 1, device=self.device)
                        # generated_images = self.DCGan.generator(noise_input.detach()).to(self.device)
                        #
                        # del noise_input
                        #
                        # discriminator_fake_output = self.DCGan.discriminator(generated_images.detach()).reshape(-1).to(self.device)
                        # del generated_images
                        # discriminator_fake_loss = self.loss_fn(discriminator_fake_output, fake_labels)
                        # discriminator_fake_loss_mean = discriminator_fake_loss.item()
                        #
                        # del fake_labels
                        # del discriminator_fake_output
                        #
                        # if self.apex_available:
                        #     with amp.scale_loss(discriminator_fake_loss, self.DCGan.discriminatorOptimizer) as scaled_loss:
                        #         scaled_loss.backward()
                        # else:
                        #     discriminator_fake_loss.backward(retain_graph=True)
                        # del discriminator_fake_loss
                        #
                        # # Add the gradients from real and fake
                        # total_discriminator_loss = discriminator_fake_loss_mean + discriminator_real_loss_mean
                        #
                        # # torch.nn.utils.clip_grad_norm_(self.DCGan.discriminator.parameters(), 1, norm_type=2)
                        # for p in self.DCGan.discriminator.parameters():
                        #     p.data.clamp_(-1, 1)
                        #
                        # self.DCGan.discriminatorOptimizer.step()
                        #
                        # # Apply Gradient Penalty every 4 steps
                        # # if batch_num % 4 == 0:
                        # # discriminator_total_loss = discriminator_total_loss + utils.gradientPenalty(batch, discriminator_real_output, self.device)
                        #
                        #
                        # if isnan(total_discriminator_loss):
                        #     print("IS NAN discriminator")
                        #     break

                        # torch.nn.utils.clip_grad_norm_(self.StyleGan.discriminator.parameters(), 1, norm_type=2)
                        # for p in self.StyleGan.discriminator.parameters():
                        #     p.data.clamp_(-0.01, 0.01)
                        ############################################################################################################



                    # Train Generator: maximize log(D(G(z)))


                    self.DCGan.discriminator.eval()
                    self.DCGan.generator.train()

                    self.DCGan.generatorOptimizer.zero_grad()

                    # generated_images.requires_grad = True

                    generator_labels = torch.ones(batch_size).to(self.device)

                    noise_input = torch.randn(batch_size, self.latent_size, 1, 1, device=self.device)
                    generated_images = self.DCGan.generator(noise_input).to(self.device)
                    del noise_input

                    generator_output = self.DCGan.discriminator(generated_images).reshape(-1).to(self.device)

                    del generated_images

                    generator_loss = self.loss_fn(generator_output, generator_labels)

                    generator_loss_mean = generator_loss.mean().item()
                    print("GENERATPR LOSS:", generator_loss_mean)
                    del generator_labels
                    del generator_output
                    self.DCGan.generator.requires_grad=True
                    if self.apex_available:
                        with amp.scale_loss(generator_loss, self.DCGan.generatorOptimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        generator_loss.backward(retain_graph=True)
                    del generator_loss

                    # torch.nn.utils.clip_grad_norm_(self.DCGan.generator.parameters(), 1, norm_type=2)
                    for p in self.DCGan.generator.parameters():
                        p.data.clamp_(-1, 1)

                    self.DCGan.generatorOptimizer.step()

                    if isnan(generator_loss_mean):
                        print("isnan generator")
                        break

                    # generator_loss_no_pl = generator_loss
                    #
                    # # Apply Path Length Regularization every 16 steps
                    # if batch_num % 10 == 0:
                    #     num_pixels = generated_images.shape[2] * generated_images.shape[3]
                    #     noise_to_add = (torch.randn(generated_images.shape)/ math.sqrt(num_pixels)).to(self.device)
                    #     outputs = (generated_images * noise_to_add)
                    #
                    #     # del generated_images
                    #     pl_gradient = grad(outputs = outputs,
                    #                        inputs = noise_input, grad_outputs = torch.ones(outputs.shape).to(self.device),
                    #                        create_graph=True, retain_graph=True, only_inputs=True)[0]
                    #     del num_pixels
                    #     del noise_to_add
                    #     del outputs
                    #
                    #     pl_length = torch.sqrt(torch.sum(torch.square(pl_gradient)))
                    #
                    #
                    #     if self.average_pl_length is not None:
                    #         pl_regularizer = ((pl_length - self.average_pl_length)**2).mean()
                    #     else:
                    #         pl_regularizer = (pl_length**2).mean()
                    #
                    #
                    #     del pl_gradient
                    #
                    #
                    #
                    #     # print("PL LENGTH IS: ", pl_length)
                    #     if self.average_pl_length == None:
                    #         self.average_pl_length = pl_length.detach().item()
                    #     else:
                    #         self.average_pl_length = self.average_pl_length * self.pl_beta + (1 - self.pl_beta) * pl_length.detach().item()
                    #     # self.average_pl_length = pl_length
                    #
                    #     del pl_length
                    #
                    #     generator_loss = generator_loss + pl_regularizer



                    # for p in self.StyleGan.generator.parameters():
                    #     p.data.clamp_(-0.01, 0.01)
                    # generator_accuracy = generator_loss.argmax == generator_labels  # TODO

                    # Update MappingNetwork weights
                    # if self.apex_available:
                    #     with amp.scale_loss(generator_loss, self.StyleGan.generatorOptimizer) as scaled_loss:
                    #         scaled_loss.backward()
                    # else:
                    #     generator_loss.backward(retain_graph = True)

                    if verbose == True:
                        if batch_num % 100 == 0 and batch_num != 0:
                            # print("average path length is: ", self.average_pl_length)
                            print("Checkpoint")
                            print("Batch: ", batch_num)
                            # print("Discriminator Mean Real Loss: ", discriminator_real_loss_mean)
                            # print("Discriminator Mean Fake Loss: ", discriminator_fake_loss_mean)
                            print("Discriminator Total Loss: ", discriminator_loss_mean)
                            # print("Discriminator Accuracy: ", discriminator_accuracy)
                            # print("Generator Loss (no pl)", generator_loss_no_pl)
                            print("Generator Loss: ", generator_loss_mean)
                            # print("Generator Accuracy: ", generator_accuracy)
                            # print("PL difference:", pl_regularizer.item())

                    # if batch_num % 1 == 0:
                    #     print(batch_num)
                    #     print("OMG NEW ONE",self.DCGan.discriminator.state_dict()[
                    #               "blocks.3.mainLine.1.weight"][0].item())
                    #     self.tensorboard_summary.add_scalar('Discriminator Example Weight',
                    #                                         self.DCGan.discriminator.state_dict()[
                    #                                             "blocks.3.mainLine.1.weight"][0].item(), self.checkpoint)
                    #     self.tensorboard_summary.add_scalar('Generator Example Weight ',
                    #                                         self.DCGan.generator.state_dict()[
                    #                                             "blocks.4.mainLine.0.weight"][0][0][0][0].item(),
                    #                                         self.checkpoint)

                    if batch_num % 100 == 0 and batch_num != 0:
                        print("Current Checkpoint is: ", self.checkpoint)
                        # generated_images = self.DCGan.generator(self.constant_input)
                        # img_grid = make_grid(generated_images)

                        # self.tensorboard_summary.add_scalar('Path Length Mean', self.average_pl_length, self.checkpoint)
                        # self.tensorboard_summary.add_scalar('Discriminator Mean Real Loss ',
                        #                                     discriminator_real_loss_mean, self.checkpoint)
                        # self.tensorboard_summary.add_scalar('Discriminator Mean Fake Loss ',
                        #                                     discriminator_fake_loss_mean, self.checkpoint)
                        self.tensorboard_summary.add_scalar('Discriminator Total Loss ', discriminator_loss_mean,
                                                            self.checkpoint)
                        self.tensorboard_summary.add_scalar('Generator Loss', generator_loss_mean, self.checkpoint)
                        # self.tensorboard_summary.add_scalar('Path Length Difference', pl_regularizer.item(), self.checkpoint)
                        # self.tensorboard_summary.add_scalar('Generator Loss (No PL)', generator_loss_no_pl.item(), self.checkpoint)
                        # self.tensorboard_summary.add_image(f'generated_image{self.checkpoint}', img_grid)
                        self.tensorboard_summary.add_scalar('Discriminator Example Weight',
                                                                self.DCGan.discriminator.state_dict()[
                                                                    "blocks.3.mainLine.1.weight"][0].item(), self.checkpoint)
                        self.tensorboard_summary.add_scalar('Generator Example Weight ',
                                                                self.DCGan.generator.state_dict()[
                                                                    "blocks.4.mainLine.0.weight"][0][0][0][0].item(),
                                                                self.checkpoint)
                        # self.tensorboard_summary.add_scalar("D")
                        # del generated_images
                        # del img_grid
                        # self.tensorboard_summary.add_scalar('Generator Weight', self.StyleGan.generator.we, self.checkpoint)
                        # self.tensorboard_summary.add_scalar('Generator Weight', generator_loss_no_pl.item(), self.checkpoint)
                        # del generator_loss_no_pl
                        del discriminator_loss_mean
                        del generator_loss_mean
                        # del pl_regularizer
                        # del discriminator_real_loss_mean
                        # del discriminator_fake_loss_mean
                        self.checkpoint = self.checkpoint + 1


                # if generator_loss_mean > 1.0:
                #     if not self.flag:
                #         self.flag = True
                #         continue
                #     self.saveModel(self.epoch)
                #     self.epoch = self.epoch + 1
                # continue
                # Right now, an epoch is never achieved
                # if epoch % 10 != 0:
                #     continue
                print("End of Epoch:  ", self.epoch)
                print("Current Checkpoint is: ", self.checkpoint)
                generated_images = self.DCGan.generator(self.constant_input)
                img_grid = make_grid(generated_images)

                # self.tensorboard_summary.add_scalar('Path Length Mean', self.average_pl_length, self.checkpoint)
                # self.tensorboard_summary.add_scalar('Discriminator Mean Real Loss ',
                #                                     discriminator_real_loss_mean, self.checkpoint)
                # self.tensorboard_summary.add_scalar('Discriminator Mean Fake Loss ',
                #                                     discriminator_fake_loss_mean, self.checkpoint)
                self.tensorboard_summary.add_scalar('Discriminator Total Loss ', discriminator_loss_mean,
                                                    self.checkpoint)
                self.tensorboard_summary.add_scalar('Generator Loss', generator_loss_mean, self.checkpoint)
                # self.tensorboard_summary.add_scalar('Path Length Difference', pl_regularizer.item(), self.checkpoint)
                # self.tensorboard_summary.add_scalar('Generator Loss (No PL)', generator_loss_no_pl.item(), self.checkpoint)
                self.tensorboard_summary.add_image(f'generated_image{self.checkpoint}', img_grid)
                self.tensorboard_summary.add_scalar('Discriminator Example Weight',
                                                        self.DCGan.discriminator.state_dict()[
                                                            "blocks.3.mainLine.1.weight"][0].item(), self.checkpoint)
                self.tensorboard_summary.add_scalar('Generator Example Weight ',
                                                        self.DCGan.generator.state_dict()[
                                                            "blocks.4.mainLine.0.weight"][0][0][0][0].item(),
                                                        self.checkpoint)
                            # self.tensorboard_summary.add_scalar("D")
                del generated_images
                del img_grid
                # self.tensorboard_summary.add_scalar('Generator Weight', self.StyleGan.generator.we, self.checkpoint)
                # self.tensorboard_summary.add_scalar('Generator Weight', generator_loss_no_pl.item(), self.checkpoint)
                # del generator_loss_no_pl
                del discriminator_loss_mean
                del generator_loss_mean
                # del pl_regularizer
                # del discriminator_real_loss_mean
                # del discriminator_fake_loss_mean
                self.saveModel(self.epoch)
                self.epoch = self.epoch + 1
                self.checkpoint = self.checkpoint + 1

            # Close TensorBoard at the end
            self.tensorboard_summary.close()
        except KeyboardInterrupt:
            self.saveModel(self.epoch)
            self.epoch = self.epoch + 1


    @torch.no_grad()
    def evaluate(self, iteration):

        def show(img):
            npimg = img.numpy()
            plt.imshow(np.transpose(npimg, (1, 2, 0)), interpolation='nearest')

        # load last iteration if training was started but not finished
        if len(listdir("saves5")) > 0:
            # the [4::-3] is because of the file name format, with the number of each checkpoint at these points
            self.loadModel(iteration)


        # noise_inputs = torch.randn(128, self.latent_size, 1, 1).cuda()
        # noise_inputs[1, :, 0, 0] = 1
        # noise_inputs = torch.arange(128)
        # print(noise_inputs.size())
        # print(self.DCGan.generator)
        # generated_images = self.DCGan.generator(noise_inputs).cpu()
        # print(generated_images.size())
        noise_inputs = torch.randn(32, self.latent_size, 1, 1).cuda()
        for x in np.arange(-1, 1, 0.2):

            noise_inputs[:, :, 0, 0] = x
            generated_images = self.DCGan.generator(noise_inputs).cpu()
            utils.showTrainingImages(generated_images, training=False)

    def create_interpolation(self):
        pass

    def saveModel(self, iteration):
        save_dict = {'generatorModel': self.DCGan.generator.state_dict(),
                     'generatorModelOptimizer': self.DCGan.generatorOptimizer.state_dict(),
                     "discriminatorModel": self.DCGan.discriminator.state_dict(),
                     "discriminatorModelOptimizer": self.DCGan.discriminatorOptimizer.state_dict(),
                     'amp': amp.state_dict(),
                     # "average_pl": self.average_pl_length,
                     # "constant_style": self.constant_style,
                     # "constant_noise": self.constant_noise,
                     # "style_network": self.StyleGan.styleNetwork.state_dict(),
                     "constant_input": self.constant_input,
                     "checkpoint": self.checkpoint}

        torch.save(save_dict, f"saves5/Gan-{iteration}.pt")

    def loadModel(self, iteration):
        load_dict = torch.load(f"saves5/Gan-{iteration}.pt")
        # self.checkpoint = 41
        # load_dict["generatorModelOptimizer"]["param_groups"][0]['lr'] = 0.0004
        # load_dict["discriminatorModelOptimizer"]["param_groups"][0]['lr'] = 0.0008

        # load_dict["generatorModelOptimizer"]["param_groups"][0]['betas'] = (0.5, 0.99)
        # print(load_dict["average_pl"])
        self.DCGan.generator.load_state_dict(load_dict["generatorModel"])
        self.DCGan.generatorOptimizer.load_state_dict(load_dict["generatorModelOptimizer"])
        self.DCGan.discriminator.load_state_dict(load_dict["discriminatorModel"])
        self.DCGan.discriminatorOptimizer.load_state_dict(load_dict["discriminatorModelOptimizer"])
        # self.average_pl_length = load_dict["average_pl"]
        # self.constant_style = load_dict["constant_style"]
        # self.constant_noise = load_dict["constant_noise"]
        # self.StyleGan.styleNetwork.load_state_dict(load_dict["style_network"])
        self.constant_input = load_dict["constant_input"]
        self.checkpoint = load_dict["checkpoint"]
        amp.load_state_dict(load_dict["amp"])

    def resetSaves(self):
        shutil.rmtree('saves')
        mkdir("saves")
        shutil.rmtree('runs')
        mkdir("runs")