import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torchvision.datasets as Datasets
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
import numpy as np
from torch.autograd import grad
# from models import MappingNetwork


def getDataLoader(batch_size, image_size):
    """
    Loads the data loader for StyleGAN2 and applies preprocessing steps to it
    :param pathname: the name of the path to the folder containing the data
    :param args: command line arguments
    :return: the custom dataset
    """
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    customDataset = Datasets.ImageFolder(root='images', transform=transform)
    # print(mnist)
    # customDataset =  MNIST('data/mnist', transform= transform)
    dataLoader = DataLoader(customDataset, batch_size=batch_size, shuffle=True)

    return dataLoader


def showImage(image):
    """
    Shows image to screen by removing batch dimention and permuting the pytorch tensor to fit pyplot standards
    :param image: image to show to the screen
    :return: void
    """
    plt.axis("off")
    image = image.cpu().detach()
    plt.imshow(image.squeeze(0).permute(1, 2, 0))
    plt.show()

def showTrainingImages(images, training=True):
    plt.figure(figsize=(images.shape[0], images.shape[0]))
    plt.axis("off")
    if training:
        plt.title("Training Images")
    else:
        plt.title("Genrated Images")
    images = make_grid(images, padding=2, normalize=True, nrow=16)
    # images = images.permute(0, 2, 3, 1).cpu().detach()
    images = images.permute(1, 2, 0)
    plt.imshow(images)
    plt.show()






def create_image_noise(batch_size, image_size, device):
    return torch.randn(batch_size, image_size, image_size, 1).uniform_(0, 1).to(device)


def createNoise(batch_size, latent_size, device):
    return torch.empty(batch_size, latent_size).normal_(mean=0,
                                                        std=0.5).to(device)  # np.random.normal(0, 1, size = [batch_size, latent_size]).astype('float32')


def createStyleNoiseList(batch_size, latent_size, num_layers, StyleVectorizer, device):
    return StyleVectorizer(createNoise(batch_size, latent_size, device))[:, None, :].expand(-1, int(num_layers), -1)


def createStyleMixedNoiseList(batch_size, latent_size, num_layers, StyleVectorizer, device):
    randomCut = np.random.randint(num_layers)
    return torch.cat((createStyleNoiseList(batch_size, latent_size, randomCut, StyleVectorizer, device),
                      createStyleNoiseList(batch_size, latent_size, (num_layers - randomCut), StyleVectorizer, device)), dim=1)


def gradientPenalty(images, probability_of_real, device):

    gradients = grad(outputs=probability_of_real, inputs=images, grad_outputs = torch.ones(probability_of_real.size()).cuda(), create_graph = True, retain_graph=True)[0]
    gradients = gradients.view(images.shape[0], -1)
    # print("in gradien t[enalty", gradients)
    return torch.sum(gradients.square()).mean()


def init_weights(m):
    # Make the weights have mean 0, std 0.02
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    # Make the weights have mean 1, std 0.02
    # Make the bias equal to zero
    elif type(m) == nn.BatchNorm2d:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def set_requires_grad(model, bool):
    for p in model.parameters():
        # print("Start of new millenia")
        # print(p.requires_grad)
        p.requires_grad = bool
        # print(p.requires_grad)
