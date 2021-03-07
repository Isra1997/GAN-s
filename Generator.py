import torch
from torch import nn
import numpy as np
from PIL import Image , ImageOps
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.optim as optim
import matplotlib.pyplot as plt
from numpy import genfromtxt
from torch.utils.data import TensorDataset


# noise function however used to get images to pass to the generator which we consider as noise
def load_images():
    # transform an images of shape[3,288,287] grescale and to a tensor
    tranform_X = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                   transforms.ToTensor()])

    train_set_X = datasets.ImageFolder("/Users/israragheb/Desktop/Masters_Dataset/Train",transform=tranform_X)

    # dataloader loads our data in batches can also shuffle our data
    train_loader = torch.utils.data.DataLoader(train_set_X,batch_size=15,shuffle=True)

    return train_loader

# creating a genegrator subclass that is a subclass of the Module which is the base class of neural networks
class Generator(nn.Module):
    """
    Generator Class
    Values:
    input_dim: the dimensions of the input vector, a scaler
    hidden_dim: the inner dimensions of the cnn hiiden layers, a scaler
    """


    def __init__(self):
        super().__init__()
        # Build the neural network according to the read papers
        self.gen =  nn.Sequential(
                nn.Conv2d(in_channels=1 ,out_channels=16 ,kernel_size=7 ,stride=1 ,padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LeakyReLU(0.2),
                nn.Conv2d(in_channels=16 ,out_channels=32 ,kernel_size=7 ,stride=1 ,padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.LeakyReLU(0.2),
                nn.Linear(66,72),
                nn.Sigmoid()
            )

    def forward_pass(self ,image):
        """
        Function that given a noise vector generates the expected spike trains.
        Parameters:
        noise: a random noise vector
        """
        # .veiw(): Allows a tensor to be a view of an exsisting tensor. View tensor shares the same underlyinb
        # data with its base tensor. Avoids explict copying of the tensor , which allows us to do fast
        # efficient reshaping, slicing and element-wise operations.
        # x = image.view()
        return self.gen(image)


def main():
    gen = Generator()
    for inputs,labels in load_images():
        print(inputs.size())
        gen.forward_pass(inputs)

if __name__ == '__main__':
     main()
