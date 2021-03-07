import torch
from torch import nn
import pandas as pd

class discriminrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.disc = nn.Sequential(
            # This will return the stimulus embedding vector
            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=7,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=7,stride=1,padding=0),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.LeakyReLU(0.2),
            # The input here is the addition of the two vectors the stimulus embeddings and Sampled Spikes
            nn.Linear(72,72),
            nn.ReLU(inplace=True),
            nn.Linear(72,72),
            nn.ReLU(inplace=True),
            nn.Linear(72,72),
            nn.ReLU(inplace=True),
            nn.Linear(72,72),
            nn.ReLU(inplace=True)
        )

    def forward(self,Spkie_count):
        return self.disc(Spkie_count)

def binned_output(path):
    data = pd.read_csv(path,sep=',',header=None)
    data.columns = ['T_'+str(i) for i in range(0,4000)]
    trial_one = data['T_0']
    print(data.shape)
    return trial_one[0:72]


def main():
   test = binned_output("/Users/israragheb/Desktop/Masters_Dataset/Train/Spike_Trains/test_0.txt")
   disc = discriminrator()
   print(disc.forward(torch.from_numpy(test.to_numpy()).float()))


if __name__ == '__main__':
    main()

