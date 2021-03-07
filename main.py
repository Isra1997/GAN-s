import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
def binning(x):
  print(x.shape)
  # We dont take the first 0-59 colums of the data as they are condisered random values
  # due to the heuristic approch that is used in the pervious model
  load_train_spikes = x[:,60:]
  print(load_train_spikes.shape)
  # create a empty array to insert the values in
  binned_output = np.zeros((8640,2))

  # binning of the spike trains
  # binning at a rate of 40 ms
  for i in range(0,load_train_spikes.shape[0]):
      for j in range(0,load_train_spikes.shape[1]-40,40):
          if (1 in load_train_spikes[i][j:j+40]):
              binned_output[i][int(j/40)]=1

  # create a empty array to rearrange the rows in
  trail_arranged_output = np.zeros((8640,2))

  # re-arrange the data to put all the trails after eachother
  # No. of trails 120
  # No. of neurons 72
  # Total no. of rows 120*72 = 8640
  for trails in range(0,120):
    for neuron_block in range(0,72):
      trail_arranged_output[trails*72+neuron_block,:] = binned_output[neuron_block*120+trails,:]


  return binned_output



def main():
  for i in range(0,100):
    load_train_spikes=np.genfromtxt('/Users/israragheb/Desktop/Generated_Spikes/test_'+str(i)+'.txt',delimiter=',')
    binned_output = binning(load_train_spikes)
    binned_output = torch.from_numpy(binned_output)
    torch.save(binned_output,'/Users/israragheb/Desktop/Binned_Trail_Arranged_Spike_Trains/test_'+ str(i) +'.pt', _use_new_zipfile_serialization=False)
    print('Finished Spkie_Train_'+str(i))

if __name__=='__main__':
  main()
