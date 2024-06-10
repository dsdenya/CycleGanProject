#Generator
#Composed of three parts: Encode, transformer, and decoder


import torch
import torch.nn as nn 
from CycleGanProject import load_and_preprocess_dataset

class Generator(nn.Module):
    def __init__(self, input_channel:int, output_channel:int, kernel_size:int):
        
        super().__init__()
    
    conv_layer = nn.Sequential (
         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)
         self.rel1 = nn.ReLu(),
         
         self.conv2 = nn.Conv2d(input_channel, output_channel, kernel_size, batch_size)
         self.rel2 = nn.ReLu()
         self.pool2 = nn.MaxPool2d(kernel_size)

         self.conv3 = nn.Conv2d(input_channel, output_channel, kernel_size, batch_size)
         self.rel3 = nn.ReLu()
         self.pool3 = nn.MaxPool2d(kernel_size)



 )
    