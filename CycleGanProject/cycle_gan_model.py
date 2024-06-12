import torch
import torch.nn as nn 
from CycleGanProject import load_and_preprocess_dataset

#Two Generators
#Composed of three parts: Encode, resnet skip connects, and decoder
#Note: nn.Sequential is used to sequentially execute layers, NOT for defining layers with specific parameters directly within it. So, I can't put something like: input = nn.Conv2d(input, output_channel, kernel_size)
class Generator(nn.Module):
    def __init__(self, input_channel:int, output_channel:int, kernel_size:int):
        
        super().__init__()
    
        self.conv_layer = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size),
                nn.ReLU(),
                
                nn.Conv2d(input_channel, output_channel, kernel_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size),

                nn.Conv2d(input_channel, output_channel, kernel_size),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size)
    )
        
class ResNet(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size):
        super().__init__()
        self.module = nn.Sequential(
            # first convolutional block
            nn.Conv2d(input_channels, output_channels, kernel_size),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size),
    
            
            nn.Conv2d(output_channels, output_channels, kernel_size),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size),
            
            nn.Conv2d(output_channels, output_channels, kernel_size),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size),
            
            nn.Conv2d(output_channels, output_channels, kernel_size),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size),
            
            nn.Conv2d(output_channels, output_channels, kernel_size),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size),
            
            #Create the add operation for later so that I can add the resnet skip connections later
        )

        