import torch
import torch.nn as nn 
from CycleGanProject import load_and_preprocess_dataset

#Two Generators
#Composed of three parts: Encode, transformer, and decoder
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
            
            #have to put the add operation here to complete the resnet skip convo blocks here later
        )

        