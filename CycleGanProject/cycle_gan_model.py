#Generator

import torch
import torch.nn as nn 
from CycleGanProject import load_and_preprocess_dataset

class Generator(nn.Module):
    def __init__(self, input_channel:int, output_channel:int, kernel_size:int, batch_size:int):
        
        super().__init__()
    
    conv_layer = nn.Sequential (
         nn.Conv2d(input_channel, output_channel, kernel_size, batch_size)
         nn.ReLu(),
         nn.Conv2d(input_channel, output_channel, kernel_size, batch_size)
         nn.ReLu()
         
         
         
         )
    