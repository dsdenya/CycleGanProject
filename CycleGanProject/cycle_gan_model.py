import torch
import torch.nn as nn
import torch.nn.functional as F
import CycleGanProject


# Two Generators
# Composed of three parts: Encode, resnet skip connects, and decoder
# Note: nn.Sequential is used to sequentially execute layers, NOT for defining layers with specific parameters directly within it. So, I can't put something like: input = nn.Conv2d(input, output_channel, kernel_size)

# Encoder
class Encoder(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, kernel_size: int):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(input_channel, output_channel, kernel_size),
            nn.ReLU(),
            nn.Conv2d(output_channel, output_channel, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size),
            nn.Conv2d(output_channel, output_channel, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size),
        )

    def forward(self, x):
        return self.conv_layer(x)

# Resnet Block
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
        )
        self.skip_module = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size),
        )

    def forward(self, x):
        return self.module(x) + self.skip_module(x)

# Decoder
class Decoder(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size):
        super().__init__()
        self.deconv_layer = nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size),
            nn.ReLU(),
            nn.ConvTranspose2d(output_channel, output_channel, kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size),
            nn.ConvTranspose2d(output_channel, output_channel, kernel_size),
            nn.ReLU(),
            nn.ConvTranspose2d(output_channel, output_channel, kernel_size),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.deconv_layer(x)


class Generator(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, kernel_size: int):
        super().__init__()

        self.encoder = Encoder(input_channel, output_channel, kernel_size)
        self.resnet = ResNet(input_channel, output_channel, kernel_size)
        self.decoder = Decoder(input_channel, output_channel, kernel_size)

    def forward(self, x):
        encoded = self.encoder(x)
        resnet = self.resnet(encoded)
        decoded = self.decoder(resnet)
        return decoded
    

#This uses a patchGAN discriminator instead of the usual discriminator
class Discriminator(nn.Module): 
    def __init__(self,input_channel:int,output_channel:int ,kernel_size:int):
        super().__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d (input_channel,output_channel,kernel_size),
            nn.Conv2d (output_channel,output_channel,kernel_size),
            nn.BatchNorm2d(output_channel),
            nn.Conv2d (output_channel,output_channel,kernel_size),
            nn.BatchNorm2d(output_channel),
            nn.Conv2d (output_channel,output_channel,kernel_size),
            nn.BatchNorm2d(output_channel),
            nn.Conv2d (output_channel,output_channel,kernel_size)
            )
        
        self.weight_init(mean=0.0, std=0.02)

    def weight_init(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, mean, std)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, input, label):
      x = torch.cat([input, label], 1)
      x = F.leaky_relu(self.discriminator(x), 0.2)
      x = F.sigmoid(x)

      return x
    
def cyclecon_loss(recon_target, target):
    m = nn.Sigmoid()
    loss = nn.BCELoss()
    output = loss(m(recon_target), target )
    return output

input_channel = 3
output_channel = 64
kernel_size = 3


generatorA = Generator(input_channel, output_channel, kernel_size)
generatorB = Generator(input_channel, output_channel, kernel_size)

discriminatorA = Discriminator(input_channel, output_channel,kernel_size)
discriminatorB = Discriminator(input_channel, output_channel,kernel_size)




