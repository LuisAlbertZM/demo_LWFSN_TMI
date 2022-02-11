import torch
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Sequential

# NETWORKS
class __unet_conv_block__(nn.Module):
    def __init__(self, indf, ondf):
        super(__unet_conv_block__, self).__init__()
        self.cblock = Sequential(
            nn.Conv2d(indf, ondf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ondf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ondf, ondf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ondf),
            nn.ReLU(inplace=True)
        )
        self.pool = Sequential(
            nn.ReflectionPad2d(padding =(0,1,0,1)),
            nn.MaxPool2d(kernel_size=2))
    def forward(self, x):
        conv = self.cblock(x)
        return(self.pool(conv), conv)

class __unet_up_block__(nn.Module):
    def __init__(self, indf, ondf, kernel_size=3, padding=1):
        super(__unet_up_block__, self).__init__()  
        self.reduce = nn.Sequential(
            nn.Conv2d(indf*2, indf, kernel_size=1, padding=0),
            nn.BatchNorm2d(indf),
            nn.ReLU(inplace=True))
        self.cblock = nn.Sequential(
            nn.Conv2d(indf  , ondf, kernel_size=3, padding=1),
            nn.BatchNorm2d(ondf),
            nn.ReLU(inplace=True))
        self.up = nn.Sequential( 
            nn.ConvTranspose2d(indf, indf, kernel_size = 1, stride=2, output_padding=1, bias=True),
            nn.BatchNorm2d(indf),
            nn.ReLU(inplace=True))
    def forward(self, x, bridge):
        conc = torch.cat([self.up(x),bridge],1)
        red = self.reduce(conc)
        conv = self.cblock(red)
        return  conv
    
class fpbConvNet2d(nn.Module):
    def __init__( self, in_chans=2, depth=5, wf=16):
        super(fpbConvNet2d, self).__init__()
        self.depth = depth
        #  Begining architecture
        prev_channels = in_chans
        self.down_path = nn.ModuleList()
        for i in range(depth):
            out_chans = int(wf*in_chans*(2**(i+1)))
            self.down_path += [__unet_conv_block__( prev_channels, out_chans)]
            prev_channels = out_chans 
        self.up_path = nn.ModuleList()
        for i in range(depth):
            out_chans = int(wf*in_chans*(2**(depth-i-1)))
            self.up_path += [__unet_up_block__( prev_channels, out_chans)]
            prev_channels = out_chans
        #print([prev_channels, in_chans])
        self.last = nn.Sequential(
            nn.Conv2d(prev_channels, in_chans, kernel_size=1))
        
    def forward(self, x):
        blocks = []
        bridges = []
        x_in = x
        
        x = x
        for i, down in enumerate(self.down_path):
            x, bridge = down(x)
            bridges.append(bridge)
        
        for i, up in enumerate(self.up_path):
            ind = self.depth - i -1
            x = up(x, bridges[ind])
            
        return(x_in + self.last(x))