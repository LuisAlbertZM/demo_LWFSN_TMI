import torch
import numpy as np
import torch.nn as nn 
import torch.nn.functional as F
from torch.nn import Sequential
from .kernels import convKernel 
from .dwtHaar import dwtHaar_2d, __upsample_zeropadding_2D__


"""
    Description: Custom implementation of inverse 2D DWT with 
    Haar kernel
"""
class customIdwtHaar_2d(nn.Module):
    def __init__( self,mode="reflect"):
        super(customIdwtHaar_2d, self).__init__()
        self.mode = mode
        self.upsample = __upsample_zeropadding_2D__()
            
        # Inverse 2D Haar DWT
        iLL = np.asarray([[[[ 1,  1], [ 1,  1]]]])*0.5
        iLH = np.asarray([[[[-1, -1], [ 1,  1]]]])*0.5
        iHL = np.asarray([[[[ 1, -1], [ 1, -1]]]])*0.5
        iHH = np.asarray([[[[ 1, -1], [-1,  1]]]])*0.5
        self.iLL = nn.Parameter(torch.from_numpy(iLL)).to(torch.float32).cuda()
        self.iLH = nn.Parameter(torch.from_numpy(iLH)).to(torch.float32).cuda()
        self.iHL = nn.Parameter(torch.from_numpy(iHL)).to(torch.float32).cuda()
        self.iHH = nn.Parameter(torch.from_numpy(iHH)).to(torch.float32).cuda()
        
    def forward(self, LL, LH, HL, HH):
        # Loading parameters
        mode=self.mode
        stride=1

        # Reshaping for easy convolutions
        #with torch.no_grad():
        # Upsampling by inserting zeros
        x = torch.cat([LL, LH, HL, HH],axis=1)
        xp =  self.upsample(x)
        xpp = F.pad(xp,(0,1,0,1),mode=mode,value=0)
        LL2, LH2, HL2, HH2 = torch.split(xpp, LL.shape[1], dim=1)

        lls = LL2.shape
        # Inverse transform
        cs = [lls[0]*lls[1],1,lls[2],lls[3]]
        iLL = F.conv2d(LL2.reshape(*cs), self.iLL, bias=None, stride=stride)
        iLH = F.conv2d(LH2.reshape(*cs), self.iLH, bias=None, stride=stride)
        iHL = F.conv2d(HL2.reshape(*cs), self.iHL, bias=None, stride=stride)
        iHH = F.conv2d(HH2.reshape(*cs), self.iHH, bias=None, stride=stride)
        
        ills = iLL.shape
        co = [lls[0], lls[1], ills[2], ills[3]]
        iLL = iLL.reshape(*co)
        iLH = iLH.reshape(*co)
        iHL = iHL.reshape(*co)
        iHH = iHH.reshape(*co)
        
        return([iLL, iLH, iHL, iHH])
    
class biasLayer(nn.Module):
    def __init__(self, chans):
        super(biasLayer, self).__init__()
        self.bias = nn.Parameter( torch.rand((chans)))
        stdv = 1. / np.sqrt(chans)
        self.bias.data.uniform_(-stdv, stdv)
        
        self.register_parameter("bias", self.bias)
    def forward(self ):
        return(self.bias)

"""
    Description: implementation of the tight frame U-Net 
"""
class rtfunet_2d(nn.Module):
    def __init__( self, in_channels=1, depth=1, wf=4):
        super(rtfunet_2d, self).__init__()
        self.depth = depth
        
        # Encoding path
        self.shrink = nn.ModuleList()
        
        self.we1 = nn.ModuleList()
        self.we2 = nn.ModuleList()
        self.wd1 = nn.ModuleList()
        self.wd2 = nn.ModuleList()
        
        self.be1  = nn.ModuleList()
        self.be2  = nn.ModuleList()
        self.bd1  = nn.ModuleList()
        self.bd2  = nn.ModuleList()
        
        self.ne1  = nn.ModuleList()
        self.ne2  = nn.ModuleList()
        self.nd1  = nn.ModuleList()
        self.nd2  = nn.ModuleList()
        
        #nn.BatchNorm2d(num_features)

        ks = 3
        prev_channels = in_channels 
        for i in range(depth):
            out_channels = wf*2**i
            
            # Weight for the wavelet frame
            self.we1 += [convKernel(out_channels, prev_channels, ks, ks) ]
            self.we2 += [convKernel(out_channels, out_channels, ks, ks) ]
            if i == 0:
                self.wd1 += [convKernel(out_channels, out_channels, ks, ks) ]
            else:
                self.wd1 += [convKernel(out_channels, prev_channels, ks, ks) ]
            self.wd2 += [convKernel(out_channels*5, out_channels, ks, ks) ]
            
            # Bias layer
            self.be1  += [biasLayer(out_channels) ]
            self.be2  += [biasLayer(out_channels) ]
            if i ==0 :
                self.bd1 += [biasLayer(out_channels) ]
            else:
                self.bd1 += [biasLayer(prev_channels) ]
            self.bd2  += [biasLayer(out_channels) ]
            
            # Normalization layers
            self.ne1  += [nn.BatchNorm2d(out_channels) ]
            self.ne2  += [nn.BatchNorm2d(out_channels) ]
            if i ==0 :
                self.nd1 += [nn.BatchNorm2d(out_channels) ]
            else:
                self.nd1 += [nn.BatchNorm2d(prev_channels) ]
            self.nd2  += [nn.BatchNorm2d(out_channels) ]
            
            prev_channels = out_channels
            
        self.weC = convKernel(wf*2**(depth), wf*2**(depth-1), ks, ks)
        self.wdC = convKernel(wf*2**(depth-1), wf*2**(depth), ks, ks)
        
        self.bdC = biasLayer(wf*2**(depth-1))
        self.beC = biasLayer(wf*2**(depth))
        
        self.ndC = nn.BatchNorm2d(wf*2**(depth-1))
        self.neC = nn.BatchNorm2d(wf*2**(depth)) 

        # Defining inverse and forward transforms
        self.dwt = dwtHaar_2d()
        self.idwt = customIdwtHaar_2d()
        self.__set_requires_grad__(self.dwt, False)
        self.__set_requires_grad__(self.idwt, False)
        self.last = convKernel(wf, in_channels, 1, 1)

    def forward(self, x, bypass_shrinkage=False):
        LH_list = []; HL_list = []
        HH_list = []; SS_list = []
        WTl = self.last().transpose(1,0)
        weC = self.weC()
        wdC = self.wdC()
        beC = self.beC()
        bdC = self.bdC()
        neC = self.neC
        ndC = self.ndC

        LL = x
        for i in range(self.depth):
            # Loading kernels
            # Convoluting signal with factorized wavelet frame        
            D, SS = self.__forwardF__(LL, i, bypass_shrinkage)
            LL, LH, HL, HH = D
            LH_list.append( LH )
            HL_list.append( HL )
            HH_list.append( HH )
            SS_list.append( SS )

        # In Han's implementation there are additional convolutions in the deepest level
        st = (1,1)
        LL = F.relu(neC(F.conv2d(LL, weC, bias = beC, stride=st, padding=1)))
        LL = F.relu(ndC(F.conv2d(LL, wdC, bias = bdC, stride=st, padding=1)))
        
        # Inverse transforming
        for i in range(self.depth):
            indx = self.depth - i -1
            LH = LH_list[indx]
            HL = HL_list[indx]
            HH = HH_list[indx]
            SS = SS_list[indx]
            
            # Inverse-wavelet transforming
            LL  = self.__inverseF__(LL, LH, HL, HH, SS, indx, bypass_shrinkage)
        x_r = F.conv2d(LL, WTl, bias = None, stride=(1,1), padding=0)
        return(x - x_r)
    
    def __set_requires_grad__(self, net, requires_grad=False):
        for param in net.parameters():
            param.requires_grad = requires_grad
            
    def __forwardF__(self, x, i, bypass_shrinkage):
        # Loading the weights
        W1 = self.we1[i]()
        W2 = self.we2[i]()
        b1 = self.be1[i]()
        b2 = self.be2[i]()
        n1 = self.ne1[i]
        n2 = self.ne2[i]
        st = (1,1)
        
        # Convolutions
        xW1 = F.relu(n1(F.conv2d(x, W1, bias = b1, stride=st, padding=1)) )
        xW2 = F.relu(n2(F.conv2d(xW1, W2, bias = b2, stride=st, padding=1)) )
            
        return( [self.dwt(xW2), xW2] )
    
    def __inverseF__(self, LL, LH, HL, HH, SS, i, bypass_shrinkage):
        # Loading the weights
        WT1 = self.wd1[i]().transpose(1,0)
        WT2 = self.wd2[i]().transpose(1,0)
        b2 = self.bd2[i]()
        b1 = self.bd1[i]()
        n2 = self.nd2[i]
        n1 = self.nd1[i]
        st = (1,1)
        
        #Inverse frame transform
        LLp, LHp, HLp, HHp = self.idwt(LL, LH, HL, HH)
        dat = torch.cat([LLp, LHp, HLp, HHp, SS], axis=1)

        x21 = F.relu(n2(F.conv2d(dat, WT2, bias = b2, stride=st, padding=1)))
        x11 = F.relu(n1(F.conv2d(x21, WT1, bias = b1, stride=st, padding=1)))

        return( x11 )
