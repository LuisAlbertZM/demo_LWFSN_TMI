########################################################
# Author: Luis Albert Zavala Mondragon
# Organization: Eindhoven University of Technology
########################################################
import numpy as np
import torch
import torch.nn as nn 
import torch.nn.functional as F


""" class: __upsample_zeropadding_2D__
    Description: Compact implementation of 2D upsampling via insertion 
    of zeros
    Constructor:
        * None
    Inputs:
        * x: Signal to be upsampled
    Outputs:
        * upsi: Upsampled signal
"""
class __upsample_zeropadding_2D__(nn.Module):

    def __init__(self):
        super(__upsample_zeropadding_2D__, self).__init__()
        
        k_np = np.asanyarray([1])
        self.k = nn.Parameter(data = torch.from_numpy(k_np),
            requires_grad=False).float().cuda().reshape((1,1,1,1))
        
    def forward(self, x):
        xs = x.shape
        x_p = x.view(xs[0]*xs[1], 1, xs[2], xs[3])
        
        up = F.conv_transpose2d(x_p, weight=self.k, stride=(2,2), dilation=1)
            
        if up.shape[2] < x.shape[2]*2:
            up = F.pad(input = up, pad = (0, 0, 0, 1), mode="reflect")
        if up.shape[3] < x.shape[3]*2:
            up = F.pad(input = up, pad = (0, 1, 0, 0), mode="reflect")
        
        us = up.shape
        upsi = up.view(xs[0], xs[1], us[2], us[3])
        return(upsi)
    
"""
    Description: Compact implementation of inverse 2D DWT with 
    Haar kernel
"""
class dwtHaar_2d(nn.Module):
    def __init__( self, undecimated=False, mode="reflect"):
        super(dwtHaar_2d, self).__init__()
        self.mode = mode
        #self.no_chans
        if undecimated: self.stride=1
        else: self.stride=2
    
        # 2D Haar DWT
        LL = np.asarray([[[[ 1,  1], [ 1,  1]]]])*0.5
        LH = np.asarray([[[[ 1,  1], [-1, -1]]]])*0.5
        HL = np.asarray([[[[-1,  1], [-1,  1]]]])*0.5
        HH = np.asarray([[[[ 1, -1], [-1,  1]]]])*0.5
        self.LL = nn.Parameter(torch.from_numpy(LL)).to(torch.float32).cuda()
        self.LH = nn.Parameter(torch.from_numpy(LH)).to(torch.float32).cuda()
        self.HL = nn.Parameter(torch.from_numpy(HL)).to(torch.float32).cuda()
        self.HH = nn.Parameter(torch.from_numpy(HH)).to(torch.float32).cuda()


    def forward(self,x):
        # Loading parameters
        mode = self.mode
        stride = self.stride

        # Reshaping for easy convolutions
        #with torch.no_grad():
        # First wavelet transform, second, wavelet transform
        xlc = F.pad(x, (1,0,1,0), mode=mode, value=0)
        xps = xlc.shape
        xlcs = xlc.shape
        
        xlc = xlc.view(xlcs[0]*xlcs[1], 1, xlcs[2], xlcs[3])
        LL = F.conv2d(xlc, self.LL, bias=None, stride=stride)
        LH = F.conv2d(xlc, self.LH, bias=None, stride=stride)
        HL = F.conv2d(xlc, self.HL, bias=None, stride=stride)
        HH = F.conv2d(xlc, self.HH, bias=None, stride=stride)
            
        osi = [xlcs[0],xlcs[1], LL.shape[2], LL.shape[3]]
        return([LL.view(*osi), LH.view(*osi), HL.view(*osi), HH.view(*osi)])

"""
    Description: Compact implementation of inverse 2D DWT with 
    Haar kernel
"""
class idwtHaar_2d(nn.Module):
    def __init__( self,mode="reflect"):
        super(idwtHaar_2d, self).__init__()
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
        xiw = iLL + iLH + iHL + iHH
        ills = iLL.shape
        
        co = [lls[0], lls[1], ills[2], ills[3]]
        return(xiw.reshape(*co))