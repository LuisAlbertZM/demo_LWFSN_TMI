import torch
import torch.nn.functional as F

def soft_threshold(x,b):
    return( F.relu(x - b )-F.relu(-x - b ) )

def tanh(t):
    k = 5
    numerator =  torch.exp(k*t) - torch.exp(-k*t)
    denominator = torch.exp(k*t) + torch.exp(-k*t)
    return(numerator/denominator)

def semiH_threshold4(x, t):
    return(x*(1-tanh(t)*torch.exp(-(x/t)**4) ) )

def semiH_threshold2(x, t):
    return(x*(1-tanh(t)*torch.exp(-(x/t)**2) ) )