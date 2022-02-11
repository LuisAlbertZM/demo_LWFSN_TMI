import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, fftshift


def impulseResponseCNN(cnn, disableScaling=False):
    # Defining an unit impulse
    imp = torch.zeros([1,1,32,32]).cuda()
    imp[0,0,16,16] = 1
    
    # Feeding the impulse to the CNN
    with torch.no_grad():
        p=128
        resp = cnn(
            F.pad(imp,(p,p,p,p) ),
            bypass_shrinkage=True
            )[:,:,p:-p,p:-p] 
    respnp = resp[0,0,:,:].detach().cpu().numpy()
    impnp = imp[0,0,:,:].detach().cpu().numpy() 


    # Making the figure
    sc=10
    
    
    # Impulse response
    if disableScaling:
        sc=5
        fig, axs = plt.subplots(nrows=1, ncols=1,figsize=(sc*1.1,sc*1.1))
        
        # Impulse response
        img1 = axs.imshow(respnp,cmap="hot")
        axs.yaxis.set_major_locator(plt.NullLocator())
        axs.xaxis.set_major_locator(plt.NullLocator())
        cbaxes = fig.add_axes([0.91, 0.38, 0.02, 0.4])
        fig.colorbar(img1, orientation='vertical', cax=cbaxes)
        plt.subplots_adjust(wspace=0, hspace= 0.0)
        plt.show()
        
        fig, axs = plt.subplots(nrows=1, ncols=1,figsize=(sc*1.1,sc*1.1))
        # Frequency response
        img=axs.imshow(np.absolute(fftshift(fft2( respnp, (256,256) ) ) ) ,cmap="hot")
        axs.yaxis.set_major_locator(plt.NullLocator())
        axs.xaxis.set_major_locator(plt.NullLocator())
        cbaxes = fig.add_axes([0.91, 0.38, 0.02, 0.8])  # 
        fig.colorbar(img, orientation='vertical', cax=cbaxes)
        plt.subplots_adjust(wspace=0, hspace= 0.0)
        plt.show()
   
    else:
    
        fig, axs = plt.subplots(nrows=1, ncols=3,figsize=(sc*1.1,sc*1.1))

        vmin=-0.2
        vmax= 1.2
        labelsize = 25
        labelcolor="y"

        # Input
        axs[0].imshow(impnp,cmap="hot",vmin=vmin,vmax=vmax)
        axs[0].yaxis.set_major_locator(plt.NullLocator())
        axs[0].xaxis.set_major_locator(plt.NullLocator())
        axs[0].text(0, 25, "Total Energy =%1.2f"%(1),
                        fontsize=16, color='white', )

        axs[1].imshow(respnp,cmap="hot",vmin=vmin,vmax=vmax)
        axs[1].yaxis.set_major_locator(plt.NullLocator())
        axs[1].xaxis.set_major_locator(plt.NullLocator())

        ener_resp = np.sum(respnp**2)
        ener_center_resp = np.sum( (respnp*impnp)**2)
        ener_center_spread = np.sum( (respnp*impnp)**2) -  ener_resp

        axs[1].text(0, 25, "Total Energy =%1.2f"%(ener_resp),
                        fontsize=16, color='white', )
        axs[1].text(0, 27.5, "Energy on center =%1.2f"%(ener_center_resp),
                        fontsize=16, color='white', )

        # Frequency response
        img=axs[2].imshow(np.absolute(fftshift(fft2( respnp, (256,256) ) ) ) ,cmap="hot",vmin=vmin,vmax=vmax)
        axs[2].yaxis.set_major_locator(plt.NullLocator())
        axs[2].xaxis.set_major_locator(plt.NullLocator())
        
        fs=16
        axs[0].set_title("Unit impulse", fontsize= fs)
        axs[1].set_title("Impulse response", fontsize= fs)
        axs[2].set_title("|Frequency response|", fontsize= fs)

            
        cbaxes = fig.add_axes([0.91, 0.38, 0.02, 0.24])  # 
        fig.colorbar(img, orientation='vertical', cax=cbaxes)

        plt.subplots_adjust(wspace=0, hspace= 0.0)
        plt.show()
        
    
import time
def averageForwProp(cnn, iters):
    # Defining an unit impulse
    im = torch.zeros([1,1,512,512]).cuda()
    t= np.zeros( (iters,1) )
    for i in np.arange(iters):
        # Feeding the impulse to the CNN
        with torch.no_grad():
            start_time = time.time()
            resp = cnn(im)
            t[i]= (time.time() - start_time)
    return([np.mean(t), np.std(t)])