# Demo of the paper "Noise reduction in CT using Learned Wavelet-Frame Shrinkage Networks

This repository contains replicates some of the relevant resolus shown in the paper "Noise reduction in CT using Learned Wavelet-Frame Shrinkage Networks" which is in review process at IEEE's Transactions on medical imaging. The things to be demonstrated are the following:

1. An overview of the noise reduction performance of the LWFSN, rLWFSN, tight frame (TF) U-Net and FBPConvNet
2. A demonstration of the impulse response of the LWFSN and rLWFSN
3. Showing that the LWFSN is capable of reconstructing a given noisy image if the soft-shrinkage section is disabled


ATTENTION

At the moment the weights of FBPConvNet and the tight frame U-Net are not uploaded because the file-size is beyond what's allowed by gitHub. We are working out a solution to provide the weights in another way.

The implementation of FBPConvNet was pereformed by adapting the U-Net implementation from the repository: https://github.com/jvanvugt/pytorch-unet


ACKNOWLEDGEMENTS

The funding of this project was provided by the European Union through the Horizon 2020 “Next generation X-ray imaging system (NEXIS)]” under Grant 780026.

The CT slices in this repository were obtained from the low-dose dataset from the cancer imaging archive (TCIA). We thank the grants EB017095 and EB017185 from the National Institute of Biomedical Imaging and Bioengineeringto to provide funding for the generation of the dataset used in this paper.
