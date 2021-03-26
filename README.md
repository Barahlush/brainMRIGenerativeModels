# Generative models for 3D MRI scans

![](https://sun9-37.userapi.com/impg/-0VOL6_KeUKyeLZD8J2ojziAVrCpRDmscqFreQ/UCDSN8ZxoC0.jpg?size=452x355&quality=96&sign=1cf79f52f9b0e07e557699e577167fb9&type=album)

This repo contains experiments on latent space exploration for detection of brain structure differences between genders and ages.

We also implemented several architectures of generative models for 3D brain MRI scans.<br> 
This is the first open-souce PyTorch realization of StyleGAN v1 for 3d models.

## Presented models: 
- Autoencoders:
- - Vanilla Autoencoder
- - Variational Autoencoder (VAE)
- GAN
- StyleGAN v1

## Data
The models were trained using [Human Connectome Young Adults](https://www.humanconnectome.org/study/hcp-young-adult/document/1200-subjects-data-release) brain sMRI scans.

**This does not contain the data, only code.** The data can be downloaded separately only via Human Connectome website. This process is **free**, but requires registration and confirmation from the Human Connectome organisation.

## Requisites
Python3, Pytorch 1.8+, Numpy, Matplotlib


# Experiment design

The idea is inspired by [[1]](#1), where the authors propose to train a logistic regression on StyleGAN's learned latent vectors and disease presence labels. The weights of the classifier can then be used as a direction of disease progression for the latent vector. It allows to predict the brain structure changes caused by e.g. aging, and prevent possible illnesses on early stages.

We train different generative models on 3D MRI scans and then try to explore their latent spaces using logistic regression weights as a direction in which we change the latent vectors in order to observe structural changes.


The models, their code, description and interactive jupyter notebooks can be found in the corresponding folders. The experiment and its results can be found in the experiment folder of the repository. 


## References
<a id="1">[1]</a> 
Kathryn Schutte and Olivier Moindrot and Paul Hérent and Jean-Baptiste Schiratti and Simon Jégou. (2021). 
Using StyleGAN for Visual Interpretability of Deep Learning Models on Medical Images. 
arXiv, 2101.07563.
