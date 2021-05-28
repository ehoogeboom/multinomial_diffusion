# Code for Multinomial Diffusion

![Banner](https://github.com/ehoogeboom/multinomial_diffusion/blob/main/images/overview_mult_diff.png?raw=true)


## Abstract
Generative flows and diffusion models have been predominantly trained on ordinal data, for example natural images. This paper introduces two extensions of flows and diffusion for categorical data such as language or image segmentation: Argmax Flows and Multinomial Diffusion. Argmax Flows are defined by a composition of a continuous distribution (such as a normalizing flow), and an argmax function. To optimize this model, we learn a probabilistic inverse for the argmax that lifts the categorical data to a continuous space. Multinomial Diffusion gradually adds categorical noise in a diffusion process, for which the generative denoising process is learned. We demonstrate that our method outperforms existing dequantization approaches on text modelling and modelling on image segmentation maps in log-likelihood.

Link: https://arxiv.org/abs/2102.05379

## Instructions
In the folder containing `setup.py`, run
```
pip install --user -e .
```
The `--user` option ensures the library will only be installed for your user.
The `-e` option makes it possible to modify the library, and modifications will be loaded on the fly.

You should now be able to use it.


## Running experiments.

Go to the experiment of interest (folder segmentation_diffusion or text_diffusion) and follow the readme instructions there.


## Acknowledgements
The Robert Bosch GmbH is acknowledged for financial support.