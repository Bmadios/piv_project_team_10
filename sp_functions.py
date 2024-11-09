# This files containg all signal processing functions used in the project

import numpy as np
from scipy.ndimage import gaussian_filter as gf
from skimage.restoration import denoise_wavelet


# Function to apply gaussian filter to the image
def gaussian_filter(image, sigma):
    """
    This function applies a gaussian filter to the image.
    It's based on the Gaussian (or normal) distribution, a bell-shaped curve, which gives the filter its name.

    Here's a breakdown of how it works:

    Gaussian Distribution: The filter uses a Gaussian function (bell curve) to assign weights to each pixel or data point in the kernel.
    The central point (the "mean") has the highest weight, and the weights decrease symmetrically as you move further from the center.

    Convolution Process: To apply the filter, you take a "kernel" (a small matrix) shaped according to the Gaussian distribution, then "convolve" it over the image or signal.
    This means you slide the kernel across each pixel or point, calculate a weighted sum based on nearby values, and replace the center value with this sum.
    The result is a smoothing effect that blurs or reduces sharp edges and noise.

    :param image:   The image to be filtered
    :param sigma:  The standard deviation of the gaussian filter
    :return:
    """
    return gf(image, sigma)


# Function to apply wavelet denoising to the image
def wavelet_denoising(image):
    """
    This function applies wavelet denoising to the image.
    Wavelet denoising is a method of reducing noise in an image by applying wavelet transforms and thresholding.

    Here's how it works:

    Wavelet Transform: The image is decomposed into wavelet coefficients using a wavelet transform.
    This transform separates the image into different frequency bands, with high-frequency bands capturing details and noise.

    Thresholding: The wavelet coefficients are thresholded to remove noise.
    Coefficients below a certain threshold are set to zero, effectively removing noise while preserving important image features.

    Inverse Transform: The denoised wavelet coefficients are then used to reconstruct the image using an inverse wavelet transform.

    :param image: The image to be denoised
    :return:
    """
    return denoise_wavelet(image, channel_axis=None)

