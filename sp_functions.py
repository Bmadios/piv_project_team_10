import numpy as np
from scipy.ndimage import gaussian_filter, median_filter
from skimage.restoration import denoise_wavelet, denoise_tv_chambolle, denoise_bilateral, denoise_nl_means
from skimage.filters import threshold_local
from bm3d import bm3d

def denoise_piv_images(inputs_data, method='gaussian', **kwargs):
    """
    Apply denoising to PIV image data using the specified method.

    Parameters:
    - inputs_data: numpy array of images, shape can be:
        (num_samples, height, width) or
        (num_samples, channels, height, width) or
        (num_samples, channels, num_images_per_sample, height, width)
    - method: string, specifying the denoising method to use 
              ('gaussian', 'median', 'wavelet', 'non_local_means', 
              'bilateral', 'fft', 'tv', 'anisotropic_diffusion', 'bm3d')
    - kwargs: additional parameters for the denoising function

    Returns:
    - denoised_data: numpy array of denoised images
    """
    inputs_shape = inputs_data.shape
    num_dims = inputs_data.ndim

    def apply_denoising(image, method, **kwargs):
        if method == 'gaussian':
            sigma = kwargs.get('sigma', 1)
            denoised_image = gaussian_filter(image, sigma=sigma)
        elif method == 'median':
            size = kwargs.get('size', 3)
            denoised_image = median_filter(image, size=size)
        elif method == 'wavelet':
            wavelet = kwargs.get('wavelet', 'db1')
            mode = kwargs.get('mode', 'soft')
            channel_axis = kwargs.get('channel_axis', None)
            denoised_image = denoise_wavelet(image, wavelet=wavelet, mode=mode, channel_axis=channel_axis)
        elif method == 'non_local_means':
            h = kwargs.get('h', 1.15)
            fast_mode = kwargs.get('fast_mode', True)
            patch_size = kwargs.get('patch_size', 5)
            patch_distance = kwargs.get('patch_distance', 6)
            channel_axis = kwargs.get('channel_axis', None)
            denoised_image = denoise_nl_means(image, h=h, fast_mode=fast_mode, 
                                              patch_size=patch_size, patch_distance=patch_distance, channel_axis=channel_axis)
        elif method == 'bilateral':
            sigma_color = kwargs.get('sigma_color', 0.05)
            sigma_spatial = kwargs.get('sigma_spatial', 15)
            channel_axis = kwargs.get('channel_axis', None)
            denoised_image = denoise_bilateral(image, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=channel_axis)
        elif method == 'fft':
            threshold = kwargs.get('threshold', 0.1)
            fft_image = np.fft.fft2(image)
            fft_shift = np.fft.fftshift(fft_image)
            magnitude = np.abs(fft_shift)
            mask = magnitude > (threshold * np.max(magnitude))
            fft_shift_filtered = fft_shift * mask
            fft_filtered = np.fft.ifftshift(fft_shift_filtered)
            denoised_image = np.abs(np.fft.ifft2(fft_filtered))
        elif method == 'tv':
            weight = kwargs.get('weight', 0.1)
            channel_axis = kwargs.get('channel_axis', None)
            denoised_image = denoise_tv_chambolle(image, weight=weight, channel_axis=channel_axis)
        elif method == 'anisotropic_diffusion':
            iterations = kwargs.get('iterations', 10)
            kappa = kwargs.get('kappa', 50)
            step = kwargs.get('step', 0.1)
            denoised_image = image.copy()
            for _ in range(iterations):
                nabla = np.gradient(denoised_image)
                diff_coeff = np.exp(-(np.sum([g**2 for g in nabla], axis=0)) / (kappa**2))
                denoised_image += step * np.sum([d * diff for d, diff in zip(diff_coeff, nabla)], axis=0)
        elif method == 'bm3d':
            sigma_psd = kwargs.get('sigma_psd', 0.2)
            denoised_image = bm3d(image, sigma_psd)
        else:
            raise ValueError(f"Unknown denoising method: {method}")
        return denoised_image

    #if num_dims in {3, 4, 5}:  # Multiple images or sequences
        #denoised_data = np.array([apply_denoising(image, method, **kwargs) for image in inputs_data])
    #else:  # Single image
        #denoised_data = apply_denoising(inputs_data, method, **kwargs)
    if num_dims == 3:
        # inputs_data shape: (num_samples, height, width)
        denoised_data = np.empty_like(inputs_data)
        num_samples = inputs_data.shape[0]
        for i in range(num_samples):
            image = inputs_data[i]
            denoised_image = apply_denoising(image, method, **kwargs)
            denoised_data[i] = denoised_image
    elif num_dims == 4:
        # inputs_data shape: (num_samples, channels, height, width)
        denoised_data = np.empty_like(inputs_data)
        num_samples, num_channels = inputs_data.shape[:2]
        for i in range(num_samples):
            for c in range(num_channels):
                image = inputs_data[i, c]
                denoised_image = apply_denoising(image, method, **kwargs)
                denoised_data[i, c] = denoised_image
    elif num_dims == 5:
        # inputs_data shape: (num_samples, channels, num_images_per_sample, height, width)
        denoised_data = np.empty_like(inputs_data)
        num_samples, num_channels, num_images = inputs_data.shape[:3]
        for i in range(num_samples):
            print(f"Denoising in Progress : Img. {i}")
            for c in range(num_channels):
                for j in range(num_images):
                    image = inputs_data[i, c, j]
                    denoised_image = apply_denoising(image, method, **kwargs)
                    denoised_data[i, c, j] = denoised_image
    else:
        raise ValueError(f"Unsupported input data dimensions: {num_dims}")

    return denoised_data
