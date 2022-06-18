import math
import numpy as np

def gaussian_filter(image, kernel_size, sigma: float, on_frequency_domain=False):
    """
    Apply the gaussian filter to the input image

    Args:
        image (ndarray): image to filter
        kernel_size (int): size of the filter.
        sigma (float): standard deviation of the Gaussian function.
        on_frequency_domain (bool): if True, the filtering happens on the
        frequency domain using the Fourier Transform. The returned image
        is transformed to the spatial domain using the Inverse Fourier
        Transform. Default is False.
    
    Returns:
        processed_image (ndarray): the image after applying the Gaussian filter.
    """
    arx = np.arange((-kernel_size // 2) + 1.0, (kernel_size // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    gauss_filter = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))
    gauss_filter =  gauss_filter / np.sum(gauss_filter)

    if on_frequency_domain:
        pad_size = image.shape[0] - kernel_size
        pad_shape = (math.floor(pad_size / 2), math.ceil(pad_size / 2))
        gauss_filter = np.pad(gauss_filter, pad_shape, mode="constant", constant_values=0)
        fourier_gauss_filter = np.fft.fft2(np.fft.ifftshift(gauss_filter))
        fourier_image = np.fft.fft2(image)
        fourier_result = np.multiply(fourier_image, fourier_gauss_filter)
        processed_image = np.real(np.fft.ifft2(fourier_result))
    else:
        kernel_radius = kernel_size // 2
        pad_image = np.pad(image, kernel_radius, mode="constant", constant_values=0)
        processed_image = np.zeros_like(image)
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                processed_image[i, j] = np.sum(pad_image[i:i+kernel_size, j:j+kernel_size] * gauss_filter)
    
    return processed_image

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import data
    from skimage.color import rgb2gray
    from skimage.filters import gaussian
    from skimage.transform import resize
    
    image = resize(rgb2gray(data.cat()), (300, 300))
    # skimage.gaussian don't have a window size parameter;
    # instead it uses the truncate parameter https://stackoverflow.com/a/43617491
    sigma = 0.5
    truncate = 3
    window_size = 5
    skimage_blur = gaussian(image, sigma=sigma, truncate=truncate, mode="constant")
    blur = gaussian_filter(image, kernel_size=window_size, sigma=sigma)
    fourier_blur = gaussian_filter(image, kernel_size=window_size, sigma=sigma, on_frequency_domain=True)

    _, axes = plt.subplots(nrows=3, ncols=3)
    axes[0, 0].imshow(skimage_blur, cmap="gray")
    axes[0, 1].imshow(blur, cmap="gray")
    axes[0, 2].imshow(np.abs(blur - skimage_blur), cmap="gray")
    
    axes[1, 0].imshow(skimage_blur, cmap="gray")
    axes[1, 1].imshow(fourier_blur, cmap="gray")
    axes[1, 2].imshow(np.abs(fourier_blur - skimage_blur), cmap="gray")

    axes[2, 0].imshow(blur, cmap="gray")
    axes[2, 1].imshow(fourier_blur, cmap="gray")
    axes[2, 2].imshow(np.abs(blur - fourier_blur), cmap="gray")

    plt.show()