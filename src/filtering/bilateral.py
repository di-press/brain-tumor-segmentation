from statistics import mean
import numpy as np


def bilateral_filter(image, kernel_size=None, spatial_sigma=1.0, intensity_sigma=None):
    """
    Apply the bilateral filter to the input image.

    Args:
        image (ndarray): input image to be filtered
        kernel_size (int): size of the filter. If None, this value is set to
        max(5, 2 * ceil(3 * spatial_sigma) + 1) to replicate the
        behaviour of skimage denoising_bilateral.
        spatial_sigma (float): standard deviation for the spatial Gaussian.
        intensity_sigma (float): standard deviation for the intensity Gaussian.
        If None, the value is set to image.std() to replicate the behaviour of
        skimage denoising_bilateral.
    
    Returns:
        processed_image (ndarray): the image after applying the bilateral filter.
    """
    # Set the values of the arguments that store None
    if kernel_size is None:
        kernel_size = np.max(5, (2 * np.ceil(3 * spatial_sigma)) + 1)
    if intensity_sigma is None:
        intensity_sigma = image.std()

    # Pad the input image
    kernel_radius = kernel_size // 2
    image = image.astype(np.float32)
    pad_image = np.pad(image, pad_width=kernel_radius, mode="constant")
    processed_image = np.zeros_like(pad_image)
    rows, columns = image.shape

    for i in range(kernel_radius, rows+kernel_radius):
        for j in range(kernel_radius, columns+kernel_radius):
            weighted_pixels = 0.0
            weights_sum = 0.0
            current_pixel = pad_image[i, j]

            # Compute the filtered value on the window o
            for m in range(-kernel_radius, kernel_radius+1):
                for n in range(-kernel_radius, kernel_radius+1):
                    shifted_pixel = pad_image[i + m, j + n]
                    spatial_weight = np.exp(-0.5 * (m**2 + n**2) / (spatial_sigma**2))
                    intensity_weight = np.exp(-0.5 * (current_pixel - shifted_pixel)**2 / (intensity_sigma**2))
                    weight = spatial_weight * intensity_weight
                    weighted_pixels += weight * shifted_pixel
                    weights_sum += weight

            processed_image[i, j] = weighted_pixels / weights_sum
    return processed_image[kernel_radius:rows+kernel_radius, kernel_radius:columns+kernel_radius]

if __name__ == "__main__":
    from skimage import io
    import matplotlib.pyplot as plt
    from skimage.filters import gaussian
    from skimage.restoration import denoise_bilateral
    from skimage.util import random_noise
    from sklearn.metrics import mean_squared_error

    image = io.imread("dataset/test/lena.jpeg", as_gray=True).astype(np.float32)
    noisy_image = image + random_noise(image)
    answer = bilateral_filter(noisy_image, kernel_size=5)
    skimage_answer = denoise_bilateral(noisy_image, win_size=5)
    _, axes = plt.subplots(nrows=2, ncols=3)
    axes[0, 0].imshow(noisy_image, cmap="gray")
    axes[0, 1].imshow(skimage_answer, cmap="gray")
    axes[0, 2].imshow(answer, cmap="gray")

    gaussian_blurred = gaussian(noisy_image)
    axes[1, 0].imshow(noisy_image, cmap="gray")
    axes[1, 1].imshow(gaussian_blurred, cmap="gray")
    axes[1, 2].imshow(answer, cmap="gray")
    plt.show()
    
    print(f"RMSE(input, bilateral) =\t\t{mean_squared_error(image, answer, squared=False)}")
    print(f"RMSE(input, sk_bilateral) =\t\t{mean_squared_error(image, skimage_answer, squared=False)}")
    print(f"RMSE(input, gaussian)=\t\t{mean_squared_error(image, gaussian_blurred, squared=False)}")