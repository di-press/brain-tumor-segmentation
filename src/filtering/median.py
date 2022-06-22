import numpy as np

def median_filter(image, kernel_size):
    """
    Apply a median filter on an image i.e. replace each pixel with the
    median of the neighboring pixels.
    
    Args:
        image (ndrray): image where the filtering will be applied.
        filter_size (int): size of the median filter.
    Returns:
        processed_image (ndarray): the resulting image after applying
        the median filter on the image.
    """
    kernel_radius = kernel_size // 2
    pad_image = np.pad(image, pad_width=kernel_radius, mode="constant")
    
    # For each pixel, store it's neighborhood as a one-dimensional vector (row vector)
    # on a list.
    sliding_windows = []
    for idx in range(image.shape[0]):
        for idy in range(image.shape[1]):
            sliding_windows.append(pad_image[idx:idx+kernel_size, idy:idy+kernel_size].flatten())
    sliding_windows = np.asarray(sliding_windows)

    # Each row of sliding_windows stores the flattened neighborhood of each pixel.
    # Hence, the median of each row is the median of the neighborhood of each pixel.
    # By reshaping the matrix to image.shape, filtered_image[x, y] stores the median
    # of the neighborhood of the pixel image[x, y].
    processed_image = np.median(sliding_windows, axis=1).reshape(image.shape)
    
    return processed_image

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import data
    from skimage.color import rgb2gray
    from skimage.util import random_noise
    from scipy import ndimage

    image = rgb2gray(data.coffee())
    image = image + random_noise(image, mode="s&p")
    median = median_filter(image, kernel_size=5)
    scipy_median = ndimage.median_filter(image, size=5)

    _, axes = plt.subplots(nrows=2, ncols=2)
    axes[0, 0].imshow(image, cmap="gray")
    axes[0, 0].set_title("Noise")
    axes[0, 1].imshow(median, cmap="gray")
    axes[0, 1].set_title("Median")
    axes[1, 0].imshow(scipy_median, cmap="gray")
    axes[1, 0].set_title("SciPy Median")
    axes[1, 1].imshow(np.abs(scipy_median - median), cmap="gray")
    axes[1, 1].set_title("Difference")
    plt.show()