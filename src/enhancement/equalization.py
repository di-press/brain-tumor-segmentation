import numpy as np

def histogram_equalization(image, number_of_bins=256):
    """
    Compute an image with an equalized histogram

    Args:
        image (ndarray): image
        number_of_bins (int): number of bins to use in the histogram.
    
    Returns:
        equalized, cumulative_histogram (ndarray): the equalized version of
        the input image and the cumulative histogram used to apply the transformation.
    """
    # Use numpy to compute normalized image histogram
    histogram, bins = np.histogram(image.flatten(), number_of_bins, density=True)
    
    # Compute the cumulative histogram
    cumulative_histogram = histogram.cumsum()
    cumulative_histogram = 255 * cumulative_histogram / cumulative_histogram[-1]

    # For each intensity value, transforms it into a new intensity using interpolation,
    # that is, computes what would be the output level 's' for an input value 'z'.
    equalized = np.interp(image.flatten(), bins[:-1], cumulative_histogram).reshape(image.shape)

    return equalized, cumulative_histogram

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import data
    from skimage import exposure

    image = data.moon()
    equalized, _ = histogram_equalization(image)
    skimage_equalized = exposure.equalize_hist(image)

    _, axes = plt.subplots(nrows=1, ncols=3)
    axes[0].hist(image, bins=32)
    axes[1].hist(equalized, bins=32)
    axes[2].hist(255.0*skimage_equalized, bins=32)
    plt.show()