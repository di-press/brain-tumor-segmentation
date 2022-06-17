import numpy as np

def histogram_equalization(image, number_bins=256):
    # Use numpy to compute normalized image histogram
    histogram, bins = np.histogram(image.flatten(), number_bins, density=True)
    
    # Compute the cumulative histogram
    cumulative_histogram = histogram.cumsum()
    cumulative_histogram = 255 * cumulative_histogram / cumulative_histogram[-1]

    # For each intensity value, transforms it into a new intensity using interpolation,
    # that is, computes what would be the output level 's' for an input value 'z'.
    equalized = np.interp(image.flatten(), bins[:-1], cumulative_histogram).reshape(image.shape)

    return equalized, cumulative_histogram