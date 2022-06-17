import numpy as np 

def gaussian_filter(size_degradation_filter: ndarray, sigma: float) -> ndarray:
    ''' Gaussian filter
    :param size_degradation_filter: defines the lateral size of the kernel/filter, default 5
    :param sigma: standard deviation (dispersion) of the Gaussian distribution
    :return matrix with a filter [size_degradation_filter x size_degradation_filter] to be used in convolution operations
    '''
    arx = np.arange((-size_degradation_filter // 2) + 1.0, (size_degradation_filter // 2) + 1.0)
    x, y = np.meshgrid(arx, arx)
    filt = np.exp(-(1/2) * (np.square(x) + np.square(y)) / np.square(sigma))

    return filt / np.sum(filt)