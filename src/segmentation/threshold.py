from tkinter import image_types
import numpy as np

def stadlbauer_local_thresholding(image):
    """
    Applies the local thresholding algorithm as proposed by
    Stadlbauer et al (see [1, 2] for details).

    Ref:
    [1] Stadlbauer, Andreas, et al. "Improved delineation of brain
    tumors: an automated method for segmentation based on pathologic
    changes of 1H-MRSI metabolites in gliomas." 
    Neuroimage 23.2 (2004): 454-461.

    [2] GORDILLO, Nelly; MONTSENY, Eduard; SOBREVILLA, Pilar. State of the
    art survey on MRI brain tumor segmentation. Magnetic resonance
    imaging, v. 31, n. 8, p. 1426-1438, 2013.

    Args:
        image (ndarray): image to be segmented.
    
    Returns:
        segmentation (ndarray): the predicted segmentation mask.
    """
    local_threshold = np.mean(image) + (3 * np.std(image))
    segmentation = (image > local_threshold)
    segmentation = segmentation.astype(np.uint8)
    return segmentation



def otsu_global_thresholding(image, max_intensity=255):
    """
    Applies the Otsu's thresholding method. This is an example of
    global thresholding method adapted from the course material.

    Args:
        image (ndarray): image to be segmented.
        max_intensity (int): maximum value of the range of the input image.
    
    Returns:
        segmentation, global_optimum_value (ndarray, int): the predicted segmentation mask and
        the global optimum value computed by the Otsu's method.
    """
    size = np.product(image.shape)
    weighted_variances = []
    histogram, _ = np.histogram(image, bins=256, range=(0, 256))
    image_iteration = (image >= 0).astype(np.uint8)
    
    for intensity in np.arange(1, max_intensity):
        image_iteration = (image >= intensity).astype(np.uint8)
        weight_a = np.sum(histogram[:intensity]) / float(size)
        weight_b = np.sum(histogram[intensity:]) / float(size)
        variance_a = np.var(image[np.where(image_iteration == 0)])
        variance_b = np.var(image[np.where(image_iteration == 1)])
        weighted_variances = weighted_variances + [weight_a * variance_a + weight_b * variance_b]
    global_optimum_value = np.argmin(weighted_variances)
    segmentation = (image >= global_optimum_value).astype(np.uint8)
    return segmentation, global_optimum_value


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import data
    from skimage import restoration

    image = data.moon()
    _, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(image, cmap="gray")
    axes[1].imshow(stadlbauer_local_thresholding(image), cmap="gray")
    plt.show()