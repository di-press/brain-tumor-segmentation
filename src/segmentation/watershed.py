import numpy as np
from skimage.segmentation import watershed
from skimage.filters import sobel

def sobel_watershed(image, background_threshold=30, object_threshold=150):
    """
    Applies the Watershed segmentation algorithm using the
    output of Sobel filter as image gradient.

    Args:
        image (ndarray): image to be segmented.
        background_threshold, object_threshold (int): threshold values
        to be used in the markers.
    
    Returns:
        segmentation, image_gradient (ndarray): the predicted segmentation
        mask and the image gradient used in the Watershed segmentation.
    """
    markers = np.zeros_like(image)
    markers[image < background_threshold] = 1
    markers[image > object_threshold] = 2
    image_gradient = sobel(image)
    segmentation = watershed(image_gradient, markers)
    segmentation = segmentation > 1
    return segmentation, image_gradient

if __name__ == "__main__":
    from skimage import data
    import matplotlib.pyplot as plt

    image = data.camera()
    result, _ = sobel_watershed(image)
    _, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(image, cmap="gray")
    axes[1].imshow(result, cmap="gray")
    plt.show()
    