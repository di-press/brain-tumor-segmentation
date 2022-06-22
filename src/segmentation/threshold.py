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
    return segmentation

if name == "__main__":
    import matplotlib.pyplot as plt
    from skimage import data
    from skimage import restoration

    image = data.moon()
    _, axes = plt.subplots(nrows=1, ncols=2)
    axes[0].imshow(image, cmap="gray")
    axes[1].imshow(stadlbauer_local_thresholding(image), cmap="gray")
    plt.show()