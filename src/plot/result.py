import matplotlib.pyplot as plt

def plot_result(input, predicted, ground_truth_mask, title, show=True, save_path: str=None):
    """
    Plot FLAIR input image, predicted segmentation and ground-truth segmentation.
    """
    fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    fig.suptitle(title)
    ax[0].imshow(input, cmap="gray")
    ax[0].set_title('Image')
    ax[1].imshow(predicted, cmap="gray")
    ax[1].set_title('Segmented image')
    ax[2].imshow(ground_truth_mask, cmap="gray")
    ax[2].set_title("Mask")

    for a in ax:
        a.set_axis_off()

    fig.tight_layout()
    
    if show:
        plt.show()
    
    if save_path is not None:
        fig.savefig(save_path)
