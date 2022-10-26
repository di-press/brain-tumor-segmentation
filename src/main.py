from aenum import Enum, NoAlias
from functools import partial
import itertools
import matplotlib.pyplot as plt

from src.enhancement import histogram_equalization
from src.filtering import bilateral_filter, gaussian_filter, median_filter
from src.segmentation import stadlbauer_local_thresholding, otsu_global_thresholding, sobel_watershed
from src.util import Dataset, Pipeline

class Methods(Enum):
    _settings_ = NoAlias

    NO_ENHANCEMENT = None
    HISTOGRAM_EQUALIZATION = partial(histogram_equalization)
    NO_FILTER = None
    GAUSSIAN_FILTER = partial(gaussian_filter)
    MEDIAN_FILTER = partial(median_filter)
    BILATERAL_FILTER = partial(bilateral_filter)
    STADLBAUER_LOCAL_THRESHOLDING = partial(stadlbauer_local_thresholding)
    OTSU_GLOBAL_THRESHOLDING = partial(otsu_global_thresholding)
    SOBEL_WATERSHED = partial(sobel_watershed)
    NO_POST_PROCESSING = None
    OPENING = None # TODO: implement
    CLOSING = None # TODO:implement

def main():
    dataset = Dataset()

    pipeline_parameters = [
        [Methods.NO_ENHANCEMENT, Methods.HISTOGRAM_EQUALIZATION],
        [Methods.NO_FILTER, Methods.GAUSSIAN_FILTER, Methods.MEDIAN_FILTER, Methods.BILATERAL_FILTER],
        [Methods.STADLBAUER_LOCAL_THRESHOLDING, Methods.OTSU_GLOBAL_THRESHOLDING, Methods.SOBEL_WATERSHED],
        #[Methods.NO_POST_PROCESSING, Methods.OPENING, Methods.CLOSING],
        [Methods.NO_POST_PROCESSING],
    ]

    parameters_args={
        "gaussian_filter": {"kernel_size": 7, "sigma": 1.0},
        "bilateral_filter": {"kernel_size": 7, "spatial_sigma": 2.0},
        "median_filter": {"kernel_size": 7}
    }
    
    for idx, parameters in enumerate(itertools.product(*pipeline_parameters)):
        assert len(parameters) == 4, f"Unexpected number of parameters: expected 4, found {len(parameters)}"
        equalization, filtering, segmentation, post_processing = parameters
        pipeline = Pipeline(parameters={
            "equalization": equalization, 
            "filtering": filtering,
            "segmentation": segmentation,
            "post_processing": post_processing
            },
            parameters_args=parameters_args
        )
        predictions, iou_scores = pipeline.apply(dataset.flair_images, dataset.mask_images)

        for idy, (prediction, iou_score) in enumerate(zip(predictions, iou_scores)):
            print(f"Experimental configuration {idx}, Image {idy}: IoU Score = {iou_score}")
            plot_result(dataset.flair_images[idy], prediction, dataset.mask_images[idy], "Result", show=False)

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

if __name__ == "__main__":
    main()