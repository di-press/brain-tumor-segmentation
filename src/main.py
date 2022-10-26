from aenum import Enum, NoAlias
from functools import partial
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pathlib
from skimage import io

from src.enhancement.equalization import histogram_equalization
from src.filtering.gaussian import gaussian_filter
from src.filtering.bilateral import bilateral_filter
from src.filtering.median import median_filter
from src.segmentation.watershed import sobel_watershed
from src.segmentation.threshold import otsu_global_thresholding, stadlbauer_local_thresholding
from src.evaluation.iou import iou_similarity_score

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
            plot_result(dataset.flair_images[idy], prediction, dataset.mask_images[idy], "Result")

class Dataset:
    def __init__(self, slice: int=78, dataset_path: pathlib.Path=pathlib.Path("dataset", "brats_subset")):
        self.dataset_path = dataset_path
        self.filenames = ["BraTS20_Training_064_flair.nii", "BraTS20_Training_064_seg.nii"]
        #self.filenames = ["BraTS20_Training_112_flair.nii", "BraTS20_Training_112_seg.nii"]
        ##self.filenames = ["BraTS20_Training_327_flair.nii", "BraTS20_Training_327_seg.nii"]
        #self.filenames = ["BraTS20_Training_234_flair.nii", "BraTS20_Training_234_seg.nii"]
        #self.filenames = self.dataset_path.rglob("*.nii")
        self.flair_filenames = [filename for filename in self.filenames if "flair" in str(filename)]
        self.mask_filenames = [filename for filename in self.filenames if "seg" in str(filename)]
        
        # Read and normalize FLAIR images
        self.flair_images = []
        for flair_file in self.flair_filenames:
            image = io.imread(dataset_path / flair_file)
            image = image[slice, :, :]
            normalized_image = (image - np.min(image)) / (np.max(image) - np.min(image))
            self.flair_images.append((255.0 * normalized_image).astype(np.uint8))

        # Binarize segmentation masks (1 is tumoral tissue, 0 is not tumoral tissue)
        self.mask_images = []
        for mask_file in self.mask_filenames:
            mask = io.imread(dataset_path / mask_file)[slice, :, :]
            binary_mask = np.zeros_like(mask)
            binary_mask[mask != 0] = 1
            self.mask_images.append(binary_mask)

        assert len(self.flair_images) == len(self.mask_images)

class Pipeline:
    def __init__(self, parameters: dict, parameters_args: dict=None):
        self._steps = []
        self._args = []
        for step in ("equalization", "filtering", "segmentation", "post_processing"):
            function = parameters.get(step, None)
            if function.value is None:
                continue
            self._steps.append(function.value)
            self._args.append(parameters_args.get(function.name.lower(), None))

    def apply(self, input_images, ground_truth_images):
        images = input_images
        for function, args in zip(self._steps, self._args):
            new_images = []
            for image in images:
                if args is not None:
                    result = function(image, **args)
                else:
                    result = function(image)

                if isinstance(result, tuple):
                    new_images.append(result[0])
                else:
                    new_images.append(result)
            images = new_images
        
        predictions = [prediction for prediction in images]
        iou_score = [iou_similarity_score(ground_truth, prediction) for ground_truth, prediction in zip(ground_truth_images, predictions)]
        return predictions, iou_score

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