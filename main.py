import pathlib
from random import sample

def main():
    dataset_path = pathlib.Path("dataset", "brats_subset")
    sample_image = dataset_path / "BraTS20_Training_094_flair.nii"
    sample_mask = dataset_path / "BraTS20_Training_094_seg.nii"
    processed_image = preprocessing(sample_image)
    predicted_segmentation = segmentation(processed_image)
    iou_value = evaluation(predicted_segmentation, sample_mask)
    show_result(sample_image, predicted_segmentation, sample_mask)

def preprocessing(image):
    """
    Placeholder for preprocessing
    TODO: implement image enhancement and image filtering
    """
    pass

def segmentation(image):
    """
    Placeholder for segmentation
    TODO: implement one image segmentation algorithm
    """
    pass

def evaluation(predicted, ground_truth_mask):
    """
    Placeholder for image segmentation evaluation.
    TODO: implement IoU metrics.
    """
    pass

def show_result(input, predicted, ground_truth_mask):
    """
    Placeholder for results visualization.
    TODO: show segmentation results for visual comparison.
    """
    pass

if __name__ == "__main__":
    main()