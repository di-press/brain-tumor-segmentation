import numpy as np
import pathlib
from skimage import io

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
        assert set(self.flair_filenames).isdisjoint(set(self.mask_filenames))
        assert set(self.flair_filenames + self.mask_filenames) == set(self.filenames)

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