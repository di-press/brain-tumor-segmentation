from collections import deque
from random import random
import numpy as np
import matplotlib.pyplot as plt

import skimage
from skimage import io
from src.filtering.gaussian import gaussian_filter
from src.filtering.bilateral import bilateral_filter
from src.filtering.median import median_filter
from src.segmentation.watershed import sobel_watershed
from src.segmentation.threshold import otsu_global_thresholding, stadlbauer_local_thresholding
from src.evaluation.iou import iou_similarity_score

names = [
    ("dataset/brats_subset/BraTS20_Training_064_flair.nii", "dataset/brats_subset/BraTS20_Training_064_seg.nii", 64),
    ("dataset/brats_subset/BraTS20_Training_112_flair.nii", "dataset/brats_subset/BraTS20_Training_112_seg.nii",112),
    ("dataset/brats_subset/BraTS20_Training_327_flair.nii", "dataset/brats_subset/BraTS20_Training_327_seg.nii", 327),
    ("dataset/brats_subset/BraTS20_Training_234_flair.nii", "dataset/brats_subset/BraTS20_Training_234_seg.nii", 234)
]

flair, mask, num = names[2]
#algo, filtername = "Otsu's Thresholding", "No filter"
#algo, filtername = "Otsu's Thresholding", "Gaussian"
#algo, filtername = "Otsu's Thresholding", "Bilateral"
#algo, filtername = "Otsu's Thresholding", "Median"

#algo, filtername = "Stadlbauer Thresholding", "No filter"
#algo, filtername = "Stadlbauer Thresholding", "Gaussian"
#algo, filtername = "Stadlbauer Thresholding", "Bilateral"
#algo, filtername = "Stadlbauer Thresholding", "Median"

#algo, filtername = "Watershed", "No filter"
#algo, filtername = "Watershed", "Gaussian"
#algo, filtername = "Watershed", "Bilateral"
algo, filtername = "Watershed", "Median"

gt = gt[78, :, :]
binary = np.zeros_like(gt)
binary[gt != 0] = 1
gt = binary
image = image[78, :, :]
nimage = (image - np.min(image)) / (np.max(image) - np.min(image))
image = (255.0*nimage).astype(np.uint8)

#image = gaussian_filter(image, kernel_size=7, sigma=1.0)
#image = bilateral_filter(image, kernel_size=7, spatial_sigma=2.0)
#image = median_filter(image, kernel_size=7)

#segmented_image, _ = otsu_global_thresholding(image)
#segmented_image = stadlbauer_local_thresholding(image)
segmented_image, _ = sobel_watershed(image)

iou_val = iou_similarity_score(gt, segmented_image)

fig, axes = plt.subplots(ncols=3, figsize=(9, 3), sharex=True, sharey=True)
ax = axes.ravel()

fig.suptitle(f"Patient {num}: {algo} with {filtername}; IOU: {iou_val:.3f}")
ax[0].imshow(image, cmap="gray")
ax[0].set_title('Image')
ax[1].imshow(segmented_image, cmap="gray")
ax[1].set_title('Segmented image')
ax[2].imshow(gt, cmap="gray")
ax[2].set_title("Mask")

for a in ax:
    a.set_axis_off()

fig.tight_layout()
plt.show()
#fig.savefig("otsu_nofilter.png")
#fig.savefig("otsu_gaussian.png")
#fig.savefig("otsu_bilateral.png")
#fig.savefig("otsu_median.png")

#fig.savefig("stadlbauer_nofilter.png")
#fig.savefig("stadlbauer_gaussian.png")
#fig.savefig("stadlbauer_bilateral.png")
#fig.savefig("stadlbauer_median.png")

fig.savefig(f"watershed_nofilter_{num}.png")
#fig.savefig("watershed_gaussian.png")
#fig.savefig("watershed_bilateral.png")
#fig.savefig("watershed_median.png")


print(iou_val)