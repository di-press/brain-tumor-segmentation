# MRI Brain Tumor Segmentation

### Students involved:

- Diany Pressato
- Matheus da Silva Araujo
- Maurilio da Motta Meireles
- Yure Pablo do Nascimento

## Abstract

As stated by <cite><a href="https://www.sciencedirect.com/science/article/abs/pii/S0730725X13001872">Nelly Gordillo et. al</a></cite> [1]:

<em>" The main
objective of image segmentation is to partition an image into
mutually exclusive regions such that each region is spatially
contiguous and the pixels within the region are homogeneous with
respect to a predefined criterion. This definition is in itself a major
limitation of most of the segmentation methods, especially when
defining and delineating “abnormal tissue types”, because the
tumors to be segmented are anatomical structures which are often
non-rigid and complex in shape, vary greatly in size and position,
and exhibit considerable variability from patient to patient. " </em>


### Objectives

This project aims to segment brain tumor from magnetic resonance imaging (MRI) scans. This task involves separating the tumoral tissue from the normal gray matter and white matter tissues of the brain. 

The simplicity of the segmentation method and the degree of human supervision of the task is important for clicinal acceptance in the field - for this reason, this project focuses more on interpretable and simple methods of segmentation than emphasizing "black box" techniques (such as Deep Learning). 

### Application

The application of this project is in medical imaging domain.


### Dataset

The chosen dataset is <cite><a href="https://www.med.upenn.edu/cbica/brats2020/data.html">BraTS2020</a></cite> [2, 3, 4, 5, 6, 7]. It is divided into three separate cohorts: Training, Validation, and Testing. The Training dataset
is composed of multi-parametric MRI (mpMRI) scans from 369 diffuse glioma patients; glioma is considered a type of tumor. Each MRI volume is skull-stripped (the skull was extracted), and there are annotated masks for the tumors. The ground truth labels (masks) were provided by expert human annotators. 

The BraTS 2020 Validation cohort is composed of 125 cases of patients with diffuse
gliomas, and it is similar to the Training Dataset. The ground truth labels for the validation data are not provided.

For now, we don't plan to use the Testing dataset in this project.

### Image Processing Tasks

According to [], the following segmentation methods can be applied to MRI Brain TUmor Segmentation task:

- K-Means
- Watershed
- Region Growing
- Thresholding (Global or Local)



### Input/Output Examples

Input Image                |  Output Image
:-------------------------:|:-------------------------:
![]()  |  ![]()

### Metrics

### References

[1] Nelly Gordillo, Eduard Montseny, Pilar Sobrevilla,
State of the art survey on MRI brain tumor segmentation,
Magnetic Resonance Imaging,
Volume 31, Issue 8,
2013,
Pages 1426-1438,
ISSN 0730-725X,
https://doi.org/10.1016/j.mri.2013.05.002.

[2]https://www.med.upenn.edu/cbica/brats2020/data.html

[3] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694

[4] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117

[5] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)

[6] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-GBM collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.KLXWJJ1Q

[7] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J. Kirby, et al., "Segmentation Labels and Radiomic Features for the Pre-operative Scans of the TCGA-LGG collection", The Cancer Imaging Archive, 2017. DOI: 10.7937/K9/TCIA.2017.GJQ7R0EF