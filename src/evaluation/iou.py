import numpy as np

def iou_similarity_score(ground_truth_mask, predicted):
    """
    Compute the Intersection Over Union similarity score.

    Args:
        ground_truth_mask (ndarray): matrix containing the ground truth segmentation.
        predicted (ndarray): matrix containing the predicted segmentation.
    
    Returns:
        (float): the similarity score between the ground truth and the prediction,
        computed as the the area of intersection divided by the area of union.
    """
    intersection = np.logical_and(predicted, ground_truth_mask)
    union = np.logical_or(predicted, ground_truth_mask)
    return np.sum(intersection) / np.sum(union)