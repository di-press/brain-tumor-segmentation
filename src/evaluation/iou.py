import numpy as np

def iou_similarity_score(ground_truth_mask, predicted):
    intersection = np.logical_and(predicted, ground_truth_mask)
    union = np.logical_or(predicted, ground_truth_mask)
    return np.sum(intersection) / np.sum(union)