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
    is_binary_array = lambda array: ((array == 0) | (array == 1)).all()
    assert is_binary_array(ground_truth_mask), f"Ground-truth segmentation is not a binary image"
    assert is_binary_array(predicted), f"Predicted segmentation is not a binary image"
    intersection = np.logical_and(predicted, ground_truth_mask)
    union = np.logical_or(predicted, ground_truth_mask)

    all_zeros_array = lambda array: not np.any(array)
    if all_zeros_array(union):
        assert all_zeros_array(ground_truth_mask)
        return 1.0
        
    return np.sum(intersection) / np.sum(union)

if __name__ == "__main__":
    from sklearn.metrics import jaccard_score
    
    mask = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    image = np.array([[0, 1, 0], [0, 1, 1], [1, 1, 1]], dtype=np.uint8)
    """
    Manual computation: 
    mask =          image =         intersection =      union = 
    [0, 0, 1]       [0, 1, 0]       [0, 0, 0]           [0, 1, 1]
    [0, 1, 1]       [0, 1, 1]       [0, 1, 1]           [0, 1, 1]
    [1, 1, 1]       [1, 1, 1]       [1, 1, 1]           [1, 1, 1]
    IOU = intersection / union = 5 / 7
    """
    skimage_iou = jaccard_score(mask, image, average="micro")
    iou_score = iou_similarity_score(mask, image)
    print(f"{iou_score}")
    assert skimage_iou == iou_score == 5/7

    for size in range(1, 101):
        mask = np.random.randint(low=0, high=2, size=(size, size))
        image = np.random.randint(low=0, high=2, size=(size, size))
        skimage_iou = jaccard_score(mask, image, average="micro")
        iou_score = iou_similarity_score(mask, image)
        assert skimage_iou == iou_score