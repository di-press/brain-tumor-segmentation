import numpy as np

def point_set(array):
    """
    Returns the set of indices where the array is greater than zero
    """
    indices = np.transpose((array > 0).nonzero()).tolist()
    return {tuple(index) for index in indices}

def complement_point_set(array):
    """
    Returns the set of indices where the array is lesser or equal to zero
    """
    indices = np.transpose((array <= 0).nonzero()).tolist()
    return {tuple(index) for index in indices}

def translate_point_set(point_set, vector: tuple):
    return {(row + vector[0], column + vector[1]) for row, column in point_set}

def reflect_point_set(point_set):
    return {(-row, -column) for row, column in point_set}

if __name__ == "__main__":
    array = np.ones((3, 3))
    indices = point_set(array)
    complement_indices = complement_point_set(array)
    assert indices.isdisjoint(complement_indices)
    for i in range(3):
        for j in range(3):
            assert (i, j) in indices
            assert (i, j) not in complement_indices
    
    array = np.zeros((3, 3))
    indices = point_set(array)
    complement_indices = complement_point_set(array)
    assert indices.isdisjoint(complement_indices)
    for i in range(3):
        for j in range(3):
            assert (i, j) not in indices
            assert (i, j) in complement_indices
    
    array = np.array([[1, 1, 0], [0, 0, 1], [0, 1, 1]])
    indices = point_set(array)
    expected_set = {(0, 0), (0, 1), (1, 2), (2, 1), (2, 2)}
    assert indices == expected_set
    complement_indices = complement_point_set(array)
    expected_complement = {(0, 2), (1, 0), (1, 1), (2, 0)}
    assert expected_complement == complement_indices
    assert indices.isdisjoint(complement_indices)

    array = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    indices = point_set(array)
    expected_set = {(0, 0), (0, 2), (1, 1), (2, 0), (2, 2)}
    assert indices == expected_set
    complement_indices = complement_point_set(array)
    expected_complement = {(0, 1), (1, 0), (1, 2), (2, 1)}
    assert expected_complement == complement_indices
    assert indices.isdisjoint(complement_indices)

    for index in indices:
        assert array[index] == 1
    for index in complement_indices:
        assert array[index] == 0
    
    shifted_indices = translate_point_set(indices, (-1, -1))
    expected_shift = {(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)}
    assert shifted_indices == expected_shift, f"{shifted_indices}"

    mirrored_indices = reflect_point_set(indices)
    expected_mirrored = {(0, 0), (0, -2), (-1, -1), (-2, 0), (-2, -2)}
    assert mirrored_indices == expected_mirrored, f"{mirrored_indices}"

    mirrored_indices = reflect_point_set(shifted_indices)
    expected_mirrored = {(1, 1), (1, -1), (0, 0), (-1, 1), (-1, -1)}
    assert mirrored_indices == expected_mirrored, f"{mirrored_indices}"