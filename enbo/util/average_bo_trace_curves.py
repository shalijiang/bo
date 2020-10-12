import numpy as np


def average_curves(list_of_arrays):
    """
    given a list of arrays of variant length, normalize the length, and average the curves
    :param list_of_arrays:
    :return: average of the curves
    """
    m = len(list_of_arrays)
    array_lengths = [len(array) for array in list_of_arrays]
    min_array_length = min(array_lengths)
    normalized_arrays = np.zeros((m, min_array_length))
    for i in range(m):
        array = list_of_arrays[i]
        length = len(array)
        thin_idx = np.int32(np.round(np.linspace(0, length - 1, min_array_length)))
        normalized_arrays[i] = array[thin_idx]
    return np.mean(normalized_arrays, axis=0)
