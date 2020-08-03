import numpy as np
import pandas as pd
import cv2

from .io import read

#output shape is the shape for each image
# TODO: we could also have it return the paths, or image 
# names or something
def get_ct_slices_from_paths(ct_paths, output_shape):
    """Reads a dataset of individual ct slices from 
    multiple series of slices.

    Args:
        ct_paths (list or array-like): List of paths to ct series 
        output_shape (tuple): desired output shape for each slice

    Returns:
        numpy.Array: array containing the reshaped slices
    """
    all_series = [read.load_image(path) for path in ct_paths]
    reshaped = [[cv2.resize(image, output_shape) for image in images] for images in all_series]
    series_lengths = [len(series) for series in reshaped]
    output_array_length = sum(series_lengths)
    print("output array len ", output_array_length)
    output_array_shape = (output_array_length,) + output_shape
    array = np.zeros(output_array_shape)

    output_index = 0
    for series_counter in range(0, len(series_lengths)):
        for image_counter in range(0, series_lengths[series_counter]):
            array[output_index] = reshaped[series_counter][image_counter]
            output_index += 1
    

    return array
