import numpy as np
import pandas as pd
import cv2
import glob

from . import io
#output shape is the shape for each image
# TODO: we could also have it return the paths, or image 
# names or something
# get_all_slices_from_scans maybe
def get_ct_slices_from_paths(ct_paths, output_shape, slices_to_take=[[]]):
    """Reads a dataset of individual ct slices from 
    multiple series of slices.

    Args:
        ct_paths (list or array-like): List of paths to ct series 
        output_shape (tuple): desired output shape for each slice
        slices_to_take (array of arrays): slice (or slices) to take from each scan

    Returns:
        numpy.Array: array containing the reshaped slices
    """
    all_series = [io.load_image(path) for path in ct_paths]
    reshaped = [[cv2.resize(image, output_shape) for image in images] for images in all_series]
    series_lengths = [len(series) for series in reshaped]
    output_array_length = sum(series_lengths)
    output_array_shape = (output_array_length,) + output_shape
    array = np.zeros(output_array_shape)

    output_index = 0
    for series_counter in range(0, len(series_lengths)):
        for image_counter in range(0, series_lengths[series_counter]):
            array[output_index] = reshaped[series_counter][image_counter]
            output_index += 1
    

    return array

#TODO: for this one we do know the output size ahead of time 
# so we can make this faster
def get_slices_from_scans(paths, output_shape, slices_to_take):
    all_series = [io.load_image(path) for path in paths]
    chosen = [[] for series in all_series]

    if len(all_series) is not len(slices_to_take):
        print("length of series is not the same as slices array")
        exit(0)
    
    for series in range(0, len(slices_to_take)):
        for slice_index in slices_to_take[series]:
            chosen_slice = all_series[series][slice_index]
            resized_slice = cv2.resize(chosen_slice, output_shape)
            chosen[series].append(resized_slice)
    
    series_lengths = [len(new_series) for new_series in chosen]
    output_array_length = sum(series_lengths)
    output_array_shape = (output_array_length,) + output_shape

    #TODO: duplicated code.
    array = np.zeros(output_array_shape)

    output_index = 0
    for series_counter in range(0, len(series_lengths)):
        for image_counter in range(0, series_lengths[series_counter]):
            array[output_index] = chosen[series_counter][image_counter]
            output_index += 1
    

    return array

# Currently this makes the most sense for getting 
# array of specific modality from tcia dataset
def filter_image_paths(dataset_root, strings_to_match, extension=""):
    paths = glob.glob(dataset_root + "/**/" + extension, recursive=True)

    print(paths)

#TODO: make it only remove paths that are subsets
def get_paths_from_ids(data_dir, ids, path_filters=[""]):
    """Read in a dataset from a list of patient ids, optionally filtering
    the path. i.e. ["DX"] for X-rays. 

    Args:
        data_dir (str): path to dataset
        ids (list or array-like): list of ids to read in, assuming each 
        id is a directory in the dataset (e.g. TCIA datasets)
        path_filters (list, optional): Any filters to apply to the path 
        (i.e. ["CT", "supine"]). Defaults to [""].
    """
    paths = []
    for id_number in ids:
        paths_for_id = glob.glob(data_dir + "/" + id_number + "/**/", recursive=True)
        
        for path_filter in path_filters:
            paths_for_id = [path for path in paths_for_id if path_filter in path]
        if not paths_for_id:    #TODO: doesn't work on a filter object
            paths.append(None)
            print("Warn: Could not find any paths for id {}".format(id_number))
        else:
            paths.extend(paths_for_id)  #TODO: find the longest one per id

        
    return paths
