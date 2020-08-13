import numpy as np
import pandas as pd
import cv2
import glob

from . import io
#output shape is the shape for each image
# TODO: we could also have it return the paths, or image 
# names or something
# get_all_slices_from_scans maybe
def load_all_slices_from_series(paths, output_shape):
    """Reads a dataset of 2d images from a 3d series

    Args:
        paths (list or array-like): List of paths to series 
        output_shape (tuple): desired output shape for each slice

    Returns:
        numpy.Array: array containing the reshaped slices
    """
    all_series = [io.load_image(path) for path in paths]
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
def load_specific_slices_from_series(paths, output_shape, slices_to_take):
    """Get specific slice or slices from series of scans.
    Takes path, desired shape and array of slice/slices to 
    take from each series. 

    Args:
        paths (array): array of paths to the series
        output_shape (tuple): desired shape of each slice
        slices_to_take (array of arrays): one array of slices 
            to take for each series

    Returns:
        np.array: every slice as specified by the slices_to_take
    """ 
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

def get_paths_to_images(data_dir, extension):
    paths = glob.glob(data_dir + "/**/*" + extension, recursive=True)

    return paths

def get_paths_from_ids(data_dir, ids, path_filters=[""]):
    """Read in a dataset from a list of patient ids, optionally filtering
    the path. i.e. (i.e. ["CT", "supine"])

    Args:
        data_dir (str): path to dataset
        ids (list or array-like): list of ids to read in, assuming each 
        id is a directory in the dataset (e.g. TCIA datasets)
        path_filters (list, optional): Any filters to apply to the path.
            Defaults to [""].

    Returns:
        array: All paths that match the ids with filters, 
            in the same order as ids
    """
    paths = []
    for id_number in ids:
        paths_for_id = glob.glob(data_dir + "/" + id_number + "/**/", recursive=True)
        for path_filter in path_filters:
            paths_for_id = [path for path in paths_for_id if path_filter in path]
        if paths_for_id:
            paths_for_id = remove_sub_paths(paths_for_id)
        if not paths_for_id:    #TODO: doesn't work on a filter object
            paths.append(None)
            print("Warn: Could not find any paths for id {}".format(id_number))
        else:
            paths.extend(paths_for_id)  #TODO: find the longest one per id

        
    return paths

def remove_sub_paths(paths):
    """Since glob.glob with recursive doesn't
    only take the longest path we need to remove
    paths that are a part of other paths.

    Args:
        paths (array): array of paths

    Returns:
        array: array of paths that aren't a subset of other paths
    """
    for path in paths:
        for other in paths:
            if path in other and path != other:
                paths.remove(path)
    
    return paths