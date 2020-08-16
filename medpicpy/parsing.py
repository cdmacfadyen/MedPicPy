"""medpicpy's higher level functions to abstract over reading 
in medical imaging data

"""
import glob
from pathlib import Path

import pandas as pd
import numpy as np
import cv2

from . import io

# opt args to add
#   - resize keeps aspect ratio?
#TODO: probably change it from taking a dataframe to taking an array (i.e. pd.Series)
def load_images_from_csv(dataframe, image_name_column, image_dir_path, output_shape):
    """Read in an array of images from paths specified in a csv

    Args:
        dataframe (pandas.DataFrame): A pandas dataframe from the csv
        image_name_column (index): Index of column with image names
        image_dir_path (string): Path to directory containing images
        output_shape (tuple): Output shape for each image

    Returns:
        np.Array: Array of images in order 
    """
    array_length = len(dataframe[image_name_column])
    array_shape = (array_length,) + output_shape    # needs to be a tuple to concatenate
    image_array = np.zeros(array_shape)

    for i in range(0, array_length):
        image_name = dataframe[image_name_column][i]
        image_path = image_dir_path + image_name
        image = io.load_image(image_path)
        resized = cv2.resize(image, output_shape)
        image_array[i] = resized

    return image_array
    
    
#TODO kind of useless since they already have the bounding boxes as arrays
def load_bounding_boxes_from_csv(
    dataframe, 
    centre_x_column, 
    centre_y_column, 
    width_column, 
    height_column, 
    x_scale_factor=1,
    y_scale_factor=1
    ): # for bounding boxes need to know if measurements are in pixels or mm
    """Read bounding boxes from dataframe of csv

    Args:
        dataframe (pandas.DataFrame): Dataframe of csv
        centre_x_column (index): Index of column for x anchor or box
        centre_y_column (index): Index of column for y anchor of box
        width_column (index): Index of column for width of box
        height_column (index): Index of column for heigh of box.
            Can be same as width column for squares or circles.
        x_scale_factor (int, optional): Factor to rescale by if image was reshaped. Defaults to 1.
        y_scale_factor (int, optional): Factor to rescale by if image was reshaped. Defaults to 1.

    Returns:
        tuple: 4 tuple of np.Arrays with x, y, widths and heights
    """
    bbox_xs = dataframe[centre_x_column]
    bbox_xs = bbox_xs.multiply(x_scale_factor)
    xs_array = bbox_xs.to_numpy(dtype=np.float16)

    bbox_ys = dataframe[centre_y_column]
    bbox_ys = bbox_ys.multiply(y_scale_factor)
    ys_array = bbox_ys.to_numpy(dtype=np.float16)


    bbox_widths = dataframe[width_column]
    bbox_widths = bbox_widths.multiply(x_scale_factor)
    widths_array = bbox_widths.to_numpy(dtype=np.float16)

    bbox_heights = dataframe[height_column]
    bbox_heights = bbox_heights.multiply(y_scale_factor)
    heights_array = bbox_heights.to_numpy(dtype=np.float16)

    array_tuple = (xs_array, ys_array, widths_array, heights_array)

    return array_tuple

# To read datasets where the class name is in the directory structure.
# i.e. covid/im001 or no-covid/im001
# pulls the class names from the path and reads in the images
# as a numpy array
def load_classes_in_directory_name(directory, image_file_wildcard, output_shape, class_level=1):
    """Parse datasets where the class name is in the 
    directory structure

    Args:
        directory (path): root directory of dataset
        image_file_wildcard (str): Wildcard for identifying images,
             e.g for png's - *.png
        output_shape (tuple): Desired output shape of images
        class_level (int, optional): Which level of directory structure 
            contains class name. Defaults to 1.

    Returns:
        list(str), np.Array : list of classes and corresponding images with correct shape
    """
    path_to_search = directory + "/**/" + image_file_wildcard
    files = glob.glob(path_to_search, recursive=True)

    number_of_files = len(files)
    array_shape = (number_of_files,) + output_shape #concatonate the tuples
    array = np.zeros(array_shape, dtype=np.int16)
    classes = np.empty(number_of_files, dtype=object)

    for index, name in enumerate(files):
        parts = Path(name).parts
        class_name = parts[class_level]

        image = io.load_image(name)
        result = cv2.resize(image, output_shape)

        classes[index] = class_name
        array[index] = result
        
    return classes, array



def load_images_from_paths(paths, output_shape):
    """2D image loading function that takes an array of 
    paths and an output shape and returns the images in 
    the same order as the paths. Requires every 
    path to have an image and every image to be resizeable 
    to the given output shape.

    For higher dimension images use load_scans_from_paths.

    Args:
        paths (list or array-like): paths of images to load
        output_shape (tuple): desired shape of each image

    Returns:
        np.array: all images in numpy format with given shape
    """
    array_length = len(paths)
    array_shape = (array_length,) + output_shape # concat tuples to get shape
    image_array = np.zeros(array_shape)

    for i in range(0, array_length):
        image_name = paths[i]
        image = io.load_image(image_name)
        resized = cv2.resize(image, output_shape)
        image_array[i] = resized
    
    return image_array

# slice axis will be -2 for most things since they 
# are 1 channel, for colour images would probably be -3
# But I don't think you get colour 3D scans
# It would work for multimodal things stacked on top of each other though
def load_scans_from_paths(
    paths,
    slice_output_shape, 
    slices_to_take,
    slice_axis=-2
    ):
    """Load an array of 3D scans into memory from their paths.

    Useful for e.g. CT or MR scans. Takes a list of paths, the output shape
    for each 2D slice and a list containing which slices 
    to take from each image. To take the first 60 slices
    pass range(0, 60). 

    The output shape should be a tuple of (int, int).
    
    Optionally take which axis to reshape the image along.
    For any scans with one channel (grayscale) slices this should 
    be -2, if there is a colour channel (or its some kind 
    of multimodal stack) then the axis would be -3. 

    Args:
        paths (list): list of paths to the scans to load
        slice_output_shape (tuple): shape each slice should be resized to
        slices_to_take (list): list of indices of slices to take
        slice_axis (int, optional): axis to resize along. Defaults to -2.

    Returns:
        np.array: array of all scans with specified size
    """

    temp_list = []
    for i in range(0, len(paths)):
        path = paths[i]
        image = io.load_image(path)
        new_image = np.zeros(((len(slices_to_take),) + image[0].shape))
        for index, slice_index in enumerate(slices_to_take):
            new_image[index] = image[slice_index]
        
        final_shape = new_image.shape[:slice_axis] + slice_output_shape + new_image.shape[:slice_axis + 2]
        final_image = np.zeros(final_shape)

        for i in range(final_shape[0]):
            image = new_image[i][slice_axis]
            image = cv2.resize(image, slice_output_shape)
            final_image[i] = image
        
        temp_list.append(final_image)
    return np.array(temp_list)

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

def stack_modalities(arrays, axis=-1):
    return np.stack(arrays, axis=axis)