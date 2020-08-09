# contains files for doing 2d segmentation reading
import pandas as pd
import numpy as np
import cv2
import glob
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, LabelBinarizer


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

# other encoding is categorical with labelencoder, or none and it just returns the series
# TODO: probably leave the encoding out and that way they can do whatever they want. 
# means that this doesn't have to require sklearn. 
# TODO: kind of useless since they already have the classes as an array,
# probably just remove
def load_classes_from_csv(dataframe, classes_column, encoding='one_hot'):
    """Read classes from column in dataframe and optionally 
    transform to one hot or categorical values. 


    Args:
        dataframe (pandas.DataFrame): DataFrame of csv
        classes_column (index): Index of column with classes
        encoding (str, optional): Encoding to be applied to classes. 
            'one_hot', 'categorical' or None. Defaults to 'one_hot'

    Returns:
        np.Array: array of encoded class names
    """
    classes = None
    encoder = None
    class_column = dataframe[classes_column]

    #check for nans
    if class_column.isnull().values.any():
        print("Warning: csv contains NaN (not a number values).")
        class_column.fillna("nan", inplace=True)

    if encoding == "one_hot":
        encoder = LabelBinarizer()
    classes = encoder.fit_transform(class_column)
    print("{} Classes found: {}".format(len(encoder.classes_),encoder.classes_))
    
    return classes

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
    """General image loading function that takes an array of 
    paths and an output shape and returns the images in 
    the same order as the paths. Requires every 
    path to have an image and every image to be resizeable 
    to the given output shape

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