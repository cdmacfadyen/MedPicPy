"""medpicpy's lower level functions for reading individual images.

"""
import os
import ntpath

import cv2
import SimpleITK as sitk
import numpy as np

mmap_counter = 0

def load_image(path, use_memory_mapping=False):
    """Load in any image or image series from a path

    Args:
        path (str): path to image or directory for a series

    Returns:
        np.Array: image in numpy format
    """
    image_name = ntpath.basename(path)
    image_as_array = None

    if os.path.isdir(path): # if its a directory its a dicom series or something similar
        series = load_series(path, use_memory_mapping=use_memory_mapping)
        return series
    
    extension = image_name.split(".")[1]
    if extension == "dcm":    # for loading an individual (2d) dicom
        image = sitk.ReadImage(path)
        image_as_array = sitk.GetArrayFromImage(image)
        image_as_array = image_as_array[0]  # only the first one since this is a 2d dico
    else:
        image_as_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        
        if image_as_array is None:  # opencv couldn't read it
            image = sitk.ReadImage(path)
            image_as_array = sitk.GetArrayFromImage(image)
    
    if use_memory_mapping:
        mmap_name = get_counter_and_update()
        mmap = np.memmap(mmap_name, dtype=np.float32, mode="w+", shape=image_as_array.shape)
        mmap[:] = image_as_array[:]
        return mmap
    else:
        return image_as_array


def load_series(path, use_memory_mapping=False): # for more than 2d dicoms. 
    """Load an image series from a directory(e.g. dicom)

    Args:
        path (str): Path to directory of series

    Returns:
        np.Array: Image in numpy array format
    """
    series_reader = sitk.ImageSeriesReader()
    file_names = series_reader.GetGDCMSeriesFileNames(path)
    series_reader.SetFileNames(file_names)
    image = series_reader.Execute()
    array = sitk.GetArrayFromImage(image)

    if use_memory_mapping:
        mmap_name = get_counter_and_update()
        mmap = np.memmap(mmap_name, dtype=np.float32, mode="w+", shape=array.shape)
        mmap[:] = array[:]
        return mmap
    else:
        return array

def allocate_array(shape, use_memory_mapping=False):
    """Allocates space for a numpy array, and abstracts over memory mapping.

    Args:
        shape (tuple): shape of numpy array
        use_memory_mapping (bool, optional): Store array on disk instead of in memory. Defaults to False.

    Returns:
        np.array: numpy array with given shape. 
    """
    if use_memory_mapping:
        mmap_name = get_counter_and_update()
        mmap = np.memmap(mmap_name, dtype=np.float32, mode="w+", shape=shape)
        return mmap
    else:
        return np.zeros(shape)

def get_counter_and_update():
    global mmap_counter
    val = mmap_counter
    path = "medpicpy_cache/" + str(val) + ".dat"
    mmap_counter += 1
    return path