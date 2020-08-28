"""medpicpy's lower level functions for reading individual images.

"""
import os
import ntpath

import cv2
import SimpleITK as sitk
import numpy as np


def load_image(path):
    """Load in any image or image series from a path

    Args:
        path (str): path to image or directory for a series

    Returns:
        np.Array: image in numpy format
    """
    print("reading " + path)
    image_name = ntpath.basename(path)
    image_as_array = None

    if os.path.isdir(path): # if its a directory its a dicom series or something similar
        series = load_series(path)
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
    
    mmap = np.memmap(path + ".dat", dtype=np.float32, mode="w+", shape=image_as_array.shape)
    mmap[:] = image_as_array[:]
    return mmap


def load_series(path): # for more than 2d dicoms. 
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

    mmap = np.memmap(path + ".dat", dtype=np.float32, mode="w+", shape=array.shape)
    mmap[:] = array[:]
    return mmap