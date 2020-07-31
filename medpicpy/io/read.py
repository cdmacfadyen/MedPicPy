import cv2
import SimpleITK as sitk
import os
import ntpath
import numpy as np


# one stop shop for image reading
def load_image(path):
    image_name = ntpath.basename(path)
    image_as_array = None
    if os.path.isdir(path): # if its a directory its a dicom series or something similar
        series = load_series(path)
        return series
    
    extension = image_name.split(".")[1]
    if extension == "pgm":
        image_as_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    elif extension == "dcm":    # for loading an individual (2d) dicom
        image = sitk.ReadImage(path)
        image_as_array = sitk.GetArrayFromImage(image)
        image_as_array = image_as_array[0]  # only the first one since this is a 2d dicom
    return image_as_array


def load_series(path): # for more than 2d dicoms. 
    series_reader = sitk.ImageSeriesReader()
    file_names = series_reader.GetGDCMSeriesFileNames(path)
    series_reader.SetFileNames(file_names)
    image = series_reader.Execute()

    array = sitk.GetArrayFromImage(image)
    return array