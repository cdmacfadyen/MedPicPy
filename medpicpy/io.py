"""medpicpy's lower level functions for reading individual images.

"""
import os
import ntpath
# import pydicom
import cv2
import SimpleITK as sitk
import numpy as np

from . import config
import logging

logging.getLogger(__name__)
mmap_counter = 0

def load_image(path, use_memory_mapping=False, scale_dicom=False):
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
    
    extension = image_name.split(".")[-1]   # i want it to be the last thing
    if extension == "dcm":    # for loading an individual (2d) dicom
        try:
            image = sitk.ReadImage(path)
            image_as_array = sitk.GetArrayFromImage(image)
            image_as_array = image_as_array[0]  # only the first one since this is a 2d dico
            if config.rescale:
                if config.rescale_options["method"] == "per_image":
                    max_value = np.max(image_as_array)
                    min_value = np.min(image_as_array)
                    image_as_array = rescale_image(image_as_array, max_value, min_value)
                elif config.rescale_options["method"] == "from_dtype":
                    if int(image.GetMetaData("0028|0103")) == 0:
                        max_value = 2 ** int(image.GetMetaData("0028|0101")) - 1 # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html
                        min_value = 0
                        image_as_array = rescale_image(image_as_array, max_value, min_value)
                    elif int(image.GetMetaData("0028|0103")) == 1:  # 2's complement
                        bits_stored = int(image.GetMetaData("0028|0101"))
                        min_value = (2 ** (bits_stored - 1))
                        max_value = (2 ** (bits_stored - 1)) - 1
                        image_as_array = rescale_image(image_as_array, max_value, min_value)
                else:
                    print("Pixel Representation not 0 or 1!")
                    exit(0)

        except RuntimeError:
            if config.suppress_errors:
                print(f"Error reading Dicom! {path} must be non-image.")
            else:
                raise

    elif extension == "npy" or extension == "npz":
        image_as_array = np.load(path)
        image_as_array = rescale_opencv_image(image_as_array)
    elif extension == "gz" or extension == "nii":
        image = sitk.ReadImage(path)    
        image_as_array = sitk.GetArrayFromImage(image)
        # print(image.GetMetaData("datatype")) # important one. https://brainder.org/2012/09/23/the-nifti-file-format/
        if config.rescale and config.rescale_options["method"] == "from_dtype":
            datatype = int(image.GetMetaData("datatype"))
            if datatype == 2:
                image_as_array = image_as_array.astype(np.uint8)
            elif datatype == 4:
                image_as_array = image_as_array.astype(np.int16)
            elif datatype == 16:
                image_as_array = image_as_array.astype(np.int32)
            elif datatype == 512:
                image_as_array = image_as_array.astype(np.uint16)
            else:
                print(f"MedPicPy currently doesn't support datatype {datatype} of nii.gz")
                exit(0)
        if config.rescale and config.rescale_options["method"] == "per_image":
            image_as_array = rescale_opencv_image(image_as_array)
    else:
        image_as_array = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image_as_array is None:  # opencv couldn't read it, maybe sitk can
            try:
                image = sitk.ReadImage(path)    
                image_as_array = sitk.GetArrayFromImage(image)  # SITK also can't read it
            except RuntimeError:
                logging.debug(f"Suppressing sitk not being able to read file: {path}")  
                if config.suppress_errors:
                    return None
                else:
                    raise
        image_as_array = rescale_opencv_image(image_as_array)

    if use_memory_mapping and image_as_array is not None:
        mmap_name = get_counter_and_update()
        mmap = np.memmap(mmap_name, dtype=np.float32, mode="w+", shape=image_as_array.shape)
        mmap[:] = image_as_array[:]
        return mmap
    else:
        return image_as_array

def np_dtype_max_and_min(dtype):
    """Takes a numpy dytpe and 
    returns the maximum and minimum possible values 
    for that dtype. 

    Args:
        dtype (numpy dtype): dtype to find max and min values

    Returns:
        (number, number): max and min value for given dtype, can be float or int
    """
    if np.issubdtype(dtype, np.integer):
        #int things
        int_info = np.iinfo(dtype)
        max_val = int_info.max
        min_val = int_info.min
        return max_val, min_val
    elif np.issubdtype(dtype, np.floating):
        #float things
        float_info = np.finfo(dtype)
        max_val = float_info.max
        min_val = float_info.min
        return max_val, min_val
    else:
        #TODO: raise exception
        pass
    
def rescale_opencv_image(image_as_numpy):
    """Rescale an image that has been loaded 
    using opencv. This loads the images directly into 
    numpy format. 

    Args:
        image_as_numpy (np array): numpy array containing image data

    Returns:
        np array: rescaled image
    """
    if config.rescale_options["method"] == "per_image":
        max_value = np.max(image_as_numpy)
        min_value = np.min(image_as_numpy)

        if max_value == min_value:
            return np.zeros_like(image_as_numpy)    # avoid dividing by zero
        else:
            return rescale_image(image_as_numpy, max_value, min_value)
    elif config.rescale_options["method"] == "from_dtype":
        shape = image_as_numpy.shape
        if len(shape) == 3:
            # We know its 3D but not 3 Channel because 3 Channel isn't allowed. 
            return rescale_opencv_image_3d(image_as_numpy)

        depth = image_as_numpy.dtype
        max_value, min_value = np_dtype_max_and_min(depth)
        return rescale_image(image_as_numpy, max_value, min_value)

def rescale_opencv_image_3d(image_as_numpy):
    """Rescale an image with more than 
    2-dimensions.

    Args:
        image_as_numpy (np array): numpy array containing image data

    Returns:
        np array: rescaled 3d image
    """
    rescaled_arr = np.zeros(image_as_numpy.shape, dtype=np.float32)
    for i in range(len(image_as_numpy)):        
        rescaled_arr[i] = rescale_opencv_image(image_as_numpy[i])
    return rescaled_arr

def rescale_image(image,
    upper_bound,
    lower_bound):
    """General feature scaling function. Takes an 
    image to rescale and the upper and lower possible 
    value for that image data. 

    Args:
        image (np array): array containing image data
        upper_bound (int): max possible pixel value in image
        lower_bound (int): min possible pixel value in image

    Returns:
        np.array: rescaled image
    """
    top = (image - lower_bound) * (config.rescale_options["max"] - config.rescale_options["min"])
    final = (top / (upper_bound - lower_bound)) + config.rescale_options["min"]
    return final + config.rescale_options["min"]


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
    path = config.cache_location + "/medpicpy_cache/" + str(val) + ".dat"
    mmap_counter += 1
    return path