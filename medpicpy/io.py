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
            # print(f"Image Type: {image.GetPixelIDTypeAsString()}")
            # print("Bits Allocated", image.GetMetaData("0028|0100"))
            # print("Bits Stored", image.GetMetaData("0028|0101"))
            # print("High Bit", image.GetMetaData("0028|0102"))
            # print("Pixel Representation", image.GetMetaData("0028|0103"))
            # print("Photometric Interpretation: ", image.GetMetaData("0028|0004"))
            # So max value for unsigned int is 2^Bits Stored - 1
            # Max Value for signed int 
            # print(f"Min value: {np.min(image_as_array)}")
            # print(f"Max value: {np.max(image_as_array)}")
            if scale_dicom:
                if int(image.GetMetaData("0028|0103")) == 0:
                    max_value = 2 ** int(image.GetMetaData("0028|0101")) - 1 # http://dicom.nema.org/medical/dicom/current/output/chtml/part03/sect_C.7.6.3.html
                    min_value = 0
                    image_as_array[image_as_array < min_value] = min_value
                    image_as_array[image_as_array > max_value] = max_value
                    image_as_array = (image_as_array) / max_value
                    if np.min(image_as_array) < 0 or np.max(image_as_array) > 1:
                        import matplotlib.pyplot as plt
                        print("Quitting because final number outwith range")
                        fig, ax = plt.subplots()
                        ax.imshow(image_as_array, cmap = "gray")
                        plt.savefig("./images/error.png")
                        exit(0)
                elif int(image.GetMetaData("0028|0103")) == 1:  # 2's complement
                    bits_stored = int(image.GetMetaData("0028|0101"))
                    min_value = (2 ** (bits_stored - 1))
                    max_value = (2 ** (bits_stored - 1)) - 1
                    image_as_array[image_as_array < -min_value] = -min_value
                    image_as_array[image_as_array > max_value] = max_value
                    # print(f"Min: -{min_value}, Max: {max_value}")
                    # print(f"Max - Min: {max_value + min_value}")
                    # print(np.max(image_as_array) + min_value)
                    # print(np.min(image_as_array) + min_value)
                    top = image_as_array + min_value
                    # print(f"TOP: {np.min(top)} --- {np.max(top)}")
                    final = top / (max_value + min_value)
                    # print(f"FINAL: {np.min(final)} --- {np.max(final)}")
                    if np.min(final) < 0 or np.max(final) > 1:
                        print("Quitting because final number outwith range")
                        import matplotlib.pyplot as plt
                        fig, ax = plt.subplots()
                        ax.imshow(final, cmap = "gray")
                        plt.savefig("./images/error.png")
                        exit(0)
                    image_as_array = final



                else:
                    print("Pixel Representation not 0 or 1!")
                    exit(0)

        except RuntimeError:
            print(f"Error reading Dicom! {path} must be non-image.")
            # exit(0)
            # print(os.path.isfile(path))
            # try:
            #     dicom = pydicom.dcmread(path)
            #     if dicom.pixel_array:
            #         print("Image present that sitk didn't read!")
            #         print(f"Image Shape: {dicom.pixel_array}" )
            #         exit(0)
            # except AttributeError:
            #     pass 

    elif extension == "npy" or extension == "npz":
        image_as_array = np.load(path)
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
    if np.issubdtype(dtype, np.integer):
        #int things
        int_info = np.iinfo(dtype)
        max_val = int_info.max
        min_val = int_info.min
        return max_val, min_val
    elif np.issubdtype(dtype, np.floating):
        #float things
        print("Wasn't expecting dtype to be float!")
        exit(0)
    
def rescale_opencv_image(image_as_numpy):
    depth = image_as_numpy.dtype
    # print(f"Depth {depth}")
    max_value, min_value = np_dtype_max_and_min(depth)
    # print(f"Max value {max_value}, min value {min_value}")
    
    numerator = image_as_numpy - min_value
    denominator = max_value - min_value
    rescaled = numerator / denominator

    # TODO: remove
    if np.min(rescaled) < 0 or np.max(rescaled) > 1:
        print("Quitting because final number outwith range")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.imshow(final, cmap = "gray")
        plt.savefig("./images/error.png")
        exit(0)
    
    return rescaled

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