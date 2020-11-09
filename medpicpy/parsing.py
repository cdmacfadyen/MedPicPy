"""medpicpy's higher level functions to abstract over reading 
in medical imaging data

"""
import glob
from pathlib import Path
from os.path import normpath

import pandas as pd
import numpy as np
import cv2
import logging

from . import io
from .utils import remove_sub_paths
from . import config

logging.getLogger(__name__)

def load_images_from_csv(dataframe, 
image_name_column, 
image_dir_path, 
output_shape,
use_memory_mapping=False):
    """Read in an array of images from paths specified in a csv

    ##Example
    ```python
    import medpicpy as med
    import pandas as pd

    description = pd.read_csv("data.csv") 
    array = med.load_images_from_csv(description, 0, "mini-MIAS/", (224, 224))
    ```
    Args:
        dataframe (pandas.DataFrame): A pandas dataframe from the csv
        image_name_column (index): Index of column with image names
        image_dir_path (string): Path to directory containing images
        output_shape (tuple): Output shape for each image
        use_memory_mapping (optional, boolean): store the data on disk instead of in memory.
            Defaults to False

    Returns:
        np.Array: Array of images in order 
    """
    image_names = dataframe[image_name_column]
    image_paths = image_names.apply(lambda x : image_dir_path + "/" + x)
    image_paths = image_paths.apply(lambda x : normpath(x))

    images = load_images_from_paths(image_paths, output_shape, use_memory_mapping=use_memory_mapping)
    return images

    
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

    ##Example
    ```python
    import medpicpy as med
    import pandas as pd

    description = pd.read_csv("data.csv") 

    # x and y scale factor are new_image_size / original_image_size
    # only set if the images were resized when being loaded in
    x_scale_factor = 224 / 1024
    y_scale_factor = 224 / 1024

    xs, ys, widths, heights = med.load_bounding_boxes_from_csv(
        description, 
        4, 
        5, 
        6, 
        6, 
        x_scale_factor=x_scale_factor, 
        y_scale_factor=y_scale_factor
    )

    ```
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
# TODO: make this work for 3D images, either make a new function or 
# add optional args (would be slice axis and slices to take)
def load_classes_in_directory_name(directory, 
    image_extension, 
    output_shape, 
    class_level=1,
    slices_to_take=None,
    slice_axis=-2,
    use_memory_mapping=False):
    """Parse datasets where the class name is in the 
    directory structure

    Use this when the class name is one of the directory names
    in the dataset structure. 
    ## Example
    If dataset has directory structure:
    ```
    dataset/
        benign/
            im001.dcm
            im002.dcm
        malignant/
            im001.dcm
            im002.dcm
    ```
    then:
    ```python
        import medpicpy as med
        
        classes, images = med.load_classes_in_directory_name(
            "dataset/",
            ".dcm",
            "(128, 128)"
        )
        print(classes)
        # ["benign", "benign", "malignant", "malignant"]
        print(images.shape)
        # (4, 128, 128)
    ```
    Args:
        directory (path): root directory of dataset
        image_extension (str): Wildcard for identifying images,
             e.g for png's - *.png
        output_shape (tuple): Desired output shape of images
        class_level (int, optional): Which level of directory structure 
            contains class name. Defaults to 1.
        use_memory_mapping (optional, boolean): store the data on disk instead of in memory.
            Defaults to False
    Returns:
        list(str), np.Array : list of classes and corresponding images with correct shape
    """
    path_to_search = directory + "/**/*" + image_extension
    files = glob.glob(path_to_search, recursive=True)
    files = remove_sub_paths(files)
    number_of_files = len(files)
    array_shape = (number_of_files,) + output_shape
    array = io.allocate_array(array_shape, use_memory_mapping=use_memory_mapping)
    classes = np.empty(number_of_files, dtype=object)

    for index, name in enumerate(files):
        parts = Path(name).parts
        class_name = parts[class_level]

        image = io.load_image(name, use_memory_mapping=use_memory_mapping)
        result = cv2.resize(image, output_shape)

        classes[index] = class_name
        array[index] = result
        
    return classes, array



def load_images_from_paths(paths, output_shape, use_memory_mapping=False):
    """2D image loading function that takes an array of 
    paths and an output shape and returns the images in 
    the same order as the paths. Requires every 
    path to have an image and every image to be resizeable 
    to the given output shape.

    For higher dimension images use load_series_from_paths.

    Args:
        paths (list or array-like): paths of images to load
        output_shape (tuple): desired shape of each image
        use_memory_mapping (optional, boolean): store the data on disk instead of in memory.
            Defaults to False
    Returns:
        np.array: all images in numpy format with given shape
    """
    array_length = len(paths)
    array_shape = (array_length,) + output_shape # concat tuples to get shape
    image_array = io.allocate_array(array_shape, use_memory_mapping=use_memory_mapping)

    for i in range(0, array_length):
        print("Loading images {} / {}".format(i + 1, len(paths)), end="\r", flush=True)
        image_name = paths[i]
        image = io.load_image(image_name, use_memory_mapping=use_memory_mapping)
        resized = cv2.resize(image, output_shape)
        image_array[i] = resized
    print("")
    return image_array

# slice axis will be -2 for most things since they 
# are 1 channel, for colour images would probably be -3
# But I don't think you get colour 3D scans
# It would work for multimodal things stacked on top of each other though
def load_series_from_paths(
    paths,
    slice_output_shape, 
    slices_to_take,
    slice_axis=-2,
    use_memory_mapping=False
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

    ## Example
    If there is dataset with structure:
    ```
    data/
        patient-data.csv
        ID-001/
            SCANS/
                CT/
                    prone.nii.gz
        ID-002/
            SCANS/
                CT/
                    prone.nii.gz
        ID-003/
            SCANS/
                CT/
                    prone.nii.gz
    ```
    then:
    ```python
    import pandas as pd
    import medpicpy as med

    description = pd.read_csv("data/patient-data.csv")
    patient_ids = description("id")
    filters = ["CT", "prone"]

    image_paths = med.get_paths_from_ids(
        "data/",
        patient_ids,
        filters
    )

    print(image_paths)
    # ["data/ID-001/CT/prone.nii.gz", "data/ID-002/CT/prone.nii.gz", "data/ID-003/CT/prone.nii.gz"]
    
    slices_to_take = range(60, 120)
    output_slice_shape = (128, 128)    # desired shape of each slice in the scan
    images = med.load_series_from_paths(
        paths, 
        output_slice_shape,
        slices_to_take
        )

    print(images.shape)
    # (3, 60, 128, 128)  
    ```
    
    Args:
        paths (list): list of paths to the scans to load
        slice_output_shape (tuple): shape each slice should be resized to
        slices_to_take (list): list of indices of slices to take
        slice_axis (int, optional): axis to resize along. Defaults to -2.
        use_memory_mapping (optional, boolean): store the data on disk instead of in memory.
            Defaults to False
    Returns:
        np.array: array of all scans with specified size
    """

    output_shape = (len(paths), len(slices_to_take)) + slice_output_shape
    output_array = io.allocate_array(output_shape, use_memory_mapping=use_memory_mapping)
    
    for i in range(0, len(paths)):
        print("Loading images {} / {}".format(i + 1, len(paths)), end="\r", flush=True)
        path = paths[i]
        image = io.load_image(path, use_memory_mapping=False)
        new_image = io.allocate_array(((len(slices_to_take),) + image[0].shape), use_memory_mapping=False)

        for index, slice_index in enumerate(slices_to_take):
            new_image[index] = image[slice_index]
        
        final_shape = new_image.shape[:slice_axis] + slice_output_shape + new_image.shape[:slice_axis + 2]
        final_image = io.allocate_array(final_shape, use_memory_mapping=use_memory_mapping)

        for j in range(final_shape[0]):
            image = new_image[j][slice_axis]
            image = cv2.resize(image, slice_output_shape)
            final_image[j] = image
        
        output_array[i] = final_image
    print("")
    return output_array

# get_all_slices_from_scans maybe
# TODO: have it return an array containing the 
# paths also
def get_length_of_all_series(paths, skip_rgb = True):
    all_series = []
    for index, path in enumerate(paths):
        if index % 100 == 0:
            print(f"Getting length of images ~ {index} of {len(paths)}", end="\r")
            logging.debug(f"Getting length of images ~ {index} of {len(paths)}")

        image = io.load_image(path)
        shape = None if image is None else image.shape
        # if len(shape) == 3:
        #     print("3d image")
        all_series.append(shape)

    print("finished reading all series")
    none_count = 0
    for series in all_series:
        if series is None:
            none_count += 1
    print(f"{none_count} out of {len(all_series)} could not be read.")
    final_paths = []
    series_and_paths = [(series, paths[i]) for i, series in enumerate(all_series) if series is not None]
    all_series = [series for series in all_series if series is not None]
    number_of_series = 0
    print("shape: ", len(all_series), all_series[0])
    for index, series in enumerate(all_series):
        path = series_and_paths[index][1]   # get the path for the image
        if len(series) == 2:  # then it is already 2D
            number_of_series += 1
            final_paths.append(path)
            continue
        elif len(series) == 3: # if its 3d
            print(f"It's 3D! {series}")
            print(path)
            # import matplotlib.pyplot as plt
            # fig, ax = plt.subplots()
            # ax.imshow(series[0], cmap="gray")
            # plt.savefig(f"./images/error.png")
            # exit(0)
            if series[0] == 1:        # if its actually 2d
                number_of_series += 1
                final_paths.append(path)
            elif series[2] == 3:
                if not config.suppress_errors:
                    print("MedPicPy does not currently work with multichannel images")
                    exit(0)                    
                continue
            else:
                for image in series:
                    number_of_series += 1
                    final_paths.append(path)
        else:
            print(f"Its not 2 or 3D!:{series}")
            exit(0)
    return number_of_series, final_paths
            
def load_all_slices_from_series(paths,
    all_series_length,
    output_shape,
    use_memory_mapping=False):
    """Reads a dataset of 2d images from a 3d series

    Args:
        paths (list or array-like): List of paths to series 
        output_shape (tuple): desired output shape for each slice
        use_memory_mapping (optional, boolean): store the data on disk instead of in memory.
            Defaults to False
    Returns:
        numpy.Array: array containing the reshaped slices
    """
    
    print(f"output_array_length: {all_series_length}")
    output_array_shape = (all_series_length,) + output_shape
    print(f"output_array_shape: {output_array_shape}")
    print("Len paths ", len(paths))
    array = io.allocate_array(output_array_shape, use_memory_mapping=use_memory_mapping)
    # output_array_length: 2970405
    images_written = 0
    for image_index, path in enumerate(paths):
        if images_written == all_series_length:
            print("Breaking early")
            break
        if image_index % 100 == 0:
            print(f"Re-loading image: {image_index} of {len(paths)}", end="\r")
            logging.debug(f"Getting length of images ~ {image_index} of {len(paths)}")
        
        image = io.load_image(path, scale_dicom=True,use_memory_mapping=use_memory_mapping)
        if image is None:
            continue
        image_shape = image.shape
        if len(image_shape) == 2:
            resized_image = cv2.resize(image, output_shape)
            array[images_written] = resized_image[:]
            images_written += 1
        elif len(image_shape) == 3:

            if image_shape[0] == 1: #its actually 2d
                resized_image = cv2.resize(image, output_shape)
                array[images_written] = resized_image[:]
                images_written += 1
            elif image_shape[2] == 3:
                continue    #skip rgb
            else:
                for slice_index in range(len(image)):
                    resized_image = cv2.resize(image[slice_index], output_shape)
                    array[images_written] = resized_image[:]
                    images_written += 1
                    
    return array
    #MR/CPTAC-GBM
    #/data2/cdcm/train/MR/Anti-PD-1_MELANOMA/7-138.dcm
    #/data2/cdcm/train/CT/ACRIN-FLT-Breast/27-215.dcm
    # # array = da.zeros(output_array_shape, chunks="auto")
    # output_index = 0
    # for series_counter in range(0, len(series_lengths)):
    #     for image_counter in range(0, series_lengths[series_counter]):
    #         array[output_index] = reshaped[series_counter][image_counter]
    #         output_index += 1
    
    # # return array.compute()

def load_specific_slices_from_series(
    paths,
    output_shape,
    slices_to_take,
    use_memory_mapping=False):
    """Get specific slice or slices from series of scans.
    Takes path, desired shape and array of slice/slices to 
    take from each series. 

    Args:
        paths (array): array of paths to the series
        output_shape (tuple): desired shape of each slice
        slices_to_take (array of arrays): one array of slices 
            to take for each series
        use_memory_mapping (optional, boolean): store the data on disk instead of in memory.
            Defaults to False

    Returns:
        np.array: every slice as specified by the slices_to_take
    """ 
    all_series = [io.load_image(path, use_memory_mapping=use_memory_mapping) for path in paths]
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

    array = io.allocate_array(output_array_shape, use_memory_mapping=use_memory_mapping)

    output_index = 0
    for series_counter in range(0, len(series_lengths)):
        for image_counter in range(0, series_lengths[series_counter]):
            array[output_index] = chosen[series_counter][image_counter]
            output_index += 1
    

    return array

def stack_modalities(arrays, axis=-1):
    """Turn a list of arrays into one multimodal array.
    
    Creates one array where each element has 
    len(arrays) images. 

    ##Example
    If we have a dataset like:
    ```
    dataset/
        ID-1/
            flair.nii.gz
            t1.nii.gz
        ID-2/
            flair.nii.gz
            t1.nii.gz
    ```
    then:
    ```python
    import medpicpy as med 
    modalities = [["flair"], ["t1"]]
    paths_for_modality = [med.get_paths_from_ids(
        "dataset/",
        ["ID-1", "ID-2"],
        path_filters = modality
    ) for modality in modalities]

    arrays = [med.load_series_from_paths(
        paths,
        (128, 128),
        range(60, 80)
    ) for paths in paths_for_modality]

    multimodal_array = med.stack_modalities(arrays)
    print(multimodal_array.shape)
    # (259, 20, 128, 128, 4)
    ```
    You might want to flatten along the first axis after 
    doing this depending on the dimensionality of the model you are using.
    ```python
    flat_multi_modal = multimodal_array.reshape(-1, *multimodal_array.shape[2:])
    print("multi modal shape: ", flat_multi_modal.shape)
    # (5180, 128, 128, 4)
    ```
    Args:
        arrays (array): array of arrays of images to stack on 
            top of each other
        axis (int, optional): The axis to stack along, 
            leaving this default is probably fine. Defaults to -1.

    Returns:
        array: the arrays stacked on top of each other
    """
    return np.stack(arrays, axis=axis)