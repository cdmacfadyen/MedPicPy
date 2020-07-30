# contains files for doing 2d segmentation reading
import pandas as pd
import numpy as np
import cv2
from .io import read
from sklearn.preprocessing import LabelEncoder, LabelBinarizer

# opt args to add
#   - resize keeps aspect ratio?

def read_images_from_csv(csv, image_name_column, image_dir_path, output_shape):

    array_length = len(csv[image_name_column])
    array_shape = (array_length,) + output_shape    # needs to be a tuple to concatenate
    image_array = np.zeros(array_shape)

    for i in range(0, array_length):
        image_name = csv[image_name_column][i]
        image_path = image_dir_path + image_name
        image = read.load_image(image_path)
        resized = cv2.resize(image, output_shape)
        image_array[i] = resized

    return image_array
def read_masks_from_csv(csv_path, mask_name_column, mask_dir_path): # do we need this when its just the same as the image one?
    
    return 0

# other encoding is categorical with labelencoder, or none and it just returns the series
def read_classes_from_csv(csv, classes_column, encoding='one_hot'):
    classes = None
    encoder = None
    class_column = csv[classes_column]

    #check for nans
    if class_column.isnull().values.any():
        print("Warning: csv contains NaN (not a number values).")
        class_column.fillna("nan", inplace=True)

    if encoding == "one_hot":
        encoder = LabelBinarizer()
    

    classes = encoder.fit_transform(class_column)
    print("{} Classes found: {}".format(len(encoder.classes_),encoder.classes_))
    
    return classes
    
def read_bounding_boxes_from_csv(
    csv, 
    centre_x_column, centre_y_column, 
    width_column, height_column, 
    x_scale_factor=1,
    y_scale_factor=1
    ): # for bounding boxes need to know if measurements are in pixels or mm
    bbox_xs = csv[centre_x_column]
    bbox_xs = bbox_xs.multiply(x_scale_factor)
    xs_array = bbox_xs.to_numpy(dtype=np.float16)

    bbox_ys = csv[centre_y_column]
    bbox_ys = bbox_ys.multiply(y_scale_factor)
    ys_array = bbox_ys.to_numpy(dtype=np.float16)


    bbox_widths = csv[width_column]
    bbox_widths = bbox_widths.multiply(x_scale_factor)
    widths_array = bbox_widths.to_numpy(dtype=np.float16)

    bbox_heights = csv[height_column]
    bbox_heights = bbox_heights.multiply(y_scale_factor)
    heights_array = bbox_heights.to_numpy(dtype=np.float16)

    array_tuple = (xs_array, ys_array, widths_array, heights_array)

    return array_tuple
