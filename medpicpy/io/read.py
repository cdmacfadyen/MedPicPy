import cv2
import SimpleITK as sitk
import os
import ntpath
import numpy as np

def load_image(path):
    image_name = ntpath.basename(path)
    extension = image_name.split(".")[1]
    image = np.zeros(1)
    if extension == "pgm":
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    
    return image