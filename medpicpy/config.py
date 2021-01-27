"""
This module allows setting of some global values that 
medpicpy references internally. 

cache_location refers to where the image data is stored 
when `use_memory_mapping` is set in a function call.

suppress_errors means that exceptions will not be thrown if 
an invalid path is read as an image. This may be desired for some 
large projects where there are non-image files hidden in the 
dataset. 

rescale means that the images are automatically scaled 
between 0 and 1 as they are loaded. Useful for machine learning 
projects. 
"""
import os

cache_location = "."
suppress_errors = False

rescale = False

rescale_options = {
    "method" : "per_image",
    "min" : 0,
    "max" : 1
}

def set_cache_location(path):
    global cache_location
    cache_location = path
    if not os.path.exists(cache_location + "/medpicpy_cache/"):
        os.mkdir(cache_location + "/medpicpy_cache/")