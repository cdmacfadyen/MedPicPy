"""
MedPicPy provides a simple API for loading medical 
imaging datasets for use in deep learning projects.

The documentation for the functions shows example usages,
see [the wiki](https://github.com/cdmacfadyen/MedPicPy/wiki)
for examples of how the library is used on real datasets 
or check out the [GitHub page](https://github.com/cdmacfadyen/MedPicPy) to install/contribute.
"""
import os
import shutil

from .parsing import *

from .io import allocate_array
from .io import load_image
from .io import load_series

from .paths import *

from . import config

cache_dir = config.cache_location + "/medpicpy_cache/"

if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
else:
    for cache_file in os.listdir(cache_dir):
        if os.path.isfile(cache_file):
            os.remove(cache_dir + cache_file)

def clear_cache():
    for cache_file in os.listdir(cache_dir):
        os.remove(cache_dir + cache_file)