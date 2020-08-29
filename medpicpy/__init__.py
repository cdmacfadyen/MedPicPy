"""
MedPicPy provides a simple API for loading medical 
imaging datasets for use in deep learning projects.

The documentation for the functions shows example usages,
or see [the wiki](https://github.com/cdmacfadyen/MedPicPy/wiki)
for examples of how the library is used on real datasets. 
"""
import os
import shutil

from .parsing import *
from .io import *
from .paths import *

cache_dir = "medpicpy_cache/"

if not os.path.exists(cache_dir):
    os.mkdir(cache_dir)
else:
    for cache_file in os.listdir(cache_dir):
        os.remove(cache_dir + cache_file)