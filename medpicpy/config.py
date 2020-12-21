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