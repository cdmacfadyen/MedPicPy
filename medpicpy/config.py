import os

cache_location = "./medpicpy_cache/"

def set_cache_location(path):
    global cache_location
    cache_location = path
    if not os.path.exists(cache_location + "/medpicpy_cache/"):
        os.mkdir(cache_location + "/medpicpy_cache/")