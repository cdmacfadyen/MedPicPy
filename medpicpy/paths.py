"""
medpicpy's functions for finding and filtering paths 
to image data.
"""

import glob

from .utils import remove_sub_paths

def get_paths_to_images(data_dir, extension):
    paths = glob.glob(data_dir + "/**/*" + extension, recursive=True)

    return paths

def get_paths_from_ids(data_dir, ids, path_filters=[""]):
    """Read in a dataset from a list of patient ids, optionally filtering
    the path. i.e. (i.e. ["CT", "supine"])

    Args:
        data_dir (str): path to dataset
        ids (list or array-like): list of ids to read in, assuming each 
        id is a directory in the dataset (e.g. TCIA datasets)
        path_filters (list, optional): Any filters to apply to the path.
            Defaults to [""].

    Returns:
        array: All paths that match the ids with filters, 
            in the same order as ids
    """
    paths = []
    for id_number in ids:
        paths_for_id = glob.glob(data_dir + "/" + id_number + "/**/", recursive=True)
        for path_filter in path_filters:
            paths_for_id = [path for path in paths_for_id if path_filter in path]
        if paths_for_id:
            paths_for_id = remove_sub_paths(paths_for_id)
        if not paths_for_id:    #TODO: doesn't work on a filter object
            paths.append(None)
            print("Warn: Could not find any paths for id {}".format(id_number))
        else:
            paths.extend(paths_for_id)  #TODO: find the longest one per id

        
    return paths