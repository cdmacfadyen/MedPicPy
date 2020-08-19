"""
medpicpy's functions for finding and filtering paths 
to image data.
"""

import glob

from .utils import remove_sub_paths

def get_paths_to_images(data_dir, extension, path_filters=[""]):
    """Search directory and subdirectories for image with the given 
    extension.

    Optionally takes a list of strings to be applied as filters 
    to the path e.g. ["CT", "prone"] or ["flair"]. These paths 
    can then be passed to load_images_from_paths.

    get_paths_from_ids is preferred where possible since 
    this function could return results in a different order depending 
    on the machine. 

    Args:
        data_dir (str): path to root of dataset
        extension (str): file extension to search for
        path_filters (list, optional): filters to apply to paths. Defaults to [""].

    Returns:
        list: list of paths to images
    """
    paths = glob.glob(data_dir + "/**/*" + extension, recursive=True)
    
    if path_filters is not [""]:
        paths = filter_paths(paths, path_filters)

    return paths

def filter_paths(paths, filters):
    """Filters a list of paths so it only contains
    paths containing all of the given filters.

    Used by get_paths_to_images.
    ## Example
    ```python
    import medpicpy as med
    paths = ["data/ID-001/PRONE/1.dcm", "data/ID-001/SUPINE/1.dcm", "data/ID-002/PRONE/1.dcm", "data/ID-002/SUPINE/1.dcm"]
    filters = ["PRONE"]
    paths = med.filter_paths(paths, filters)
    print(paths)
    # ["data/ID-001/PRONE/1.dcm", "data/ID-002/PRONE/1.dcm"]
    ```
    Args:
        paths (array): array of paths
        filters (array): filters paths must contain

    Returns:
        array: paths that contain the filters
    """
    paths = [path for path in paths if all([path_filter in path for path_filter in filters])]
    return paths

def get_paths_from_ids(data_dir, 
    ids,
    path_filters=[""], 
    read_individual_files=True):
    """Read in a dataset from a list of patient ids, optionally filtering
    the path. e.g. ["CT", "supine", ".nii.gz"].

    Use this if your dataset has a structure like 
    'data_dir/patient_id/.../image'.
    Optionally searches for directories instead of 
    individual files, use this for e.g dicom series.
    You may want to include the file extension in the 
    filters.

    ## Example
    If there is dataset with structure:
    ```
    data/
        patient-data.csv
        ID-001/
            SCANS/
                CT/
                    prone.dcm
                    supine.dcm
                DX/
                    scan.nii.gz
        ID-002/
            SCANS/
                CT/
                    prone.dcm
                    supine.dcm
                DX/
                    scan.nii.gz
        ID-003/
            SCANS/
                CT/
                    prone.dcm
                    supine.dcm
                DX/
                    scan.nii.gz
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
    # ["data/ID-001/CT/prone.dcm", "data/ID-002/CT/prone.dcm", "data/ID-003/CT/prone.dcm"]
    ```

    With directory structure like so, with 
    images in a dicom series then set read_individual_files=False:
    ```
    data/
        patient-data.csv
        ID-001/
            SCANS/
                CT/
                    001.dcm
                    002.dcm
                DX/
                    scan.nii.gz
        ID-002/
            SCANS/
                CT/
                    001.dcm
                    002.dcm
                DX/
                    scan.nii.gz
        ID-003/
            SCANS/
                CT/
                    001.dcm
                    002.dcm
                DX/
                    scan.nii.gz
    ```
    then:
    ```python
    import pandas as pd
    import medpicpy as med

    description = pd.read_csv("data/patient-data.csv")
    patient_ids = description("id")
    filters = ["CT"]

    image_paths = med.get_paths_from_ids(
        "data/",
        patient_ids,
        filters,
        read_individual_files=False
    )

    print(image_paths)
    # ["data/ID-001/CT/", "data/ID-002/CT/", "data/ID-003/CT/"]
    ```
    Args:
        data_dir (str): path to dataset
        ids (list or array-like): list of ids to read in, assuming each 
            id is a directory in the dataset (e.g. TCIA datasets)
        path_filters (list, optional): Any filters to apply to the path.
            Defaults to [""].
        read_individual_files (bool, optional): specifies to look for 
            individual files or directories. Defaults to True.

    Returns:
        array: All paths that match the ids with filters, 
            in the same order as ids
    """
    paths = []
    for id_number in ids:
        paths_for_id = ""
        if read_individual_files:
            paths_for_id = glob.glob(data_dir + "/" + id_number + "/**/*", recursive=True)
        else:
            paths_for_id = glob.glob(data_dir + "/" + id_number + "/**/", recursive=True)

        for path_filter in path_filters:
            paths_for_id = [path for path in paths_for_id if path_filter in path]
        if paths_for_id:
            paths_for_id = remove_sub_paths(paths_for_id)
        if not paths_for_id:
            paths.append(None)
            print("Warn: Could not find any paths for id {}".format(id_number))
        else:
            paths.extend(paths_for_id)

        
    return paths