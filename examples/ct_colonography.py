import matplotlib.pyplot as plt
import pandas as pd

import medpicpy as med

# read in one of the csvs
df = pd.read_csv("colonography/large_polyps.csv")

# find the paths that start with the patient ID's and contain "PRONE"
paths = med.get_paths_from_ids("colonography/CT COLONOGRAPHY/",
    df["TCIA Number"],
    path_filters=["PRONE"],
    read_individual_files=False)

# if you didn't download the full dataset
# you will need to find remove results that don't have scans
missing_paths = [index for index, element in enumerate(paths) if element is None]

# now we can remove those patients from the metadata since we have 
# no scan for them
slices_to_take = df["Slice# polyp Prone"]
slices_to_take = slices_to_take.drop(missing_paths) 
paths = [path for path in paths if path is not None]

# the slices to take from an image are separated by a "/", 
# so we need to split on / and then cast to numeric data type 
slices_to_take = slices_to_take.apply(lambda x : str(x).split("/")) 
slices_to_take = [[int(slice_index) for slice_index in slices] for slices in list(slices_to_take)]

# now we pass this information in to the function 
# along with a desired output shape for each slice
filtered_slices = med.load_specific_slices_from_series(paths,
    (128, 128), 
    slices_to_take)

# display one of the images
fig, ax = plt.subplots()
ax.imshow(filtered_slices[0], cmap="gray")
plt.show()