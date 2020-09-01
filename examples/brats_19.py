import pandas as pd
import medpicpy as med

csv = "brats_2019/MICCAI_BraTS_2019_Data_Training/survival_data.csv"
metadata = pd.read_csv(csv)
ids = metadata["BraTS19ID"]

data_path = "brats_2019/MICCAI_BraTS_2019_Data_Training/HGG"
filters =["flair", "t1ce","t1.", "t2"]

paths_for_modalities = [
    med.get_paths_from_ids(
    data_path,
    ids,
    path_filters=[modality]
) for modality in filters]

images = [
    med.load_series_from_paths(paths, (128, 128), range(60, 80))
    for paths in paths_for_modalities
]

for i in range(0, len(images)):
    print("{} shape: {}".format(filters[i], images[i].shape))
# (259, 20, 128, 128)

multi_modal = med.stack_modalities(images)
print("multimodal shape: ", multi_modal.shape)
# (259, 20, 128, 128, 4)

mask_paths = med.get_paths_from_ids(
    data_path,
    ids,
    path_filters=["seg"]
)

masks = med.load_series_from_paths(
    mask_paths,
    (128, 128),
    range(60, 80)
)

print("masks shape: ", masks.shape)
# (259, 20, 128, 128)