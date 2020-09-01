import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import medpicpy as med

description = pd.read_csv("mini-MIAS/data.txt", header=None, delim_whitespace=True) # delim whitespace because the data is space separated

description[0] = description[0].apply(lambda x : "{}.pgm".format(x))
array = med.load_images_from_csv(description, 0, "mini-MIAS/", (224, 224))


description[4] = pd.to_numeric(description[4], errors="coerce")
description[5] = pd.to_numeric(description[5], errors="coerce")

classes = description[3]
classes = classes.fillna("N")   # N for normal

x_scale_factor = 224 / 1024
y_scale_factor = 224 / 1024

xs, ys, widths, heights = med.load_bounding_boxes_from_csv(
    description, 
    4, 
    5, 
    6, 
    6, 
    x_scale_factor=x_scale_factor, 
    y_scale_factor=y_scale_factor
)

print(classes)
image = array[0]

fig, ax = plt.subplots()

ax.imshow(image, cmap="gray")
bbox = patches.Circle((xs[0], ys[0]), widths[0],
    linewidth=1,
    edgecolor="r",
    fill=False)

ax.add_patch(bbox)
plt.show()