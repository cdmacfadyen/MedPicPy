MedPicPy is a Python library to simplify ingesting medical imaging 
datasets for feeding in to deep learning pipelines. 

Medical imaging datasets can be difficult to read in with many 
different file formats and dataset structures. MedPicPy
provides functions to abstract over these difficulties 
and turn the data into an easy to use Numpy array. 
MedPicPy is built on SimpleITK and OpenCV so it 
can read many imaging formats. 
### Table of Contents
- [Why use MedPicPy?](#why-use-medpicpy)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Code Example](#code-example)
  - [mini-MIAS Breast Cancer Classification](#mini-mias-breast-cancer-classification)
    - [Full Script](#full-script)
- [API Reference](#api-reference)
- [Contribute](#contribute)

## Why use MedPicPy?
* Turns data straight into numpy array format which to be fed 
into a machine learning model.
* Streamlines reading in data so you can focus on the model. 
* Simple functions that work with 2D, 3D or higher dimensional data.

## Installation
todo

## Basic Usage
Generally for a machine learning dataset the 
metadata about the image will be stored in a csv, in the 
directory structure, or a combination of those two things. 
This package has functions for obtaining paths to 
images by searching a dataset for paths containing 
certain strings (e.g. "CT" or "DX"). These paths can
then be passed into a MedPicPy image loading function
which takes the paths and returns the image data in 
numpy format, ready to be used in a machine learning model. 
See the examples below. 

## Code Example
The [wiki page](https://github.com/cdmacfadyen/MedPicPy/wiki) contains 
several examples of how this can be used with different kinds of dataset.
Here is an example of how to ingest the mini-MIAS dataset for 
breast cancer segmentation. 

### mini-MIAS Breast Cancer Classification
You can find this dataset at [this link](http://peipa.essex.ac.uk/info/mias.html). It's small (~100Mb) so its a good place to get started with medical imaging data. Once you download it the metadata is contained in the README, so open that and copy it into a new file. For this example it has been moved into a file called `data.txt`. 

Import pandas and medpicpy and read the data using pandas.
```python
import medpicpy as med
import pandas as pd

description = pd.read_csv("mini-MIAS/data.txt", header=None, delim_whitespace=True) # delim whitespace because the data is space separated
```
Next we need to format the data so that we can feed it into our csv reading function. Currently the image names in the metadata do not match the actual image names. The lambda function appends ".pgm" to the end of all of the image names in this dataframe. 
```python
description[0] = description[0].apply(lambda x : "{}.pgm".format(x)) # append .pgm to image names
```
Now we can load in all the images with `load_images_from_csv` which takes the dataframe, where the image names are in the dataframe, the path to the image directory, and the desired output shape of each image (which will depend on the model you are using). This loads all of the images in to memory.
```python
images = med.load_images_from_csv(description, 0, "mini-MIAS/", (224, 224))
```
The mini-MIAS data also has class and bounding box information and we can read those in too. First the data needs cleaned. We also resized the images in the last step so we need to move the bounding boxes by the right amount. We know from the metadata that the images were originally 1024 x 1024 so we can find the scale factor by finding our output image size over our input image size.
```python
classes = description[3]
classes = classes.fillna("N")

# convert bounding box data to numeric data type
description[4] = pd.to_numeric(description[4], errors="coerce")
description[5] = pd.to_numeric(description[5], errors="coerce")

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
```
You will probably want to convert your class data to a one-hot array, 
sklearn's [OneHotEncoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html#sklearn.preprocessing.OneHotEncoder) is 
useful for this. 
#### Full Script
See the full script here with a some simple visualisation code at the end.
```python
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

``` 

## API Reference
There is an API reference on the [pages site](https://cdmacfadyen.github.io/MedPicPy/). It contains detailed documentation and examples of how functions can be used. 

## Contribute
Feel free to contribute! Check out the issues if you 
want to find something to do, or add an issue if you think the 
package could be extended. Pull requests will be accepted provided 
they don't break anything and the feature is easy to use. 

Please ensure that all modules/functions added have docstrings, ideally
with an example usage and then run `./scripts/makedocs.sh` to add the 
documentation to the pages site.