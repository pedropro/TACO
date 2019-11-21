<p align="center">
<img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/logonav.png" width="25%"/>
</p>

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3242156.svg)](https://doi.org/10.5281/zenodo.3242156) 

TACO is a growing image dataset of waste in the wild. It contains images of litter taken under
diverse environments: woods, roads and beaches. These images are manually labeled and segmented
according to a hierarchical taxonomy to train and evaluate object detection algorithms. Currently,
images are hosted on Flickr and we have a server that is collecting more images and
annotations @ [tacodataset.org](http://tacodataset.org)


<div align="center">
  <div class="column">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/1.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/2.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/3.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/4.png" width="17%" hspace="3">
    <img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/5.png" width="17%" hspace="3">
  </div>
</div>
</br>

If you use this dataset and API in a publication, please cite us: &nbsp;
```
@misc{Taco19,
  author       = {Pedro F. Proença and Pedro Simões},
  title        = {TACO: Trash Annotations in Context Dataset},
  year         = 2019,
  doi          = {10.5281/zenodo.3242156},
  url          = {http://tacodataset.org}
}
```
For convenience, annotations are provided in COCO format. Check the metadata here:
http://cocodataset.org/#format-data

TACO is still relatively small, but it is growing. Stay tuned!

# News

**November 20, 2019** - TACO is officially open for new annotations: http://tacodataset.org/annotate

# Getting started

To download the dataset images simply issue
```
python3 download.py
```
Our API contains a notebook ``demo.pynb`` to inspect the dataset and visualize annotations. To use ``demo.pynb``, you require:
* [jupyter](https://jupyter.org/)
* [seaborn](https://seaborn.pydata.org/)
* [python cocoapi](https://github.com/cocodataset/cocoapi)

### Trash Detection

The implementation of [Mask-RCNN by Matterport](https://github.com/matterport/Mask_RCNN)  is included in ``/detector``
with a few modifications. Requirements are the same. Before using this, first use the `split_dataset.py` script to generate 
N random train, val, test subsets. For example, run this inside the directory `detector`:
```
python3 split_dataset.py --dataset_dir ../data
```

For further usage instructions, check ``detector/detector.py``.

As you can see [here](http://tacodataset.org/stats), most of the original classes of TACO have very few annotations, therefore these must be either left out or merged together. Depending on the problem, ``detector/taco_config`` contains several class maps to target classes, which maintain the most dominant classes, e.g., Can, Bottles and Plastic bags. Feel free to make your own classes.

<p align="center">
<img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/teaser.gif" width="75%"/></p>
