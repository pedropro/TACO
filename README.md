<p align="center">
<img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/logonav.png" width="25%"/>
</p>

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

For convenience, annotations are provided in COCO format. Check the metadata here:
http://cocodataset.org/#format-data

TACO is still relatively small, but it is growing. Stay tuned!

# Publications

For more details check our paper: https://arxiv.org/abs/2003.06975

If you use this dataset and API in a publication, please cite us using: &nbsp;
```
@article{taco2020,
    title={TACO: Trash Annotations in Context for Litter Detection},
    author={Pedro F Proença and Pedro Simões},
    journal={arXiv preprint arXiv:2003.06975},
    year={2020}
}
```

# News
**December 20, 2019** - Added more 785 images and 2642 litter segmentations. <br/>
**November 20, 2019** - TACO is officially open for new annotations: http://tacodataset.org/annotate

# Getting started

### Requirements 

To install the required python packages simply type
```
pip3 install -r requirements.txt
```
Additionaly, to use ``demo.pynb``, you will also need [coco python api](https://github.com/cocodataset/cocoapi). You can get this using
```
pip3 install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

### Download

To download the dataset images simply issue
```
python3 download.py
```
Alternatively, download from [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3242156.svg)](https://doi.org/10.5281/zenodo.3242156) 

Our API contains a jupyter notebook ``demo.pynb`` to inspect the dataset and visualize annotations.

**Unlabeled data**

A list of URLs for both unlabeled and labeled images is now also provided in `data/all_image_urls.csv`.
Each image contains one URL for each original image (second column) and one URL for a VGA-resized version (first column)
for images hosted by Flickr. If you decide to annotate these images using other tools, please make them public and contact us so we can keep track.

**Unofficial data**

Annotations submitted via our website are added weekly to `data/annotations_unofficial.json`. These have not yet been been reviewed by us -- some may be inaccurate or have poor segmentations. 
You can use the same command to download the respective images:
```
python3 download.py --dataset_path ./data/annotations_unofficial.json
```

### Trash Detection

The implementation of [Mask R-CNN by Matterport](https://github.com/matterport/Mask_RCNN)  is included in ``/detector``
with a few modifications. Requirements are the same. Before using this, the dataset needs to be split. You can either donwload our [weights and splits](https://github.com/pedropro/TACO/releases/tag/1.0) or generate these from scratch using the `split_dataset.py` script to generate 
N random train, val, test subsets. For example, run this inside the directory `detector`:
```
python3 split_dataset.py --dataset_dir ../data
```

For further usage instructions, check ``detector/detector.py``.

As you can see [here](http://tacodataset.org/stats), most of the original classes of TACO have very few annotations, therefore these must be either left out or merged together. Depending on the problem, ``detector/taco_config`` contains several class maps to target classes, which maintain the most dominant classes, e.g., Can, Bottles and Plastic bags. Feel free to make your own classes.

<p align="center">
<img src="https://raw.githubusercontent.com/wiki/pedropro/TACO/images/teaser.gif" width="75%"/></p>
