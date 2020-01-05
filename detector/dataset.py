"""

Author: Pedro F. Proenza

"""

import os
import json
import numpy as np
import copy
import utils

from PIL import Image, ExifTags

from pycocotools.coco import COCO

class Taco(utils.Dataset):

    def load_taco(self, dataset_dir, round, subset, class_ids=None,
                  class_map=None, return_taco=False, auto_download=False):
        """Load a subset of the TACO dataset.
        dataset_dir: The root directory of the TACO dataset.
        round: split number
        subset: which subset to load (train, val, test)
        class_ids: If provided, only loads images that have the given classes.
        class_map: Dictionary used to assign original classes to new class system
        return_coco: If True, returns the COCO object.
        auto_download: Automatically download and unzip MS-COCO images and annotations
        """

        # TODO: Once we got the server running
        # if auto_download is True:
        #     self.auto_download(dataset_dir, subset, year)
        ann_filepath = os.path.join(dataset_dir , 'annotations')
        if round != None:
            ann_filepath += "_" + str(round) + "_" + subset + ".json"
        else:
            ann_filepath += ".json"

        assert os.path.isfile(ann_filepath)

        # Load dataset
        dataset = json.load(open(ann_filepath, 'r'))

        # Replace dataset original classes before calling the coco Constructor
        # Some classes may be assigned background to remove them from the dataset
        self.replace_dataset_classes(dataset, class_map)

        taco_alla_coco = COCO()
        taco_alla_coco.dataset = dataset
        taco_alla_coco.createIndex()

        # Add images and classes except Background
        # Definitely not the most efficient way
        image_ids = []
        background_id = -1
        class_ids = sorted(taco_alla_coco.getCatIds())
        for i in class_ids:
            class_name = taco_alla_coco.loadCats(i)[0]["name"]
            if class_name != 'Background':
                self.add_class("taco", i, class_name)
                image_ids.extend(list(taco_alla_coco.getImgIds(catIds=i)))
            else:
                background_id = i
        image_ids = list(set(image_ids))

        if background_id > -1:
            class_ids.remove(background_id)

        print('Number of images used:', len(image_ids))

        # Add images
        for i in image_ids:
            self.add_image(
                "taco", image_id=i,
                path=os.path.join(dataset_dir, taco_alla_coco.imgs[i]['file_name']),
                width=taco_alla_coco.imgs[i]["width"],
                height=taco_alla_coco.imgs[i]["height"],
                annotations=taco_alla_coco.loadAnns(taco_alla_coco.getAnnIds(
                    imgIds=[i], catIds=class_ids, iscrowd=None)))
        if return_taco:
            return taco_alla_coco

    def add_transplanted_dataset(self, dataset_dir, class_map = None):

        # Load dataset
        ann_filepath = os.path.join(dataset_dir, 'annotations.json')
        dataset = json.load(open(ann_filepath, 'r'))

        # Map dataset classes
        self.replace_dataset_classes(dataset, class_map)

        taco_alla_coco = COCO()
        taco_alla_coco.dataset = dataset
        taco_alla_coco.createIndex()

        class_ids = sorted(taco_alla_coco.getCatIds())

        # Select images by class
        # Add images
        image_ids = []
        background_id = -1
        for i in class_ids:
            class_name = taco_alla_coco.loadCats(i)[0]["name"]
            if class_name != 'Background':
                image_ids.extend(list(taco_alla_coco.getImgIds(catIds=i)))
                # TODO: Select how many
            else:
                background_id = i
        image_ids = list(set(image_ids))

        if background_id > -1:
            class_ids.remove(background_id)

        # Retrieve list of training image ids
        train_image_ids = [x['id'] for x in self.image_info]

        nr_train_images_so_far = len(train_image_ids)

        # Add images
        transplant_counter = 0
        for i in image_ids:
            if taco_alla_coco.imgs[i]['source_id'] in train_image_ids:
                transplant_counter += 1
                self.add_image(
                    "taco", image_id=i+nr_train_images_so_far,
                    path=os.path.join(dataset_dir, taco_alla_coco.imgs[i]['file_name']),
                    width=taco_alla_coco.imgs[i]["width"],
                    height=taco_alla_coco.imgs[i]["height"],
                    annotations=taco_alla_coco.loadAnns(taco_alla_coco.getAnnIds(
                        imgIds=[i], catIds=class_ids, iscrowd=None)))

        print('Number of transplanted images added: ', transplant_counter, '/', len(image_ids))

    def load_image(self, image_id):
        """Load the specified image and return as a [H,W,3] Numpy array."""

        # Load image. TODO: do this with opencv to avoid need to correct orientation
        image = Image.open(self.image_info[image_id]['path'])
        img_shape = np.shape(image)

        # load metadata
        exif = image._getexif()
        if exif:
            exif = dict(exif.items())
            # Rotate portrait images if necessary (274 is the orientation tag code)
            if 274 in exif:
                if exif[274] == 3:
                    image = image.rotate(180, expand=True)
                if exif[274] == 6:
                    image = image.rotate(270, expand=True)
                if exif[274] == 8:
                    image = image.rotate(90, expand=True)

        # If has an alpha channel, remove it for consistency
        if img_shape[-1] == 4:
            image = image[..., :3]

        return np.array(image)

    def auto_download(self, dataDir, dataType, dataYear):
        """TODO: Download the TACO dataset/annotations if requested."""


    def load_mask(self, image_id):
        """Load instance masks for the given image.

        Different datasets use different ways to store masks. This
        function converts the different mask format to one format
        in the form of a bitmap [height, width, instances].

        Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """

        image_info = self.image_info[image_id]

        instance_masks = []
        class_ids = []
        annotations = self.image_info[image_id]["annotations"]
        # Build mask of shape [height, width, instance_count] and list
        # of class IDs that correspond to each channel of the mask.
        for annotation in annotations:
            class_id = self.map_source_class_id("taco.{}".format(annotation['category_id']))
            if class_id:
                m = utils.annToMask(annotation, image_info["height"],image_info["width"])
                # Some objects are so small that they're less than 1 pixel area
                # and end up rounded out. Skip those objects.
                if m.max() < 1:
                    continue
                # Is it a crowd? If so, use a negative class ID.
                if annotation['iscrowd']:
                    # Use negative class ID for crowds
                    class_id *= -1
                    # For crowd masks, annToMask() sometimes returns a mask
                    # smaller than the given dimensions. If so, resize it.
                    if m.shape[0] != image_info["height"] or m.shape[1] != image_info["width"]:
                        m = np.ones([image_info["height"], image_info["width"]], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        # Pack instance masks into an array
        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            # Call super class to return an empty mask
            return super(Taco, self).load_mask(image_id)

    def replace_dataset_classes(self, dataset, class_map):
        """ Replaces classes of dataset based on a dictionary"""
        class_new_names = list(set(class_map.values()))
        class_new_names.sort()
        class_originals = copy.deepcopy(dataset['categories'])
        dataset['categories'] = []
        class_ids_map = {}  # map from old id to new id

        # Assign background id 0
        has_background = False
        if 'Background' in class_new_names:
            if class_new_names.index('Background') != 0:
                class_new_names.remove('Background')
                class_new_names.insert(0, 'Background')
            has_background = True

        # Replace categories
        for id_new, class_new_name in enumerate(class_new_names):

            # Make sure id:0 is reserved for background
            id_rectified = id_new
            if not has_background:
                id_rectified += 1

            category = {
                'supercategory': '',
                'id': id_rectified,  # Background has id=0
                'name': class_new_name,
            }
            dataset['categories'].append(category)
            # Map class names
            for class_original in class_originals:
                if class_map[class_original['name']] == class_new_name:
                    class_ids_map[class_original['id']] = id_rectified

        # Update annotations category id tag
        for ann in dataset['annotations']:
            ann['category_id'] = class_ids_map[ann['category_id']]
