"""

Author: Pedro F. Proenza

This source modifies and extends the work done by:

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License
Written by Waleed Abdulla

------------------------------------------------------------

Usage:

    # First make sure you have split the dataset into train/val/test set. e.g. You should have annotations_0_train.json
    # in your dataset dir.
    # Otherwise, You can do this by calling
    python3 split_dataset.py --dataset_dir ../data

    # Train a new model starting from pre-trained COCO weights on train set split #0
    python3 -W ignore detector.py train --model=coco --dataset=../data --class_map=./taco_config/map_10.csv --round 0

    # Continue training a model that you had trained earlier
    python3 -W ignore detector.py train  --dataset=../data --model=<model_name> --class_map=./taco_config/map_10.csv --round 0

    # Continue training the last model you trained with image augmentation
    python3 detector.py train --dataset=../data --model=last --round 0 --class_map=./taco_config/map_10.csv --use_aug

    # Test model and visualize predictions image by image
    python3 detector.py test --dataset=../data --model=<model_name> --round 0 --class_map=./taco_config/map_10.csv

    # Run COCO evaluation on a trained model
    python3 detector.py evaluate --dataset=../data --model=<model_name> --round 0 --class_map=./taco_config/map_10.csv

    # Check Tensorboard
    tensorboard --logdir ./models/logs

"""

import os
import time
import numpy as np
import json
import csv
import random
from imgaug import augmenters as iaa

from dataset import Taco
import model as modellib
from model import MaskRCNN
from config import Config
import visualize
import utils
import matplotlib.pyplot as plt

from pycocotools.cocoeval import COCOeval
from pycocotools import mask as maskUtils

# Root directory of the models
ROOT_DIR = os.path.abspath("./models")

# Path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Testing functions
############################################################

def test_dataset(model, dataset, nr_images):

    for i in range(nr_images):

        image_id = dataset.image_ids[i] if nr_images == len(dataset.image_ids) else random.choice(dataset.image_ids)

        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
        info = dataset.image_info[image_id]

        r = model.detect([image], verbose=0)[0]

        print(r['class_ids'].shape)
        if r['class_ids'].shape[0]>0:
            r_fused = utils.fuse_instances(r)
        else:
            r_fused = r

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 16))

        # Display predictions
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
                                    dataset.class_names, r['scores'], title="Predictions", ax=ax1)

        visualize.display_instances(image, r_fused['rois'], r_fused['masks'], r_fused['class_ids'],
                                     dataset.class_names, r_fused['scores'], title="Predictions fused", ax=ax2)

        # # Display ground truth
        visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, dataset.class_names, title="GT", ax=ax3)

        # Voilà
        plt.show()

############################################################
#  COCO Evaluation
############################################################

def build_coco_results(dataset, image_ids, rois, class_ids, scores, masks):
    """Arrange resutls to match COCO specs in http://cocodataset.org/#format
    """
    # If no results, return an empty list
    if rois is None:
        return []

    results = []
    for image_id in image_ids:
        # Loop through detections
        for i in range(rois.shape[0]):
            class_id = class_ids[i]
            score = scores[i]
            bbox = np.around(rois[i], 1)
            mask = masks[:, :, i]

            result = {
                "image_id": image_id,
                "category_id": dataset.get_source_class_id(class_id, "taco") if dataset.num_classes>2 else 1,
                "bbox": [bbox[1], bbox[0], bbox[3] - bbox[1], bbox[2] - bbox[0]],
                "score": score,
                "segmentation": maskUtils.encode(np.asfortranarray(mask))
            }
            results.append(result)
    return results


def evaluate_coco(model, dataset, coco, eval_type="bbox", limit=0, image_ids=None):
    """Runs official COCO evaluation.
    dataset: A Dataset object with valiadtion data
    eval_type: "bbox" or "segm" for bounding box or segmentation evaluation
    limit: if not 0, it's the number of images to use for evaluation
    """
    # Pick TACO images from the dataset
    image_ids = image_ids or dataset.image_ids

    # Limit to a subset
    if limit:
        image_ids = image_ids[:limit]

    # Get corresponding TACO image IDs.
    taco_image_ids = [dataset.image_info[id]["id"] for id in image_ids]

    t_prediction = 0
    t_start = time.time()
    results = []
    for i, image_id in enumerate(image_ids):
        # Load image
        image = dataset.load_image(image_id)

        # Run detection
        t = time.time()
        r = model.detect([image], verbose=0)[0]
        # r = utils.fuse_instances(r)
        t_prediction += (time.time() - t)


        if not model.config.DETECTION_SCORE_RATIO:
            scores = r["scores"]
        else:
            scores = r["scores"]/(r["full_scores"][:,0]+0.0001)


        # Convert results to COCO format
        # Cast masks to uint8 because COCO tools errors out on bool
        image_results = build_coco_results(dataset, taco_image_ids[i:i + 1],
                                           r["rois"], r["class_ids"],
                                           scores,
                                           r["masks"].astype(np.uint8))
        results.extend(image_results)

    # Load results. This modifies results with additional attributes.
    coco_results = coco.loadRes(results)

    # utils.compute_confusion_matrix(coco_results, coco)

    # Evaluate
    cocoEval = COCOeval(coco, coco_results, eval_type)
    cocoEval.params.imgIds = taco_image_ids
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    print("Prediction time: {}. Average {}/image".format(
        t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run Mask R-CNN on TACO.')
    parser.add_argument("command", metavar="<command>",help="Opt: 'train', 'evaluate', 'test'")
    parser.add_argument('--model', required=True, metavar="/path/weights.h5", help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--dataset', required=True, metavar="/path/dir", help='Directory of the dataset')
    parser.add_argument('--round', required=True, type=int, help='Split number')
    parser.add_argument('--lrate', required=False, default=0.001, type=float, help='learning rate')
    parser.add_argument('--use_aug', dest='aug', action='store_true')
    parser.set_defaults(aug=False)
    parser.add_argument('--use_transplants', required=False, default=None, help='Path to transplanted dataset')
    parser.add_argument('--class_map', required=True, metavar="/path/file.csv", help=' Target classes')

    args = parser.parse_args()
    print("Command: ", args.command)
    print("Model: ", args.model)
    print("Dataset: ", args.dataset)
    print("Logs: ", DEFAULT_LOGS_DIR)

    # Read map of target classes
    class_map = {}
    map_to_one_class = {}
    with open(args.class_map) as csvfile:
        reader = csv.reader(csvfile)
        class_map = {row[0]: row[1] for row in reader}
        map_to_one_class = {c: 'Litter' for c in class_map}

    # Load datasets
    if args.command == "train":

        # Training dataset.
        dataset_train = Taco()
        dataset_train.load_taco(args.dataset, args.round, "train", class_map=class_map, auto_download=None)
        if args.use_transplants:
            dataset_train.add_transplanted_dataset(args.use_transplants, class_map=class_map)
        dataset_train.prepare()
        nr_classes = dataset_train.num_classes

        # Validation dataset
        dataset_val = Taco()
        dataset_val.load_taco(args.dataset, args.round, "val", class_map=class_map, auto_download=None)
        dataset_val.prepare()
    else:
        # Test dataset
        dataset_test = Taco()
        taco = dataset_test.load_taco(args.dataset, args.round, "test", class_map=class_map, return_taco=True)
        dataset_test.prepare()
        nr_classes = dataset_test.num_classes

    # Configurations
    if args.command == "train":
        class TacoTrainConfig(Config):
            NAME = "taco"
            IMAGES_PER_GPU = 2
            GPU_COUNT = 1
            STEPS_PER_EPOCH = min(1000,int(dataset_train.num_images/(IMAGES_PER_GPU*GPU_COUNT)))
            USE_MINI_MASK = True
            MINI_MASK_SHAPE = (512, 512)
            NUM_CLASSES = nr_classes
            LEARNING_RATE = args.lrate
        config = TacoTrainConfig()
    else:
        class TacoTestConfig(Config):
            NAME = "taco"
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0 if args.command == "evaluate" else 10
            NUM_CLASSES = nr_classes
            USE_OBJECT_ZOOM = False
        config = TacoTestConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = MaskRCNN(mode="training", config=config, model_dir=DEFAULT_LOGS_DIR)
    else:
        model = MaskRCNN(mode="inference", config=config, model_dir=DEFAULT_LOGS_DIR)

    # Select weights file to load
    if args.model.lower() == "coco":
        model_path = COCO_MODEL_PATH
        # Download weights file
        if not os.path.exists(model_path):
            utils.download_trained_weights(model_path)
    elif args.model.lower() == "last":
        # Find last trained weights
        model_path = model.find_last()[1]
    elif args.model.lower() == "imagenet":
        # Start from ImageNet trained weights
        model_path = model.get_imagenet_weights()
    else:
        _, model_path = model.get_last_checkpoint(args.model)

    # Load weights
    if args.model.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(model_path, None, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(model_path, model_path, by_name=True)

    # Train or evaluate
    if args.command == "train":

        if args.aug:
            if not config.USE_OBJECT_ZOOM:
                # Image Augmentation Pipeline
                augmentation_pipeline = iaa.Sequential([
                    iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="AWGN"),
                    iaa.GaussianBlur(sigma=(0.0, 3.0), name="Blur"),
                    # iaa.Dropout([0.0, 0.05], name='Dropout'), # drop 0-5% of all pixels
                    iaa.Fliplr(0.5),
                    iaa.Add((-20, 20),name="Add"),
                    iaa.Multiply((0.8, 1.2), name="Multiply"),
                    iaa.Affine(scale=(0.8, 2.0)),
                    iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}),
                    iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees
                ], random_order=True)
            else:
                # Nevermind the image translation and scaling as this is done already during zoom in
                augmentation_pipeline = iaa.Sequential([
                    iaa.AdditiveGaussianNoise(scale=0.01 * 255, name="AWGN"),
                    iaa.GaussianBlur(sigma=(0.0, 3.0), name="Blur"),
                    # iaa.Dropout([0.0, 0.05], name='Dropout'), # drop 0-5% of all pixels
                    iaa.Fliplr(0.5),
                    iaa.Add((-20, 20), name="Add"),
                    iaa.Multiply((0.8, 1.2), name="Multiply"),
                    iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees
                ], random_order=True)
        else:
            augmentation_pipeline = None

        # Save training meta to log dir
        training_meta = {
            'number of classes': nr_classes,
            'round': args.round,
            'use_augmentation': args.aug,
            'use_transplants': args.use_transplants != None,
            'learning_rate': config.LEARNING_RATE,
            'layers_trained': 'all'}

        subdir = os.path.dirname(model.log_dir)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

        if not os.path.isdir(model.log_dir):
            os.mkdir(model.log_dir)

        train_meta_file = model.log_dir + '_meta.json'
        with open(train_meta_file, 'w+') as f:
            f.write(json.dumps(training_meta))

        # Training all layers
        model.train(dataset_train, dataset_val,learning_rate=config.LEARNING_RATE, epochs=100,
                    layers='all', augmentation=augmentation_pipeline)

        # Training last layers
        # Finetune layers from ResNet stage 4 and up
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE,
        #             epochs=120,
        #             layers='4+')

        # Training only heads
        # model.train(dataset_train, dataset_val,
        #             learning_rate=config.LEARNING_RATE / 10,
        #             epochs=160,
        #             layers='heads')

    elif args.command == "evaluate":
        nr_eval_images = len(dataset_test.image_ids)
        print("Running COCO evaluation on {} images.".format(nr_eval_images))
        evaluate_coco(model, dataset_test, taco, "segm", limit=0)

        # Binary problem (litter/not litter)
        # dataset_test_binary = Taco()
        # taco_binary = dataset_test_binary.load_taco(args.dataset, args.round, "test", class_map=map_to_one_class, return_taco=True)
        # dataset_test_binary.prepare()
        # evaluate_coco(model, dataset_test_binary, taco_binary, "segm", limit=0)

    elif args.command == "test":
        test_dataset(model, dataset_test, len(dataset_test.image_ids))
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'evaluate'".format(args.command))
