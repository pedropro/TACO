'''
This script downloads TACO's images from Flickr given an annotation json file
Code written by Pedro F. Proenza, 2019
'''

import os.path
import argparse
import json
from PIL import Image
import requests
from io import BytesIO
import sys
import concurrent.futures




file_paths_and_url=[]
parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', required=False, default= './data/annotations.json', help='Path to annotations')
args = parser.parse_args()

dataset_dir = os.path.dirname(args.dataset_path)

def downloadImage(url,file_path):
    if not os.path.isfile(file_path):
        return

    try:
        response = requests.get(url_original)
        img = Image.open(BytesIO(response.content))
        if img._getexif():
            img.save(file_path, exif=img.info["exif"])
        else:
            img.save(file_path)
    except Exception as e:
        print ("Exception: ",e," at ",url_original,file_path)

with open(args.dataset_path, 'r') as f:
    annotations = json.loads(f.read())

    nr_images = len(annotations['images'])
    for i in range(nr_images):
        image = annotations['images'][i]

        file_name = image['file_name']
        url = image['flickr_url']

        file_path = os.path.join(dataset_dir, file_name)

        # Create subdir if necessary
        subdir = os.path.dirname(file_path)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

        file_paths_and_url.append((url,file_path))

with concurrent.futures.ThreadPoolExecutor(max_workers=100) as executor:
    future_to_url = {executor.submit(downloadImage, *entry): entry for entry in file_paths_and_url}
    for future in concurrent.futures.as_completed(future_to_url):
        try:
            data = future.result()
        except Exception as exc:
            data = str(type(exc))

    sys.stdout.write('Finished. The dataset is available under data :) \n')


