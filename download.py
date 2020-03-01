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
from threading import Thread
from time import sleep

NTHREADS=50

parser = argparse.ArgumentParser(description='')
parser.add_argument('--dataset_path', required=False, default= './data/annotations.json', help='Path to annotations')
args = parser.parse_args()

dataset_dir = os.path.dirname(args.dataset_path)

print('Note. If for any reason the connection is broken. Just call me again and I will start where I left.')

def download_img(url,fpath):
    # Load and Save Image
    resp = requests.get(url)
    img = Image.open(BytesIO(resp.content))
    if img._getexif():
        img.save(fpath, exif=img.info["exif"])
    else:
        img.save(fpath)

thrds=[]
# Load annotations
with open(args.dataset_path, 'r') as f:
    annotations = json.loads(f.read())

    nr_images = len(annotations['images'])
    for i in range(nr_images):

        image = annotations['images'][i]

        file_name = image['file_name']
        url_original = image['flickr_url']
        url_resized = image['flickr_640_url']

        file_path = os.path.join(dataset_dir, file_name)

        # Create subdir if necessary
        subdir = os.path.dirname(file_path)
        if not os.path.isdir(subdir):
            os.mkdir(subdir)

        if not os.path.isfile(file_path):
            thrds.append(Thread(target=download_img,args=(url_original,file_path,)))
            thrds[-1].setDaemon(True)
            thrds[-1].start()
            while len(thrds)>=NTHREADS:
                sleep(2)
                for ix,t in enumerate(thrds):
                    if not t.is_alive():
                        thrds.pop(ix)
        #     # Load and Save Image
        #     response = requests.get(url_original)
        #     img = Image.open(BytesIO(response.content))
        #     if img._getexif():
        #         img.save(file_path, exif=img.info["exif"])
        #     else:
        #         img.save(file_path)

        # Show loading bar
        i+=1
        bar_size = 30
        x = int(bar_size * i / nr_images)
        sys.stdout.write("%s[%s%s] - %i/%i\r" % ('Loading: ', "=" * x, "." * (bar_size - x), i, nr_images))
        sys.stdout.flush()

    sys.stdout.write('Finished\n')