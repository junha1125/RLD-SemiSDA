import os
from skimage.io import imread
import torch
import torchvision
import torchxrayvision as xrv
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Resize MIMIC dataset images')
    parser.add_argument('--root_folder', type=str, required=True,
                      help='Root folder containing subfolders with JPG images')
    return parser.parse_args()

# Function to resize a single image
def resize_image(image_path):
    img = imread(image_path)
    img = xrv.datasets.normalize(img, maxval=255, reshape=True)
    trainsforms = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),xrv.datasets.XRayResizer(224)])
    img = trainsforms(img) 
    file_name = image_path[:-4].replace('files', 'files_224')
    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    np.save(file_name, img)
    print(f'Resized and overwrote: {image_path}')

def main():
    args = parse_args()
    
    # Iterate through subfolders and process JPG images
    for root, _, files in os.walk(args.root_folder):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_path = os.path.join(root, file)
                resize_image(image_path)

if __name__ == '__main__':
    main()