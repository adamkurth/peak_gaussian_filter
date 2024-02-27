import os
import re
import h5py as h5
import argparse 
import numpy as np
import matplotlib.pyplot as plt
from collections import namedtuple

class GaussianMask:
    def __init__(self):
        self.cxfel_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.images_dir = self.__walk__(self.cxfel_root)
    
    def __walk__(self):
        # returns images/ directory
        start = self.cxfel_root
        for root, dirs, files in os.walk(start):
            if "images" in dirs:
                return os.path.join(root, "images")
        raise Exception("Could not find the 'images' directory starting from", start_path)

    def __args__():
        # 
        parser = argparse.ArgumentParser(description='Apply a Gaussian mask to an HDF5 image')
        parser.add_argument('--a', type=float, default=1.0, help='Gaussian scale factor for x-axis', required=False)
        parser.add_argument('--b', type=float, default=0.8, help='Gaussian scale factor for y-axis', required=False)
        return parser.parse_args()


    def __load_h5__(self, image_dir):
        # input argument is .../CXFEL/images/
        
        # Choose a random image
        image_files = os.listdir(image_dir)
        random_image = np.random.choice(image_files)

        # Join the random image to the images directory path
        image_path = os.path.join(image_dir, random_image)
        
        print("Loading image:", random_image)
        try:
            with h5.File(image_path, 'r') as f:
                data = f['entry/data/data'][:]
            return data, image_path
        except Exception as e:
            raise OSError(f"Failed to read {image_path}: {e}")
            
