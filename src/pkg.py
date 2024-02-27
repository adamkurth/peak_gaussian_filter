import os
import re
import h5py as h5
import argparse  
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage.feature import peak_local_max
from scipy.signal import find_peaks, peak_prominences, peak_widths
from skimage import filters
from skimage.filters import median
from skimage.morphology import disk
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity

from collections import namedtuple


class GaussianMask:
    def __init__(self):
        self.cxfel_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.images_dir = self._walk()
        self.loaded_image, self.image_path = self._load_h5(self.images_dir)
        self.args = self._args()
        self.params = (self.args.a, self.args.b)
        self.mask = self.gaussian_mask()
        self.peaks = self._find_peaks()

    @staticmethod
    def _args():
        parser = argparse.ArgumentParser(description='Apply a Gaussian mask to an HDF5 image and detect peaks with noise reduction')
        # Gaussian mask parameters
        parser.add_argument('--a', type=float, default=1.0, help='Gaussian scale factor for x-axis', required=False)
        parser.add_argument('--b', type=float, default=0.8, help='Gaussian scale factor for y-axis', required=False)
        
        # Noise reduction parameter
        parser.add_argument('--median_filter_size', type=int, default=3, help='Size of the median filter for noise reduction', required=False)
        
        # Peak detection parameters
        parser.add_argument('--min_distance', type=int, default=10, help='Minimum number of pixels separating peaks', required=False)
        parser.add_argument('--prominence', type=float, default=1.0, help='Required prominence of peaks', required=False)
        parser.add_argument('--width', type=float, default=5.0, help='Required width of peaks', required=False)
        parser.add_argument('--min_prominence', type=float, default=0.1, help='Minimum prominence to consider a peak', required=False)
        parser.add_argument('--min_width', type=float, default=1.0, help='Minimum width to consider a peak', required=False)
        parser.add_argument('--threshold_value', type=float, default=500, help='Threshold value for peak detection', required=False)
        # Region of interest parameters for peak analysis
        parser.add_argument('--region_size', type=int, default=5, help='Size of the region to extract around each peak for analysis', required=False)
        
        return parser.parse_args()
        
    def _walk(self):
        # returns images/ directory
        start = self.cxfel_root
        for root, dirs, files in os.walk(start):
            if "images" in dirs:
                return os.path.join(root, "images")
        raise Exception("Could not find the 'images' directory starting from", start)
    
    def _load_h5(self, image_dir):
        image_files = [f for f in os.listdir(image_dir) if f.endswith('.h5')]
        # choose a random image to load
        random_image = np.random.choice(image_files)
        image_path = os.path.join(image_dir, random_image)
        print("Loading image:", random_image)
        try:
            with h5.File(image_path, 'r') as file:
                data = file['entry/data/data'][:]
            return data, image_path
        except Exception as e:
            raise OSError(f"Failed to read {image_path}: {e}")
    
    def _find_peaks(self, use_1d=False):
        """
        This function processes the loaded image to find and refine peaks.
        It first reduces noise using a median filter, then applies a Gaussian mask.
        After initial peak detection, it refines the peaks based on prominence and width criteria.
        """
        # Noise reduction
        denoised_image = median(self.loaded_image, disk(self.args.median_filter_size))
        # Gaussian mask application
        masked_image = self._apply(denoised_image)
        # Initial peak detection
        coordinates = peak_local_max(masked_image, min_distance=self.args.min_distance)
        # Peak refinement
        if use_1d:
            refined_peaks = self._refine_peaks_1d(masked_image)
        else: 
            refined_peaks = self._refine_peaks_2d(masked_image, coordinates)
        return refined_peaks
    
    def _refine_peaks(self, masked_image, coordinates):
        """
        Refines detected peaks using the prominence and width criteria.
        Extracts a region around each peak and analyzes it to determine the true peaks.
        """
        refined_peaks = []
        for coord in coordinates:
            x, y = coord
            peak_region = ArrayRegion(masked_image)
            region = peak_region.extract_region(x, y, self.args.region_size)
            # Peak property calculation 
            peaks, properties = find_peaks(region, prominence=self.args.prominence, width=self.args.width)
            # Peak confirmation based on prominence and width
            if peaks.size > 0:
                prominences = peak_prominences(region, peaks)
                widths = peak_widths(region, peaks)
                if np.any(prominences > self.args.min_prominence) and np.any(widths > self.args.min_width):
                    refined_peaks.append((x, y))
        return refined_peaks
    
    def _refine_peaks_2d(self, images, coordinates):
        """
        Refines detected peaks in a 2D image based on custom criteria.
        """
        threshold = self.args.threshold_value
        refined_peaks = []
        image = self.loaded_image
        for x, y in coordinates:
            # Extract a small region around the peak
            region = image[max(0, x-10):x+10, max(0, y-10):y+10]
            
            # Example criterion: Check if the peak is significantly brighter than the median of its surrounding
            if image[x, y] > np.median(region) + threshold:
                refined_peaks.append((x, y))
        return refined_peaks

    def _refine_peaks_1d(self, image, axis=0):
        """
        Applies 1D peak refinement to each row or column of the image.
        axis=0 means each column is treated as a separate 1D signal; axis=1 means each row.
        """
        refined_peaks = []
        num_rows, num_columns = image.shape
        for index in range(num_columns if axis == 0 else num_rows):
            # Extract a row or column based on the specified axis
            signal = image[:, index] if axis == 0 else image[index, :]
            
            # Find peaks in this 1D signal
            peaks, _ = find_peaks(signal, prominence=self.args.prominence, width=self.args.width)
            
            # Store refined peaks with their original coordinates
            for peak in peaks:
                if axis == 0:
                    refined_peaks.append((peak, index))  # For columns
                else:
                    refined_peaks.append((index, peak))  # For rows
                    
        return refined_peaks
    
    def _apply(self, image=None):
        if image is None:
            image = self.loaded_image
        return image * self.mask
 
    def _check_corners(self):
        print("Corner values of the mask:")
        print(f"Top left: {self.mask[0, 0]}, Top right: {self.mask[0, -1]},\n Bottom left: {self.mask[-1, 0]}, Bottom right: {self.mask[-1, -1]}\n\n")

    def gaussian_mask(self):
        a, b = self.params
        image_shape = self.loaded_image.shape
        center_x, center_y = image_shape[1] // 2, image_shape[0] // 2
        x = np.linspace(0, image_shape[1] - 1, image_shape[1])
        y = np.linspace(0, image_shape[0] - 1, image_shape[0])
        x, y = np.meshgrid(x - center_x, y - center_y)
        sigma_x = image_shape[1] / (2.0 * a)
        sigma_y = image_shape[0] / (2.0 * b)
        gaussian = np.exp(-(x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)))
        gaussian /= gaussian.max()
        return gaussian

    def _display_mask(self):
        shape, mask = self.loaded_image.shape, self.mask 
        plt.imshow(mask, extent=(0, shape[1], 0, shape[0]), origin='lower', cmap='viridis')
        plt.colorbar()
        plt.title('Elliptical Gaussian Mask')
        plt.show()
    
    def _display_peaks_2d(self, img_threshold=0.005):
        image, peaks = self.loaded_image, self.peaks
        plt.figure(figsize=(10, 10))
        masked_image = np.ma.masked_less_equal(image, img_threshold) # mask values less than threshold (for loading speed)
        plt.imshow(masked_image, cmap='viridis')
        
        # filter peaks by threshold
        flt_peaks = [coord for coord in peaks if image[coord] > img_threshold]
        for x,y in flt_peaks: 
            plt.scatter(y, x, color='r', s=50, marker='x') 
            
        plt.title('Image with Detected Peaks')            
        plt.xlabel('X-axis (ss)')
        plt.ylabel('Y-axis (fs)')
        plt.show()
        
    def _display_peaks_3d(self, img_threshold=0.005):
        image, peaks = self.loaded_image, self.peaks
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # grid 
        x, y = np.arange(0, image.shape[1], 1), np.arange(0, image.shape[0], 1)
        X, Y = np.meshgrid(x, y)
        Z = np.ma.masked_less_equal(image, img_threshold)
        flt_peaks = [coord for coord in peaks if image[coord] > img_threshold]
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0, antialiased=False, alpha=0.6)
        p_x, p_y = zip(*flt_peaks)
        p_z = np.array([image[px, py] for px, py in flt_peaks])
        ax.scatter(p_y, p_x, p_z, color='r', s=50, marker='x', label='Peaks')
        
        # labels 
        ax.set_title('3D View of Image with Detected Peaks')
        ax.set_xlabel('X-axis (ss)')
        ax.set_ylabel('Y-axis (fs)')
        ax.set_zlabel('Intensity')
        fig.colorbar(surf, shrink=0.5, aspect=5, label='Intensity')
    
        plt.legend()
        plt.show()
    
    def _display_peak_finding(self):
        """
        Visualizes the peak finding process by displaying the original,
        denoised, Gaussian masked images, and the final image with refined peaks.
        """
        def normalize(image): # for visualization
            return (image - np.min(image)) / (np.max(image) - np.min(image))
        
        # Denoise the image and normalize
        denoised_image = median(self.loaded_image, disk(self.args.median_filter_size))
        norm_denoised_image = normalize(denoised_image)
        
        # Apply Gaussian mask and normalize
        gaussian_masked_image = self._apply(denoised_image)
        norm_gaussian_masked_image = normalize(gaussian_masked_image)
    
        # Perform peak detection and refinement
        refined_peaks = self._find_peaks()  # Ensure this returns 2D peak coordinates        
            
        fig, axs = plt.subplots(1, 4, figsize=(20, 5), subplot_kw={'xticks': [], 'yticks': []})
        
        axs[0].imshow(normalize(self.loaded_image), cmap='viridis')
        axs[0].set_title('Original Image')
        
        axs[1].imshow(norm_denoised_image, cmap='viridis')
        axs[1].set_title('Denoised Image')
        
        axs[2].imshow(norm_gaussian_masked_image, cmap='viridis')
        axs[2].set_title('Gaussian Masked Image')
        
        axs[3].imshow(norm_gaussian_masked_image, cmap='viridis')
        axs[3].scatter([y for x, y in refined_peaks], [x for x, y in refined_peaks], color='r', s=50, marker='x')
        axs[3].set_title('Refined Peaks')
        
        plt.tight_layout()
        plt.show()
                
    def _display_masked_image(self):
        masked_image = self._apply()
        plt.imshow(masked_image, cmap='viridis')
        plt.colorbar()
        plt.title('Masked Image')
        plt.show()
        
class PeakThresholdProcessor: 
    def __init__(self, image_array, threshold_value=0):
        self.image_array = image_array
        self.threshold_value = threshold_value
    
    def set_threshold_value(self, new_threshold_value):
        self.threshold_value = new_threshold_value
    
    def get_coordinates_above_threshold(self):  
        coordinates = np.argwhere(self.image_array > self.threshold_value)
        return coordinates
    
    def get_local_maxima(self):
        image_1d = self.image_array.flatten()
        peaks, _ = find_peaks(image_1d, height=self.threshold_value)
        coordinates = [self.flat_to_2d(idx) for idx in peaks]
        return coordinates
        
    def flat_to_2d(self, index):
        shape = self.image_array.shape
        rows, cols = shape
        return (index // cols, index % cols) 
    
class ArrayRegion:
    def __init__(self, array):
        self.array = array
        self.x_center = 0
        self.y_center = 0
        self.region_size = 0
    
    def set_peak_coordinate(self, x, y):
        self.x_center = x
        self.y_center = y
    
    def set_region_size(self, size):
        #limit that is printable in terminal
        self.region_size = size
        max_printable_region = min(self.array.shape[0], self.array.shape[1]) //2
        self.region_size = min(size, max_printable_region)
    
    def get_region(self):
        x_range = slice(self.x_center - self.region_size, self.x_center + self.region_size+1)
        y_range = slice(self.y_center - self.region_size, self.y_center + self.region_size+1)
        region = self.array[x_range, y_range]
        return region

    def extract_region(self, x_center, y_center, region_size):
            self.set_peak_coordinate(x_center, y_center)
            self.set_region_size(region_size)
            region = self.get_region()

            # Set print options for better readability
            np.set_printoptions(precision=8, suppress=True, linewidth=120, edgeitems=7)
            return region
    

