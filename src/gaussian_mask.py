import os
import glob 
import argparse
import h5py as h5
import numpy as np 
import matplotlib.pyplot as plt
import label_finder 

def load_data(choice):
    if choice: # whether at work or not
        file_path = 'images/DATASET1-1.h5'
    elif choice == False:
        water_background_dir = '/Users/adamkurth/Documents/vscode/CXFEL_Image_Analysis/CXFEL/waterbackground_subtraction/images/'
        file_path = os.path.join(water_background_dir,'9_18_23_high_intensity_3e8keV-2.h5')
        
    matching_files = glob.glob(file_path)
    if not matching_files:
        raise FileNotFoundError(f"No files found matching pattern: \n{file_path}")
        
    try:
        with h5.File(file_path, 'r') as f:
            data = f['entry/data/data'][:]
        return data, file_path
    except Exception as e:
        raise OSError(f"Failed to read {file_path}: {e}")
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Apply a Gaussian mask to an HDF5 image')
    parser.add_argument('--a', type=float, default=1.0, help='Gaussian scale factor for x-axis')
    parser.add_argument('--b', type=float, default=0.8, help='Gaussian scale factor for y-axis')
    return parser.parse_args()
    
def gaussian_filter(x, y, sigma_x, sigma_y):
    """
    Calculate a 2D Gaussian filter.

    Parameters:
        x (numpy.ndarray): X-axis values.
        y (numpy.ndarray): Y-axis values.
        sigma_x (float): Standard deviation along the X-axis.
        sigma_y (float): Standard deviation along the Y-axis.

    Returns:
        numpy.ndarray: 2D Gaussian filter.
    """
    gaussian = np.exp(-(x ** 2 / (2 * sigma_x ** 2) + y ** 2 / (2 * sigma_y ** 2)))    
    gaussian /= gaussian.max()
    return gaussian
    
def gaussian_mask(image_data, params):
    a, b = params
    image_shape = image_data.shape
    center_x, center_y = image_shape[1] / 2, image_shape[0] / 2
    
    # create a grid of x and y values 
    x = np.linspace(0, image_shape[1] - 1, image_shape[1])
    y = np.linspace(0, image_shape[0] - 1, image_shape[0])
    x, y = np.meshgrid(x - center_x, y - center_y)
    
    # standard deviation    
    sigma_x = image_shape[1] / (2.0 * a)
    sigma_y = image_shape[0] / (2.0 * b)

    gaussian_mask = gaussian_filter(x, y, sigma_x, sigma_y)    
    return gaussian_mask
    
def display_data(data, img_threshold=0.004):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Create x, y indices
    x, y = np.indices(data.shape[:2])
    # Flatten the 2D array into 1D for z-values
    z = data.ravel()

    # apply threshold
    mask = z > img_threshold

    # Filter x, y, z based on the threshold
    x, y, z = x.ravel()[mask], y.ravel()[mask], z[mask]

    scatter = ax.scatter(x, y, z, c=z, cmap='viridis', marker='o')
    ax.set_title('3D Scatter Plot of Image Data (Values > 0.004)')
    plt.colorbar(scatter)
    plt.show()

def display_scaled(data, img_threshold=0.0000001):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Scale the data
    scaled_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    x, y = np.indices(scaled_data.shape[:2])
    z = scaled_data.ravel()  # Flatten the 2D array into 1D for z-values

    # Apply the threshold on the scaled data
    mask = z > img_threshold

    # Filter x, y, z based on the threshold
    x, y, z = x.ravel()[mask], y.ravel()[mask], z[mask]

    scatter = ax.scatter(x, y, z, c=z, cmap='gray', marker='o')
    ax.set_title('3D Scatter Plot of Scaled Image Data (Values > 0.004)')
    plt.colorbar(scatter)
    plt.show()

def display_peaks(data, peaks, img_threshold=0.004):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Prepare data for plotting
    x, y = np.indices(data.shape[:2])
    z = data.ravel()  # Flatten the 2D array into 1D for z-values

    # Apply the threshold
    mask = z > img_threshold

    # Filter x, y, z based on the threshold
    x, y, z = x.ravel()[mask], y.ravel()[mask], z[mask]

    # Plot all data points
    ax.scatter(x, y, z, c='blue', marker='o', alpha=0.5, label='Data')

    # Highlight peaks
    if peaks:
        peak_x, peak_y = zip(*peaks)
        peak_z = data[peak_x, peak_y]
        ax.scatter(peak_x, peak_y, peak_z, c='red', marker='^', label='Peaks')

    ax.set_title('3D Scatter Plot of Image Data with Peaks Highlighted')
    ax.legend()
    plt.show()

def display_mask(image_data, mask):
    shape = image_data.shape
    plt.imshow(mask, extent=(0, shape[1], 0, shape[0]), origin='lower', cmap='viridis')
    plt.colorbar()
    plt.title('Elliptical Gaussian Mask')
    plt.show()

def display_mask_3d(image_data, mask, peaks, params=(1.0, 0.8), img_threshold=0.0004, visualize='both'):
    a, b = params
    shape = image_data.shape
    
    # Create a grid for coordinates
    x = np.linspace(0, shape[1], shape[1])
    y = np.linspace(0, shape[0], shape[0])
    x, y = np.meshgrid(x, y)
    
    # Flatten the arrays for scatter plot coordinates
    x_data = x.ravel()
    y_data = y.ravel()
    z_data = image_data.ravel()  # Image data values as height for scatter plot
    
    # Apply the threshold to filter points
    mask = z_data > img_threshold
    x_filtered = x_data[mask]
    y_filtered = y_data[mask]
    z_filtered = z_data[mask]
    
    # Calculate the Gaussian filter values
    sigma_x = shape[1] / (2 * a)
    sigma_y = shape[0] / (2 * b)
    gaussian = gaussian_filter(x - shape[1] / 2, y - shape[0] / 2, sigma_x, sigma_y)
    
    # Create a figure and 3D axes
    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(111, projection='3d')
    
    if visualize == 'data' or visualize == 'both':
        # Scatter plot of the filtered image data points
        data_scatter = ax.scatter(x_filtered, y_filtered, z_filtered, c='blue', marker='o', alpha=0.7, label='Filtered Data')
    
    if visualize == 'filter' or visualize == 'both':
        # Plot the Gaussian filter as a heatmap
        gaussian_surf = ax.plot_surface(x, y, gaussian, cmap='viridis', linewidth=0, antialiased=True, alpha=0.6)
    
    # Highlight peaks if they are present
    if peaks:
        peak_x, peak_y = zip(*peaks)
        peak_z = np.array([image_data[px, py] for px, py in peaks])
        peaks_scatter = ax.scatter(peak_x, peak_y, peak_z, c='red', marker='^', label='Peaks')

    # Set labels and title
    ax.set_xlabel('X axis')
    ax.set_ylabel('Y axis')
    ax.set_zlabel('Intensity')
    ax.set_title('3D Visualization of Data and Gaussian Filter with Peaks Highlighted')

    # Add a color bar for the Gaussian heatmap
    mappable = plt.cm.ScalarMappable(cmap='viridis')
    mappable.set_array(gaussian)
    cbar = plt.colorbar(mappable, shrink=0.5, aspect=5, ax=ax)
    cbar.set_label('Gaussian Filter Intensity')

    ax.legend()

    plt.show()
    
def check_corners(mask):
    print("Corner values of the mask:")
    print(f"Top left: {mask[0, 0]}, Top right: {mask[0, -1]}, Bottom left: {mask[-1, 0]}, Bottom right: {mask[-1, -1]}")

def apply_mask(image_data, mask, params=(1.0, 0.8)):
    a, b = params
    gaussian_mask(image_data, params=(a, b))
    masked_image = image_data * mask 
    return masked_image

def main():
    args = parse_arguments()
    a = args.a
    b = args.b
    params = (a, b)
    print(f'Using mask parameters: a={a}, b={b}\n')
    
    # loading and finding peaks above threshold
    image_data, file_path = label_finder.load_data(False) # at work/home boolean
    
    threshold = 1000
    coordinates = label_finder.main(file_path, threshold, display=False)
    coordinates = [tuple(coord) for coord in coordinates] 
    # print('\n', f'manually found coordinates {coordinates}\n')

    threshold_processor = label_finder.PeakThresholdProcessor(image_data, threshold)
    peaks = threshold_processor.get_local_maxima()

    # display methods 
    # display_data(image_data)
    # display_scaled(image_data)
    # display_peaks(image_data, peaks)
    
    mask = gaussian_mask(image_data, params)

    # display methods for mask
    # display_mask_3d(image_data, mask, peaks, (a,b), img_threshold=0.0004, gaussian_filter=gaussian_filter)
    # display_mask(image_data, mask)
    # display_mask_3d(image_data, mask, peaks, (a,b), visualize='both')
    # display_mask_3d(image_data, mask, peaks, (a,b), visualize='filter')

    # check_corners(mask)
    masked_image = apply_mask(image_data, mask, params)

    
if __name__ == "__main__":
    main()