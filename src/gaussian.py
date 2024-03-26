import os
import pkg

# path management
g = pkg.GaussianMask()
args = g.args()
params = (args.a, args.b)
print(f'Using mask parameters: a={params[0]}, b={params[1]}\n')

# load random h5 in images/
image, image_path, peaks = g.loaded_image, g.image_path, g.peaks 

# g.peaks uses _find_peaks() method

mask = g.gaussian_mask()
masked_image = g.apply()

threshold = 1000
a = pkg.ArrayRegion(image)
p = pkg.PeakThresholdProcessor(image, threshold)
refined_peaks = g.find_peaks()

# display methods  

v = pkg.Visualize(g)
# v._display_mask()
v.display_masked_image()
v.display_images() 
v.display_peak_regions()
v.display_peaks_2d() # very good 
v.display_peaks_3d()