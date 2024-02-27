import os
import pkg

# path management
g = pkg.GaussianMask()
args = g._args()
params = (args.a, args.b)
print(f'Using mask parameters: a={params[0]}, b={params[1]}\n')

# load random h5 in images/
image, image_path, peaks = g.loaded_image, g.image_path, g.peaks

g._display_mask()
mask = g.gaussian_mask()
masked_image = g._apply()

threshold = 1000
a = pkg.ArrayRegion(image)
p = pkg.PeakThresholdProcessor(image, threshold)
coords = p.get_local_maxima()
# g._display_peaks_2d()
g._display_peaks_3d()


