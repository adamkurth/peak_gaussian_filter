import os

import pkg

# path management
g = pkg.GaussianMask()
g.__args__()
images_dir = g.images_dir

# load random h5 in images/
image, image_path = g.__load_h5__(g.images_dir)



