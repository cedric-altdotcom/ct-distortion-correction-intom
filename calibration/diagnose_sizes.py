import numpy as np
from scipy import ndimage
from skimage.measure import regionprops
from dot_detector_2d import DotDetector2D

detector = DotDetector2D()
img = detector.load_image("Dotplate_500_1000.tif")

# Replicate your thresholding pipeline
img_norm = (img.astype(float) - img.min()) / (img.max() - img.min() + 1e-8)
img_smooth = ndimage.gaussian_filter(img_norm, sigma=1.5)
binary = img_smooth < np.percentile(img_smooth, 25)
labeled, num_features = ndimage.label(binary)
regions = regionprops(labeled)

# Extract all area sizes and print the distribution
areas = sorted([r.area for r in regions if 10 < r.area < 50000])
print(f"Total objects tracked: {len(areas)}")
print(f"Smallest detected object area: {areas[0]} pixels")
print(f"10th percentile area: {np.percentile(areas, 10)} pixels")
print(f"50th percentile (Median background dot area): {np.percentile(areas, 50)} pixels")
print(f"90th percentile area: {np.percentile(areas, 90)} pixels")
print(f"Largest detected object area: {areas[-1]} pixels")