"""Sphere detection from CT volumes"""
import numpy as np
from scipy import ndimage
from config.default_config import get_config
import logging

logger = logging.getLogger(__name__)


class SphereDetector:
    """Detect spheres in CT volumes using morphological operations"""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self.detection_config = self.config.detection
    
    def detect_spheres(self, volume: np.ndarray) -> np.ndarray:
        """
        Detect sphere centers in 3D CT volume.
        
        Args:
            volume: 3D CT volume (shape: depth, height, width)
            
        Returns:
            Array of sphere center coordinates (N, 3) in voxels [z, y, x]
        """
        logger.info(f"Detecting spheres in volume shape {volume.shape}...")
        
        # Normalize volume
        vol_min, vol_max = volume.min(), volume.max()
        volume_norm = (volume - vol_min) / (vol_max - vol_min + 1e-8)
        
        # Threshold
        # Dynamic Midpoint Threshold
        binary = volume_norm > 0.01
        
        # Morphological operations
        radius = self.detection_config.morphological_radius
        struct = ndimage.generate_binary_structure(3, 2)
        binary = ndimage.binary_closing(binary, structure=struct, iterations=1)
        binary = ndimage.binary_opening(binary, structure=struct, iterations=1)
        
        # Label connected components
        labeled, num_features = ndimage.label(binary)
        logger.info(f"Found {num_features} connected components")
        
        # Extract centers of mass
        centers = ndimage.center_of_mass(binary, labeled, range(1, num_features + 1))
        centers = np.array(centers)
        
        logger.info(f"Detected {len(centers)} sphere candidates")
        return centers
    
    def refine_centers(self, volume: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Refine sphere centers using center-of-mass with local window.
        
        Args:
            volume: 3D CT volume
            centers: Initial sphere centers (N, 3)
            
        Returns:
            Refined centers (N, 3)
        """
        radius = self.detection_config.sphere_diameter_voxels // 2
        refined = []
        
        for center in centers:
            z, y, x = center.astype(int)
            z_min = max(0, z - radius)
            z_max = min(volume.shape[0], z + radius)
            y_min = max(0, y - radius)
            y_max = min(volume.shape[1], y + radius)
            x_min = max(0, x - radius)
            x_max = min(volume.shape[2], x + radius)
            
            patch = volume[z_min:z_max, y_min:y_max, x_min:x_max]
            local_center = ndimage.center_of_mass(patch)
            
            refined_center = np.array([
                z_min + local_center[0],
                y_min + local_center[1],
                x_min + local_center[2]
            ])
            refined.append(refined_center)
        
        return np.array(refined)
