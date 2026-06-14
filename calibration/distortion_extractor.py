"""Extract distortion vectors from calibration plate scan"""
import numpy as np
from typing import Tuple
import logging
from scipy.spatial import distance_matrix

logger = logging.getLogger(__name__)


class DistortionExtractor:
    """Extract distortion as displacement vectors from measured vs expected sphere positions"""
    
    def __init__(self, config):
        self.config = config
        self.calibration_config = config
    
    def generate_expected_grid(self) -> np.ndarray:
        """
        Generate expected sphere positions in grid coordinates (mm).
        Returns (N, 2) array of [y, x] positions in mm.
        """
        rows = self.calibration_config['grid']['rows']
        cols = self.calibration_config['grid']['cols']
        pitch = self.calibration_config['grid']['pitch_mm']
        
        # Center the grid
        grid_height = (rows - 1) * pitch
        grid_width = (cols - 1) * pitch
        start_y = -grid_height / 2
        start_x = -grid_width / 2
        
        positions = []
        for i in range(rows):
            for j in range(cols):
                y = start_y + i * pitch
                x = start_x + j * pitch
                positions.append([y, x])
        
        return np.array(positions)  # (N, 2)
    
    def extract_distortion(self, detected_centers_voxels: np.ndarray, 
                           voxel_size_mm: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate distortion vectors from detected vs expected positions.
        
        Args:
            detected_centers_voxels: Detected sphere centers (N, 3) in voxels [z, y, x]
            voxel_size_mm: Voxel size in mm
            
        Returns:
            expected_positions: Expected positions (N, 2) in mm [y, x]
            detected_positions: Detected positions (N, 2) in mm [y, x]
            distortion_vectors: Displacement vectors (N, 2) in mm [dy, dx]
        """
        logger.info("Extracting distortion vectors...")
        
        # Generate expected grid
        expected = self.generate_expected_grid()  # (N, 2) [y, x] in mm
        
        # Convert detected centers to mm (use y, x only, ignore z)
        # Re-center raw voxels around (0, 0) before converting to mm
        centered_voxels = detected_centers_voxels[:, 1:] - 250.0
        detected = centered_voxels * voxel_size_mm  # (N, 2) [y, x]
        
        logger.info("Matching detected spheres to closest expected grid points...")

        # Compute distance between every detected point and every expected point
        dists = distance_matrix(detected, expected)

        # Find the best expected match index for each detected sphere
        matched_expected_idx = np.argmin(dists, axis=1)

        # Reorder the expected grid to align perfectly with the detected order
        expected_matched = expected[matched_expected_idx]
        detected_matched = detected
        
        # Calculate distortion vectors
        distortion = detected_matched - expected_matched  # (N, 2) [dy, dx]
        
        logger.info(f"Matched {len(detected_matched)} spheres")
        logger.info(f"Mean distortion: dy={distortion[:, 0].mean():.3f}mm, dx={distortion[:, 1].mean():.3f}mm")
        logger.info(f"Std distortion: dy={distortion[:, 0].std():.3f}mm, dx={distortion[:, 1].std():.3f}mm")
        
        return expected_matched, detected_matched, distortion
