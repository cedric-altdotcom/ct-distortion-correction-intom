"""Complete calibration workflow: scan -> detect -> extract distortion -> visualize"""
import numpy as np
import logging
from pathlib import Path
import yaml

from calibration.sphere_detector import SphereDetector
from calibration.distortion_extractor import DistortionExtractor
from visualization.distortion_visualizer import DistortionVisualizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_calibration_specs() -> dict:
    """Load calibration specs from YAML"""
    spec_path = Path(__file__).parent.parent / "config" / "calibration_specs.yaml"
    with open(spec_path, 'r') as f:
        specs = yaml.safe_load(f)
    return specs


def create_synthetic_calibration_volume(specs: dict, size: Tuple[int, int, int] = (400, 500, 500)) -> np.ndarray:
    """
    Create synthetic CT volume with spheres at known positions.
    
    Args:
        specs: Calibration specifications
        size: Volume shape (depth, height, width) in voxels
        
    Returns:
        3D volume with synthetic spheres
    """
    logger.info(f"Generating synthetic calibration volume {size}...")
    
    volume = np.zeros(size, dtype=np.float32)
    
    rows = specs['grid']['rows']
    cols = specs['grid']['cols']
    pitch = specs['grid']['pitch_mm']
    sphere_diameter = specs['spheres']['diameter_mm']
    voxel_size = specs['scanner']['voxel_size_mm']
    fov = specs['scanner']['field_of_view_mm']
    
    # Center of volume
    center_z, center_y, center_x = np.array(size) / 2
    
    # Expected positions in mm
    grid_height = (rows - 1) * pitch
    grid_width = (cols - 1) * pitch
    start_y = -grid_height / 2
    start_x = -grid_width / 2
    
    sphere_radius_voxels = (sphere_diameter / voxel_size) / 2
    
    for i in range(rows):
        for j in range(cols):
            y_mm = start_y + i * pitch
            x_mm = start_x + j * pitch
            
            # Convert to voxels
<<<<<<< HEAD
            # Convert to voxels
            y_vox_ideal = center_y + y_mm / voxel_size
            x_vox_ideal = center_x + x_mm / voxel_size

            # Inject subtle radial pincushion distortion
            r2 = (y_mm**2 + x_mm**2) / 10000.0
            y_vox = center_y + (y_mm / voxel_size) * (1.0 + 0.08 * r2)
            x_vox = center_x + (x_mm / voxel_size) * (1.0 + 0.08 * r2)
            
            z_vox = center_z
            
            # Draw sphere (simple: fill within radius)
            z_min = max(0, int(np.floor(z_vox - sphere_radius_voxels)))
            z_max = min(size[0], int(np.ceil(z_vox + sphere_radius_voxels)))
            y_min = max(0, int(np.floor(y_vox - sphere_radius_voxels)))
            y_max = min(size[1], int(np.ceil(y_vox + sphere_radius_voxels)))
            x_min = max(0, int(np.floor(x_vox - sphere_radius_voxels)))
            x_max = min(size[2], int(np.ceil(x_vox + sphere_radius_voxels)))
=======
            y_vox = center_y + y_mm / voxel_size
            x_vox = center_x + x_mm / voxel_size
            z_vox = center_z
            
            # Draw sphere (simple: fill within radius)
            z_min = max(0, int(z_vox - sphere_radius_voxels))
            z_max = min(size[0], int(z_vox + sphere_radius_voxels))
            y_min = max(0, int(y_vox - sphere_radius_voxels))
            y_max = min(size[1], int(y_vox + sphere_radius_voxels))
            x_min = max(0, int(x_vox - sphere_radius_voxels))
            x_max = min(size[2], int(x_vox + sphere_radius_voxels))
>>>>>>> a64f9f97cb8a03f6bd3bddf246cecf7789dd5ffc
            
            for z in range(z_min, z_max):
                for y in range(y_min, y_max):
                    for x in range(x_min, x_max):
                        dist = np.sqrt((z - z_vox)**2 + (y - y_vox)**2 + (x - x_vox)**2)
                        if dist <= sphere_radius_voxels:
                            volume[z, y, x] = max(volume[z, y, x], 1000 - 100*dist/sphere_radius_voxels)
    
    logger.info(f"Generated volume with {rows*cols} spheres")
    return volume


def main():
    """Run full calibration workflow"""
    logger.info("=" * 60)
    logger.info("CT CALIBRATION WORKFLOW")
    logger.info("=" * 60)
    
    # Load specs
    specs = load_calibration_specs()
    logger.info(f"Loaded specs: {specs['grid']['rows']}x{specs['grid']['cols']} grid")
    
    # Create synthetic volume
<<<<<<< HEAD
    volume = create_synthetic_calibration_volume(specs, size=(400, 500, 500))
    
    # --- RESTORED DETECTION LINES START HERE ---
=======
    volume = create_synthetic_calibration_volume(specs)
    
>>>>>>> a64f9f97cb8a03f6bd3bddf246cecf7789dd5ffc
    # Detect spheres
    detector = SphereDetector()
    centers_raw = detector.detect_spheres(volume)
    centers_refined = detector.refine_centers(volume, centers_raw)
    logger.info(f"Detected {len(centers_refined)} spheres")
<<<<<<< HEAD
    # --- RESTORED DETECTION LINES END HERE ---
=======
>>>>>>> a64f9f97cb8a03f6bd3bddf246cecf7789dd5ffc
    
    # Extract distortion
    extractor = DistortionExtractor(specs)
    expected, detected, distortion = extractor.extract_distortion(
        centers_refined, 
        specs['scanner']['voxel_size_mm']
    )
    
    # Visualize
    visualizer = DistortionVisualizer(output_dir="results/")
    visualizer.plot_distortion_combined(
        expected, 
        distortion,
        specs['scanner']['field_of_view_mm']
    )
    visualizer.plot_vector_field(
        expected,
        distortion,
        specs['scanner']['field_of_view_mm']
    )
    
    logger.info("=" * 60)
    logger.info("Workflow complete! Check results/ directory")
    logger.info("=" * 60)


if __name__ == "__main__":
    from typing import Tuple
    main()
