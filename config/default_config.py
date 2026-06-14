"""
Main configuration for CT distortion correction pipeline
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple
from enum import Enum


class InterpolationMethod(Enum):
    """Available distortion map interpolation methods"""
    RBF = "rbf"
    THIN_PLATE_SPLINE = "tps"
    KRIGING = "kriging"


@dataclass
class DetectionConfig:
    """Dot/sphere detection parameters"""
    
    # Sphere detection (3D morphological operations)
    sphere_diameter_voxels: int = 25  # ~3mm at 100-150µm voxels
    min_intensity_threshold: float = 500  # HU or normalized intensity
    morphological_radius: int = 12  # voxels, for binary morphology
    gaussian_sigma: float = 1.0  # voxels, for smoothing
    
    # Subvoxel refinement
    enable_subvoxel_refinement: bool = True
    subvoxel_method: str = "center_of_mass"  # or "gaussian_fit"
    
    # Quality control
    min_sphere_compactness: float = 0.85  # 1.0 = perfect sphere
    max_detection_error_voxels: float = 0.5  # reject if >0.5 voxel error


@dataclass
class MappingConfig:
    """Distortion map generation parameters"""
    
    # Interpolation
    method: InterpolationMethod = InterpolationMethod.THIN_PLATE_SPLINE
    regularization_strength: float = 1e-4  # TPS smoothing parameter
    rbf_kernel: str = "thin_plate"  # or "multiquadric", "gaussian", etc.
    rbf_epsilon: float = 1.0  # RBF shape parameter
    
    # Validation
    cross_validation_folds: int = 5
    leave_one_out: bool = False
    validation_metric: str = "rmse"  # root mean square error
    max_acceptable_residual_mm: float = 0.05  # 50 µm
    
    # Output resolution
    output_grid_resolution: Tuple[int, int, int] = (128, 128, 50)  # voxels


@dataclass
class UncertaintyConfig:
    """Uncertainty quantification parameters"""
    
    # Error propagation
    enable_monte_carlo: bool = True
    monte_carlo_samples: int = 1000
    confidence_level: float = 0.95  # 95% confidence interval
    
    # Noise model
    measurement_noise_mm: float = 0.0003  # ±0.3 µm from spec
    position_quantization_mm: float = 0.00015  # half-voxel, 150µm voxel size
    
    # Sensitivity analysis
    enable_sensitivity_analysis: bool = True
    parameter_variation_percent: float = 5.0  # ±5% parameter sweep


@dataclass
class CorrectionConfig:
    """Image correction/warping parameters"""
    
    # Interpolation for warping
    warp_interpolation_order: int = 3  # cubic spline
    extrapolation_mode: str = "constant"  # how to handle out-of-bounds
    extrapolation_value: float = -1000  # HU value for extrapolated regions
    
    # Performance
    batch_processing: bool = True
    batch_size_slices: int = 32
    parallel_workers: int = 4


@dataclass
class CalibrationConfig:
    """Overall calibration pipeline config"""
    
    # Plate specs
    plate_size_mm: Tuple[float, float] = (200.0, 200.0)
    grid_rows: int = 8
    grid_cols: int = 8
    sphere_diameter_mm: float = 3.0
    sphere_pitch_mm: float = 25.0
    sphere_distance_accuracy_mm: float = 0.3e-3  # ±0.3 µm
    
    # Voxel/geometry
    field_of_view_mm: float = 300.0
    voxel_size_mm: float = 0.12  # 120 µm (will be auto-detected from actual data)
    volume_shape: Tuple[int, int, int] = (2500, 2500, 400)  # approximate for 120MP, 300mm FOV
    
    # Sub-configs
    detection: DetectionConfig = field(default_factory=DetectionConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    uncertainty: UncertaintyConfig = field(default_factory=UncertaintyConfig)
    correction: CorrectionConfig = field(default_factory=CorrectionConfig)
    
    # Logging
    verbose: bool = True
    debug_visualizations: bool = False
    output_dir: str = "results/"


# Global default config instance
DEFAULT_CONFIG = CalibrationConfig()


def get_config() -> CalibrationConfig:
    """Get the current default configuration"""
    return DEFAULT_CONFIG


def update_config(updates: Dict[str, Any]) -> None:
    """
    Update configuration parameters at runtime.
    
    Example:
        update_config({
            "voxel_size_mm": 0.15,
            "detection.sphere_diameter_voxels": 20
        })
    """
    for key, value in updates.items():
        if "." in key:
            # Nested update (e.g., "detection.sphere_diameter_voxels")
            parts = key.split(".")
            obj = DEFAULT_CONFIG
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)
        else:
            setattr(DEFAULT_CONFIG, key, value)


if __name__ == "__main__":
    print("Default Configuration:")
    print(DEFAULT_CONFIG)
    print("\n✓ Configuration module ready")
