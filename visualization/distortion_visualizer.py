"""Visualize distortion maps as images"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.interpolate import griddata
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DistortionVisualizer:
    """Visualize distortion vectors and maps"""
    
    def __init__(self, output_dir: str = "results/"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_distortion_combined(self, expected_pos: np.ndarray, 
                                 distortion_vectors: np.ndarray,
                                 detector_fov_mm: float = 300.0) -> None:
        """
        Create combined X and Y distortion maps in one image.
        
        Args:
            expected_pos: Expected sphere positions (N, 2) in mm [y, x]
            distortion_vectors: Displacement vectors (N, 2) in mm [dy, dx]
            detector_fov_mm: Detector field of view in mm
        """
        logger.info("Creating combined X/Y distortion map...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Grid for interpolation
        grid_size = 100
        y_range = np.linspace(-detector_fov_mm/2, detector_fov_mm/2, grid_size)
        x_range = np.linspace(-detector_fov_mm/2, detector_fov_mm/2, grid_size)
        yy, xx = np.meshgrid(y_range, x_range)
        
        # Y displacement map
        dy_interp = griddata(expected_pos, distortion_vectors[:, 0], 
                            (yy, xx), method='linear', fill_value=0)
        im0 = axes[0].contourf(xx, yy, dy_interp, levels=20, cmap='RdBu_r')
        axes[0].scatter(expected_pos[:, 1], expected_pos[:, 0], c='black', s=50, marker='+')
        axes[0].set_xlabel('X (mm)')
        axes[0].set_ylabel('Y (mm)')
        axes[0].set_title('Y Distortion (mm)')
        axes[0].set_aspect('equal')
        plt.colorbar(im0, ax=axes[0])
        
        # X displacement map
        dx_interp = griddata(expected_pos, distortion_vectors[:, 1], 
                            (yy, xx), method='linear', fill_value=0)
        im1 = axes[1].contourf(xx, yy, dx_interp, levels=20, cmap='RdBu_r')
        axes[1].scatter(expected_pos[:, 1], expected_pos[:, 0], c='black', s=50, marker='+')
        axes[1].set_xlabel('X (mm)')
        axes[1].set_ylabel('Y (mm)')
        axes[1].set_title('X Distortion (mm)')
        axes[1].set_aspect('equal')
        plt.colorbar(im1, ax=axes[1])
        
        plt.tight_layout()
        output_path = self.output_dir / "distortion_xy_map.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved combined X/Y map to {output_path}")
        plt.close()
    
    def plot_vector_field(self, expected_pos: np.ndarray, 
                         distortion_vectors: np.ndarray,
                         detector_fov_mm: float = 300.0) -> None:
        """
        Create vector field visualization.
        
        Args:
            expected_pos: Expected sphere positions (N, 2) in mm [y, x]
            distortion_vectors: Displacement vectors (N, 2) in mm [dy, dx]
            detector_fov_mm: Detector field of view in mm
        """
        logger.info("Creating vector field visualization...")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot vectors
        scale = 50  # Scale for visibility
        ax.quiver(expected_pos[:, 1], expected_pos[:, 0], 
                 distortion_vectors[:, 1], distortion_vectors[:, 0],
                 distortion_vectors[:, 0]**2 + distortion_vectors[:, 1]**2,
                 cmap='hot', scale=scale, width=0.003)
        
        # Plot sphere positions
        ax.scatter(expected_pos[:, 1], expected_pos[:, 0], c='blue', s=100, marker='o', alpha=0.6)
        
        ax.set_xlim(-detector_fov_mm/2, detector_fov_mm/2)
        ax.set_ylim(-detector_fov_mm/2, detector_fov_mm/2)
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_title('Distortion Vector Field')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / "distortion_vectors.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved vector field to {output_path}")
        plt.close()
