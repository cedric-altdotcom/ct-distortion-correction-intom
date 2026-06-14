"""Metrology reporting engine for coded calibration plate analysis"""
import numpy as np
import json
import logging
from typing import Dict, Tuple, Optional
from datetime import datetime
import os

logger = logging.getLogger(__name__)


class MetrologyReportGenerator:
    """Generate high-precision metrology reports from calibration plate analysis"""
    
    def __init__(self, geometry):
        """
        Initialize report generator.
        
        Args:
            geometry: CodedDotPlateGeometry object
        """
        self.geometry = geometry
        self.report_data = None
    
    def generate_metrology_report(self, 
                                 design_regular_mm: np.ndarray,
                                 detected_regular_mm: np.ndarray,
                                 discrepancies_um: np.ndarray,
                                 block_ids: Optional[np.ndarray] = None,
                                 matched_design_indices: Optional[np.ndarray] = None,
                                 output_dir: Optional[str] = None,
                                 image_filename: str = "calibration_plate.tif") -> Dict:
        """
        Generate comprehensive high-precision metrology report.
        
        Args:
            design_regular_mm: Design positions (N, 2) in mm
            detected_regular_mm: Detected positions (N, 2) in mm
            discrepancies_um: Error vectors (N, 2) in µm
            block_ids: Optional block assignment for each dot (N,)
            matched_design_indices: Optional indices mapping detected to design dots
            output_dir: Output directory for reports
            image_filename: Name of source image file
            
        Returns:
            Dictionary with complete metrology data
        """
        logger.info("Generating high-precision metrology report...")
        
        # Compute per-dot precision error vectors
        report_data = self._compute_precision_errors(
            design_regular_mm, detected_regular_mm, discrepancies_um
        )
        
        # Assign block IDs if not provided
        if block_ids is None:
            block_ids = self._assign_block_ids_from_positions(design_regular_mm)
        
        report_data['block_ids'] = block_ids
        report_data['matched_design_indices'] = matched_design_indices
        
        # Compute global statistics
        report_data['global_stats'] = self._compute_global_statistics(
            report_data['euclidean_magnitude_um']
        )
        
        # Compute per-block statistics
        report_data['per_block_stats'] = self._compute_per_block_statistics(
            block_ids, discrepancies_um, report_data['euclidean_magnitude_um']
        )
        
        # Add metadata
        report_data['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'image_filename': image_filename,
            'total_dots_analyzed': len(discrepancies_um),
            'plate_dimensions_mm': (self.geometry.plate_width_mm, self.geometry.plate_height_mm),
            'macro_grid': (self.geometry.macro_rows, self.geometry.macro_cols),
            'regular_dot_diameter_um': self.geometry.regular_dot_diameter_um,
            'code_dot_diameter_um': self.geometry.code_dot_diameter_um,
            'dot_pitch_um': self.geometry.dot_pitch_um
        }
        
        self.report_data = report_data
        
        # Generate visualizations and exports
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate plots
            self._generate_heatmap(report_data, design_regular_mm, 
                                  os.path.join(output_dir, 'metrology_heatmap.png'))
            self._generate_quiver_plot(report_data, design_regular_mm, 
                                      os.path.join(output_dir, 'error_vector_plot.png'))
            self._generate_block_heatmap(report_data, 
                                        os.path.join(output_dir, 'block_error_heatmap.png'))
            
            # Export JSON report
            json_path = os.path.join(output_dir, 'metrology_report.json')
            self._export_json_report(report_data, json_path)
            
            # Export CSV for per-dot analysis
            csv_path = os.path.join(output_dir, 'dot_errors.csv')
            self._export_csv_report(report_data, design_regular_mm, csv_path)
            
            logger.info(f"Metrology report generated in {output_dir}")
        
        return report_data
    
    def _compute_precision_errors(self, design_mm: np.ndarray, 
                                  detected_mm: np.ndarray,
                                  discrepancies_um: np.ndarray) -> Dict:
        """
        Compute per-dot precision error components.
        
        Args:
            design_mm: Design positions (N, 2) [y, x] in mm
            detected_mm: Detected positions (N, 2) [y, x] in mm
            discrepancies_um: Error vectors (N, 2) [dy, dx] in µm
            
        Returns:
            Dictionary with error components
        """
        logger.info("Computing per-dot precision error vectors...")
        
        # Error components
        dy_um = discrepancies_um[:, 0]  # Y-component
        dx_um = discrepancies_um[:, 1]  # X-component
        
        # Euclidean magnitude
        dr_um = np.sqrt(dy_um**2 + dx_um**2)
        
        # Polar angle of error vectors
        error_angle_deg = np.arctan2(dy_um, dx_um) * 180.0 / np.pi
        
        return {
            'design_positions_mm': design_mm,
            'detected_positions_mm': detected_mm,
            'dy_um': dy_um,
            'dx_um': dx_um,
            'euclidean_magnitude_um': dr_um,
            'error_angle_deg': error_angle_deg
        }
    
    def _compute_global_statistics(self, magnitudes_um: np.ndarray) -> Dict:
        """Compute global error statistics across entire plate"""
        return {
            'mean_um': float(np.mean(magnitudes_um)),
            'median_um': float(np.median(magnitudes_um)),
            'std_um': float(np.std(magnitudes_um)),
            'max_um': float(np.max(magnitudes_um)),
            'min_um': float(np.min(magnitudes_um)),
            'p95_um': float(np.percentile(magnitudes_um, 95)),
            'p99_um': float(np.percentile(magnitudes_um, 99)),
            'rmse_um': float(np.sqrt(np.mean(magnitudes_um**2)))
        }
    
    def _compute_per_block_statistics(self, block_ids: np.ndarray, 
                                     discrepancies_um: np.ndarray,
                                     magnitudes_um: np.ndarray) -> Dict:
        """Compute statistics grouped by block (0-79)"""
        per_block = {}
        
        for block_id in range(self.geometry.macro_rows * self.geometry.macro_cols):
            mask = block_ids == block_id
            if not np.any(mask):
                continue
            
            block_dy = discrepancies_um[mask, 0]
            block_dx = discrepancies_um[mask, 1]
            block_dr = magnitudes_um[mask]
            
            # Block position in macro-grid
            block_row = block_id // self.geometry.macro_cols
            block_col = block_id % self.geometry.macro_cols
            
            per_block[int(block_id)] = {
                'block_position': (int(block_row), int(block_col)),
                'num_dots': int(np.sum(mask)),
                'mean_error_um': float(np.mean(block_dr)),
                'median_error_um': float(np.median(block_dr)),
                'std_error_um': float(np.std(block_dr)),
                'max_error_um': float(np.max(block_dr)),
                'min_error_um': float(np.min(block_dr)),
                'mean_dy_um': float(np.mean(block_dy)),
                'mean_dx_um': float(np.mean(block_dx)),
                'std_dy_um': float(np.std(block_dy)),
                'std_dx_um': float(np.std(block_dx))
            }
        
        logger.info(f"Computed statistics for {len(per_block)} blocks")
        return per_block
    
    def _assign_block_ids_from_positions(self, design_mm: np.ndarray) -> np.ndarray:
        """
        Assign block IDs based on dot positions.
        
        Args:
            design_mm: Design positions (N, 2) [y, x] in mm
            
        Returns:
            Array of block IDs (N,)
        """
        block_ids = np.zeros(len(design_mm), dtype=int)
        
        for i, pos_mm in enumerate(design_mm):
            y_mm, x_mm = pos_mm
            
            # Determine block row and column
            block_row = int(y_mm / self.geometry.block_height_mm)
            block_col = int(x_mm / self.geometry.block_width_mm)
            
            # Clamp to valid range
            block_row = np.clip(block_row, 0, self.geometry.macro_rows - 1)
            block_col = np.clip(block_col, 0, self.geometry.macro_cols - 1)
            
            block_id = block_row * self.geometry.macro_cols + block_col
            block_ids[i] = block_id
        
        return block_ids
    
    def _generate_heatmap(self, report_data: Dict, design_mm: np.ndarray, 
                         output_path: str) -> None:
        """Generate 2D error magnitude heatmap on physical plate layout"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            logger.info("Generating metrology heatmap...")
            
            fig, ax = plt.subplots(figsize=(14, 12))
            
            # Plot background
            ax.set_xlim(0, self.geometry.plate_width_mm)
            ax.set_ylim(0, self.geometry.plate_height_mm)
            ax.set_aspect('equal')
            ax.invert_yaxis()  # Top-left origin
            
            # Scatter plot with error magnitude as color
            magnitudes = report_data['euclidean_magnitude_um']
            scatter = ax.scatter(design_mm[:, 1], design_mm[:, 0], 
                               c=magnitudes, cmap='jet', s=100, 
                               alpha=0.7, edgecolors='black', linewidth=0.5)
            
            # Add color bar
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label('Error Magnitude ($\mu$m)', rotation=270, labelpad=20)
            
            # Add block grid lines
            for i in range(self.geometry.macro_rows + 1):
                y = i * self.geometry.block_height_mm
                ax.axhline(y, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            
            for j in range(self.geometry.macro_cols + 1):
                x = j * self.geometry.block_width_mm
                ax.axvline(x, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            
            ax.set_xlabel('X Position (mm)', fontsize=12)
            ax.set_ylabel('Y Position (mm)', fontsize=12)
            ax.set_title('Calibration Plate Metrology Heatmap\n(Error Magnitude by Dot Position)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.2)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved heatmap to {output_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Heatmap generation failed: {e}")
    
    def _generate_quiver_plot(self, report_data: Dict, design_mm: np.ndarray,
                             output_path: str) -> None:
        """Generate error vector quiver plot showing systematic distortions"""
        try:
            import matplotlib.pyplot as plt
            
            logger.info("Generating error vector quiver plot...")
            
            fig, ax = plt.subplots(figsize=(14, 12))
            
            # Quiver plot: error vectors magnified for visibility
            dy_um = report_data['dy_um']
            dx_um = report_data['dx_um']
            
            # Scale vectors for visibility (magnify by 10x for visualization)
            scale_factor = 10.0
            dy_scaled_mm = dy_um / 1000.0 * scale_factor
            dx_scaled_mm = dx_um / 1000.0 * scale_factor
            
            # Color by magnitude
            magnitudes = report_data['euclidean_magnitude_um']
            
            quiver = ax.quiver(design_mm[:, 1], design_mm[:, 0], 
                              dx_scaled_mm, dy_scaled_mm,
                              magnitudes, cmap='viridis', 
                              angles='xy', scale_units='xy', scale=1.0,
                              width=0.3)
            
            # Add color bar
            cbar = plt.colorbar(quiver, ax=ax)
            cbar.set_label('Error Magnitude ($\mu$m)', rotation=270, labelpad=20)
            
            ax.set_xlim(0, self.geometry.plate_width_mm)
            ax.set_ylim(0, self.geometry.plate_height_mm)
            ax.set_aspect('equal')
            ax.invert_yaxis()
            
            # Add block grid
            for i in range(self.geometry.macro_rows + 1):
                y = i * self.geometry.block_height_mm
                ax.axhline(y, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            
            for j in range(self.geometry.macro_cols + 1):
                x = j * self.geometry.block_width_mm
                ax.axvline(x, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
            
            ax.set_xlabel('X Position (mm)', fontsize=12)
            ax.set_ylabel('Y Position (mm)', fontsize=12)
            ax.set_title(f'Systematic Distortion Analysis\n(Vectors magnified {scale_factor}× for visibility)', 
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.2)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved quiver plot to {output_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Quiver plot generation failed: {e}")
    
    def _generate_block_heatmap(self, report_data: Dict, 
                               output_path: str) -> None:
        """Generate heatmap of per-block average errors"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as patches
            
            logger.info("Generating per-block error heatmap...")
            
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Create 2D grid of block errors
            block_errors = np.zeros((self.geometry.macro_rows, self.geometry.macro_cols))
            
            for block_id, stats in report_data['per_block_stats'].items():
                block_row = stats['block_position'][0]
                block_col = stats['block_position'][1]
                block_errors[block_row, block_col] = stats['mean_error_um']
            
            # Plot heatmap
            im = ax.imshow(block_errors, cmap='RdYlGn_r', aspect='auto', origin='upper')
            
            # Add text annotations
            for i in range(self.geometry.macro_rows):
                for j in range(self.geometry.macro_cols):
                    block_id = i * self.geometry.macro_cols + j
                    if block_id in report_data['per_block_stats']:
                        error = block_errors[i, j]
                        text = ax.text(j, i, f'{error:.1f}',
                                     ha="center", va="center", color="black", fontsize=9)
            
            # Color bar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Mean Error per Block ($\mu$m)', rotation=270, labelpad=20)
            
            ax.set_xticks(np.arange(self.geometry.macro_cols))
            ax.set_yticks(np.arange(self.geometry.macro_rows))
            ax.set_xticklabels(np.arange(self.geometry.macro_cols))
            ax.set_yticklabels(np.arange(self.geometry.macro_rows))
            
            ax.set_xlabel('Block Column', fontsize=12)
            ax.set_ylabel('Block Row', fontsize=12)
            ax.set_title('Mean Error by Block (0-79)\n(Macro-grid error distribution)', 
                        fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved block heatmap to {output_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Block heatmap generation failed: {e}")
    
    def _export_json_report(self, report_data: Dict, output_path: str) -> None:
        """Export comprehensive JSON metrology report"""
        try:
            logger.info("Exporting JSON metrology report...")
            
            # Prepare JSON-serializable data
            json_data = {
                'metadata': report_data['metadata'],
                'global_statistics': report_data['global_stats'],
                'per_block_statistics': report_data['per_block_stats'],
                'summary': {
                    'total_dots_analyzed': report_data['metadata']['total_dots_analyzed'],
                    'mean_error_um': report_data['global_stats']['mean_um'],
                    'std_error_um': report_data['global_stats']['std_um'],
                    'max_error_um': report_data['global_stats']['max_um'],
                    'rmse_um': report_data['global_stats']['rmse_um'],
                    'p95_error_um': report_data['global_stats']['p95_um']
                }
            }
            
            with open(output_path, 'w') as f:
                json.dump(json_data, f, indent=2)
            
            logger.info(f"Saved JSON report to {output_path}")
        except Exception as e:
            logger.error(f"JSON export failed: {e}")
    
    def _export_csv_report(self, report_data: Dict, design_mm: np.ndarray,
                          output_path: str) -> None:
        """Export per-dot error analysis as CSV"""
        try:
            logger.info("Exporting CSV dot error report...")
            
            import csv
            
            dy_um = report_data['dy_um']
            dx_um = report_data['dx_um']
            dr_um = report_data['euclidean_magnitude_um']
            block_ids = report_data['block_ids']
            angles = report_data['error_angle_deg']
            
            with open(output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['dot_index', 'block_id', 'design_y_mm', 'design_x_mm', 
                               'error_dy_um', 'error_dx_um', 'error_magnitude_um', 
                               'error_angle_deg'])
                
                for i in range(len(dy_um)):
                    writer.writerow([
                        i,
                        int(block_ids[i]),
                        f'{design_mm[i, 0]:.4f}',
                        f'{design_mm[i, 1]:.4f}',
                        f'{dy_um[i]:.4f}',
                        f'{dx_um[i]:.4f}',
                        f'{dr_um[i]:.4f}',
                        f'{angles[i]:.2f}'
                    ])
            
            logger.info(f"Saved CSV report to {output_path}")
        except Exception as e:
            logger.error(f"CSV export failed: {e}")
    
    def get_summary_text(self) -> str:
        """Generate human-readable summary text"""
        if self.report_data is None:
            return "No report data available"
        
        stats = self.report_data['global_stats']
        meta = self.report_data['metadata']
        
        summary = f"""
================================================================================
                    METROLOGY REPORT SUMMARY
================================================================================
Generated: {meta['timestamp']}
Source Image: {meta['image_filename']}
Total Dots Analyzed: {meta['total_dots_analyzed']}
Plate Dimensions: {meta['plate_dimensions_mm'][0]} × {meta['plate_dimensions_mm'][1]} mm
Macro-Grid: {meta['macro_grid'][0]} rows × {meta['macro_grid'][1]} columns (80 blocks)

================================================================================
                    GLOBAL ERROR STATISTICS
================================================================================
Mean Error:        {stats['mean_um']:.3f} µm
Median Error:      {stats['median_um']:.3f} µm
Std Deviation:     {stats['std_um']:.3f} µm
Max Error:         {stats['max_um']:.3f} µm
Min Error:         {stats['min_um']:.3f} µm
95th Percentile:   {stats['p95_um']:.3f} µm
99th Percentile:   {stats['p99_um']:.3f} µm
RMSE:              {stats['rmse_um']:.3f} µm

================================================================================
                    PER-BLOCK ANALYSIS
================================================================================
Blocks with highest errors:
"""
        
        # Sort blocks by mean error
        block_stats = self.report_data['per_block_stats']
        sorted_blocks = sorted(block_stats.items(), 
                             key=lambda x: x[1]['mean_error_um'], 
                             reverse=True)
        
        for block_id, stats in sorted_blocks[:5]:
            block_row, block_col = stats['block_position']
            summary += f"  Block {block_id} (row {block_row}, col {block_col}): {stats['mean_error_um']:.2f} µm ± {stats['std_error_um']:.2f} µm ({stats['num_dots']} dots)\n"
        
        summary += "\n" + "=" * 80 + "\n"
        
        return summary
