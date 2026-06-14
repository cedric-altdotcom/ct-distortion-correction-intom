import os
import logging
import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
from dot_detector_2d import DotDetector2D

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s'
)
logger = logging.getLogger("MetrologyTest")

def run_pipeline_test(image_path: str, output_dir: str = "test_results"):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"🚀 Starting Metrology Validation on: {image_path}")
    
    # 1. Initialize the detector instance
    detector = DotDetector2D()
    
    try:
        image = detector.load_image(image_path)
        
        # --- CRITICAL BOUNDING MATRIX FOR HIGH-RESOLUTION ANALYSIS ---
        detections = detector.detect_and_classify_dots(
            image,
            gaussian_sigma=1.5,
            min_area_pixels=15, 
            max_area_pixels=3000,
            regular_dot_area_range=(30, 250),   # Forced bounds for 500 µm dots
            code_dot_area_range=(300, 1500)     # Forced bounds for 800 µm dots
        )
        
        regular_dots_px = detections['regular_dots']
        code_dots_px = detections['code_dots']
        
        # Extract features using local classification blocks
        detected_anchors_px, block_ids, *decode_data = detector.identify_anchor_points(code_dots_px, regular_dots_px)
        design_anchors_mm = detector.design_positions['anchor_points']
        
        if len(detected_anchors_px) >= 4:
            from scipy.spatial import distance_matrix


            # --- TRANSFORMATION PARAMETERS ---
            known_mm_per_px = 0.05422  # Target scale parameter
            anchors_rough_mm = detected_anchors_px * known_mm_per_px
            
            # Base orientation from decoder metadata
            rotation_deg = 168.9
            if 'decode_data' in locals() and decode_data and isinstance(decode_data[0], list):
                try:
                    measured_angles = [d.get('rotation_deg', 168.9) for d in decode_data[0] if isinstance(d, dict)]
                    rotation_deg = float(np.median(measured_angles))
                except Exception:
                    pass

            logger.info("🔄 Running Final Global Optimization Grid...")

            best_pairs_count = -1
            best_valid_px = None
            best_design_matched = None
            best_config_name = ""

            # Exhaustive permutation space across flips, 180° shifts, and 90° axial swaps
            search_space = [
                ("Standard Phase A", 1.0, 1.0, 0.0, False),
                ("Standard Phase B", 1.0, 1.0, 180.0, False),
                ("Mirror Y Phase A", 1.0, -1.0, 0.0, False),
                ("Mirror Y Phase B", 1.0, -1.0, 180.0, False),
                ("Mirror X Phase A", -1.0, 1.0, 0.0, False),
                ("Mirror X Phase B", -1.0, 1.0, 180.0, False),
                ("Inverted Both A", -1.0, -1.0, 0.0, False),
                ("Inverted Both B", -1.0, -1.0, 180.0, False),
                # Axial Transpositions (if X and Y streams are physically swapped)
                ("Axial Swap Phase A", 1.0, 1.0, 0.0, True),
                ("Axial Swap Phase B", 1.0, 1.0, 180.0, True),
                ("Axial Swap Mirror Y", 1.0, -1.0, 0.0, True),
                ("Axial Swap Mirror X", -1.0, 1.0, 0.0, True),
            ]

            for name, flip_x, flip_y, phase_shift, swap_axes in search_space:
                pts_modified = anchors_rough_mm.copy()
                
                if swap_axes:
                    pts_modified = pts_modified[:, [1, 0]]
                    
                pts_modified[:, 0] *= flip_y
                pts_modified[:, 1] *= flip_x
                
                effective_angle = rotation_deg + phase_shift
                
                for direction in [1.0, -1.0]:
                    rotation_rad = np.radians(direction * effective_angle)
                    cos_r, sin_r = np.cos(rotation_rad), np.sin(rotation_rad)
                    R = np.array([[cos_r, -sin_r], [sin_r,  cos_r]])
                    
                    center_pixel = np.mean(pts_modified, axis=0)
                    anchors_rotated_mm = (pts_modified - center_pixel) @ R.T + center_pixel
                    
                    translation_offset = np.mean(design_anchors_mm, axis=0) - np.mean(anchors_rotated_mm, axis=0)
                    anchors_aligned_mm = anchors_rotated_mm + translation_offset
                    
                    # Compute full cost matrix [Detections x Design]
                    dists = distance_matrix(anchors_aligned_mm, design_anchors_mm)
                    
                    # Hungarian Algorithm: Finds the absolute best global pairing mathematically
                    # bypassing local step-distance constraints completely
                    row_ind, col_ind = opt.linear_sum_assignment(dists)
                    
                    # Pull matching errors for this configuration path
                    pairing_distances = dists[row_ind, col_ind]
                    
                    # Evaluate structural fit (count pairings aligned within a 4.0 mm tolerance window)
                    valid_fit_mask = pairing_distances < 4.0
                    valid_pairs_count = np.sum(valid_fit_mask)
                    
                    if valid_pairs_count > best_pairs_count:
                        best_pairs_count = valid_pairs_count
                        # Store matching indices
                        good_rows = row_ind[valid_fit_mask]
                        good_cols = col_ind[valid_fit_mask]
                        
                        best_valid_px = detected_anchors_px[good_rows]
                        best_design_matched = design_anchors_mm[good_cols]
                        best_config_name = f"{name} (Dir={direction})"

            # Commit globally optimized mapping selections
            valid_detected_px = best_valid_px
            design_anchors_matched = best_design_matched
            
            logger.info(f"🎯 Global Optimization finalized: [{best_config_name}]")
            logger.info(f"   ↳ Retained clean 1-to-1 CAD pairs: {len(valid_detected_px)} / {len(design_anchors_mm)}")

            # Compute Exact Homography Calibration Matrix using verified pairs
            H, error = detector.compute_homography_from_anchors(valid_detected_px, design_anchors_matched)
            
            # Match regular micro-dot fields across the projection grid
            design_matched, detected_mm, discrepancies_um = detector.match_regular_dots_to_design(
                regular_dots_px, detector.design_positions['regular_dots']
            )
            
            results = {
                'homography_matrix': H,
                'discrepancies_um': discrepancies_um,
                'design_regular_mm': design_matched,
                'um_per_pixel': detector.um_per_pixel
            }
        else:
            logger.error("❌ Test Failed: Insufficient anchor points detected (< 4).")
            results = {}
            
    except Exception as e:
        logger.error(f"❌ Pipeline crashed during image processing: {e}", exc_info=True)
        return

    if results.get('homography_matrix') is None:
        logger.error("❌ Test Failed: Homography calculation failed.")
        return
        
    discrepancies = results.get('discrepancies_um')
    if discrepancies is None or len(discrepancies) == 0:
        logger.error("❌ Test Failed: No regular dots matched the CAD specification.")
        return

    logger.info("✅ Core processing complete. Generating metrology analytics...")

    # 3. Compute structural error statistics
    dx = discrepancies[:, 1]      # X error components (microns)
    dy = discrepancies[:, 0]      # Y error components (microns)
    dr = np.sqrt(dx**2 + dy**2)   # Total spatial scalar offset metric

    design_mm = results['design_regular_mm']

    # 4. Plot 1: The Metrology Spatial Error Heatmap
    plt.figure(figsize=(11, 9))
    sc = plt.scatter(design_mm[:, 1], design_mm[:, 0], c=dr, cmap='jet', s=15, alpha=0.8)
    plt.colorbar(sc, label="Error Magnitude (µm)")
    plt.xlabel("Plate X Dimension (mm)")
    plt.ylabel("Plate Y Dimension (mm)")
    plt.title(f"Metrology Error Heatmap\nMean: {dr.mean():.2f}µm | Max: {dr.max():.2f}µm")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(output_dir, "metrology_heatmap.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 5. Plot 2: Vector Distortion Field
    plt.figure(figsize=(11, 9))
    quiver_scale = 10.0  # Increased visibility factor for micro-deviations
    plt.quiver(design_mm[:, 1], design_mm[:, 0], dx, -dy, dr, cmap='jet', 
               angles='xy', scale_units='xy', scale=quiver_scale)
    plt.xlabel("Plate X Dimension (mm)")
    plt.ylabel("Plate Y Dimension (mm)")
    plt.title(f"Deformation Vector Fields (Exaggerated scale: {quiver_scale}x)")
    plt.gca().invert_yaxis()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(output_dir, "metrology_quiver.png"), dpi=200, bbox_inches='tight')
    plt.close()

    # 6. Export Serialization Parameters
    stats = {
        "summary": {
            "total_matched_dots": int(len(dr)),
            "mean_error_um": float(dr.mean()),
            "median_error_um": float(np.median(dr)),
            "max_error_um": float(dr.max()),
            "std_dev_um": float(dr.std()),
            "estimated_resolution_um_per_px": float(results.get('um_per_pixel', 0))
        }
    }
    
    with open(os.path.join(output_dir, "test_metrics.json"), "w") as f:
        json.dump(stats, f, indent=4)

    logger.info(f"🎉 Test complete! Deliverables exported to folder: '{output_dir}/'")
    print(f"\n--- TEST REPORT ---")
    print(f"Matched Dots: {stats['summary']['total_matched_dots']}")
    print(f"Mean Spatial Shift: {stats['summary']['mean_error_um']:.3f} µm")
    print(f"Maximum Distortion: {stats['summary']['max_error_um']:.3f} µm")
    print(f"Resolution Scale:   {stats['summary']['estimated_resolution_um_per_px']:.3f} µm/pixel")

if __name__ == "__main__":
    target_ct_image = "Dotplate_500_1000.tif" 
    run_pipeline_test(target_ct_image)