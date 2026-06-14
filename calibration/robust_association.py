"""Robust feature-to-CAD association with strict 1-to-1 mapping and Hungarian algorithm"""
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance_matrix
import cv2

logger = logging.getLogger(__name__)


class RobustFeatureAssociation:
    """
    Implements strict 1-to-1 feature-to-CAD mapping using Hungarian algorithm.
    
    Prevents multiple image points from collapsing to the same CAD point,
    which is the root cause of corrupted homographies and infinite resolution errors.
    """
    
    def __init__(self, geometry):
        """
        Initialize feature association engine.
        
        Args:
            geometry: CodedDotPlateGeometry object
        """
        self.geometry = geometry
    
    def associate_detected_to_design_strict(self,
                                          detected_mm: np.ndarray,
                                          design_mm: np.ndarray,
                                          max_match_distance_mm: float = 0.5) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Perform strict 1-to-1 association using Hungarian algorithm.
        
        Ensures:
        - Every detected point maps to AT MOST one design point
        - Every design point is matched to AT MOST one detected point
        - Total assignment cost (distance) is globally minimized
        
        Args:
            detected_mm: Detected feature positions (M, 2) in mm
            design_mm: Design CAD positions (N, 2) in mm
            max_match_distance_mm: Maximum allowed matching distance (mm)
            
        Returns:
            (detected_matched_mm, design_matched_mm, association_quality_score)
            - detected_matched_mm: Matched detected points (K, 2)
            - design_matched_mm: Matched design points (K, 2)
            - association_quality_score: Fraction of detected points successfully matched [0-1]
        """
        logger.info(f"Performing strict 1-to-1 association: {len(detected_mm)} detected vs {len(design_mm)} design points")
        
        if len(detected_mm) == 0 or len(design_mm) == 0:
            logger.warning("No points to associate")
            return np.empty((0, 2)), np.empty((0, 2)), 0.0
        
        # Compute pairwise distance matrix
        dists_mm = distance_matrix(detected_mm, design_mm)
        
        # Convert to cost matrix (Hungarian minimizes cost)
        # Set costs for impossible matches to a high value
        cost_matrix = dists_mm.copy()
        cost_matrix[dists_mm > max_match_distance_mm] = 1e6  # Penalize distant matches
        
        # Apply Hungarian algorithm for optimal 1-to-1 assignment
        detected_indices, design_indices = linear_sum_assignment(cost_matrix)
        
        # Filter out invalid matches (distance > threshold)
        valid_mask = dists_mm[detected_indices, design_indices] <= max_match_distance_mm
        detected_indices = detected_indices[valid_mask]
        design_indices = design_indices[valid_mask]
        
        logger.info(f"Hungarian algorithm: {len(detected_indices)} valid 1-to-1 matches")
        
        # Extract matched points
        detected_matched = detected_mm[detected_indices]
        design_matched = design_mm[design_indices]
        
        # Quality score: fraction of detected points that found a match
        quality_score = len(detected_indices) / len(detected_mm) if len(detected_mm) > 0 else 0.0
        
        logger.info(f"Association quality: {quality_score*100:.1f}% ({len(detected_indices)}/{len(detected_mm)} matched)")
        
        return detected_matched, design_matched, quality_score
    
    def compute_homography_robust(self,
                                 detected_anchor_px: np.ndarray,
                                 design_anchor_mm: np.ndarray,
                                 detected_regular_px: np.ndarray,
                                 design_regular_mm: np.ndarray) -> Tuple[Optional[np.ndarray], float, float]:
        """
        Compute homography with validation and safe resolution extraction.
        
        Implements:
        1. Strict 1-to-1 anchor matching to establish ground truth
        2. Robust homography calculation with RANSAC
        3. Safe resolution computation with zero-division protection
        
        Args:
            detected_anchor_px: Detected L-anchor centers (K, 2) in pixels
            design_anchor_mm: Design anchor positions (K, 2) in mm
            detected_regular_px: All detected regular dots (N, 2) in pixels
            design_regular_mm: All design regular dots (M, 2) in mm
            
        Returns:
            (homography_matrix, um_per_pixel, quality_score)
            - homography_matrix: 3×3 homography matrix (or None if fails)
            - um_per_pixel: Estimated resolution in µm/pixel (or 0 if invalid)
            - quality_score: Confidence score (0-1)
        """
        logger.info("Computing robust homography with strict feature association...")
        
        if len(detected_anchor_px) < 4 or len(design_anchor_mm) < 4:
            logger.error(f"Insufficient anchors: {len(detected_anchor_px)} detected, {len(design_anchor_mm)} design (need ≥4)")
            return None, 0.0, 0.0
        
        # Step 1: Strict 1-to-1 anchor matching
        det_anchors_matched, design_anchors_matched, anchor_quality = self.associate_detected_to_design_strict(
            detected_anchor_px, design_anchor_mm, max_match_distance_mm=50.0
        )
        
        if len(det_anchors_matched) < 4:
            logger.error(f"Not enough valid anchor matches: {len(det_anchors_matched)} < 4")
            return None, 0.0, 0.0
        
        # Step 2: Compute homography with RANSAC
        try:
            H, status = cv2.findHomography(det_anchors_matched, design_anchors_matched, cv2.RANSAC, 10.0)
            
            if H is None:
                logger.error("Homography computation failed (returned None)")
                return None, 0.0, 0.0
            
            logger.info(f"Homography computed successfully. RANSAC inliers: {np.sum(status)}/{len(status)}")
            
            # Step 3: Safe resolution computation
            um_per_pixel = self._compute_resolution_safe(H, det_anchors_matched, design_anchors_matched)
            
            # Step 4: Quality assessment
            quality = self._assess_homography_quality(H, det_anchors_matched, design_anchors_matched)
            
            logger.info(f"Homography resolution: {um_per_pixel:.3f} µm/pixel, Quality: {quality:.3f}")
            
            return H, um_per_pixel, quality
            
        except Exception as e:
            logger.error(f"Homography computation exception: {e}")
            return None, 0.0, 0.0
    
    def _compute_resolution_safe(self, H: np.ndarray,
                                 detected_mm: np.ndarray,
                                 design_mm: np.ndarray) -> float:
        """
        Safely compute pixel-to-world resolution from homography.
        
        Avoids division by zero and validates the result.
        
        Args:
            H: 3×3 homography matrix (pixels → mm)
            detected_mm: Source points (N, 2) in mm
            design_mm: Target points (N, 2) in mm
            
        Returns:
            Resolution in µm/pixel, or 0.0 if invalid
        """
        try:
            # Inverse homography: mm → pixels
            H_inv = np.linalg.inv(H)
            
            # Compute scale by checking distance transformation
            # Method: take two reference points and compute pixel/mm ratio
            
            if len(design_mm) < 2:
                logger.warning("Insufficient points to compute resolution")
                return 0.0
            
            # Use first two distinct design points as reference
            ref_idx = [0, 1]
            
            # Physical distance between reference points (mm)
            phys_dist_mm = np.linalg.norm(design_mm[ref_idx[0]] - design_mm[ref_idx[1]])
            
            if phys_dist_mm < 1e-6:
                logger.warning("Reference points too close; cannot compute reliable resolution")
                return 0.0
            
            # Project reference points to pixel space using inverse homography
            ref_pixels_homog = H_inv @ np.array([
                [design_mm[ref_idx[0], 1], design_mm[ref_idx[0], 0], 1],  # [x_mm, y_mm, 1]
                [design_mm[ref_idx[1], 1], design_mm[ref_idx[1], 0], 1]
            ]).T
            
            # Normalize homogeneous coordinates
            ref_pixels = ref_pixels_homog[:2, :] / ref_pixels_homog[2, :]
            
            # Pixel distance
            pixel_dist = np.linalg.norm(ref_pixels[:, 0] - ref_pixels[:, 1])
            
            if pixel_dist < 1e-6:
                logger.warning("Pixel distance too small; cannot compute reliable resolution")
                return 0.0
            
            # Resolution: mm/pixel, then convert to µm/pixel
            mm_per_pixel = phys_dist_mm / pixel_dist
            um_per_pixel = mm_per_pixel * 1000.0
            
            # Sanity check: resolution should be positive and reasonable (1-1000 µm/pixel for typical CT)
            if um_per_pixel <= 0 or um_per_pixel > 10000:
                logger.warning(f"Unreasonable resolution computed: {um_per_pixel:.2f} µm/pixel")
                return 0.0
            
            logger.info(f"Resolution computed: {um_per_pixel:.3f} µm/pixel (mm/px: {mm_per_pixel:.6f})")
            return um_per_pixel
            
        except Exception as e:
            logger.error(f"Resolution computation failed: {e}")
            return 0.0
    
    def _assess_homography_quality(self, H: np.ndarray,
                                  detected_mm: np.ndarray,
                                  design_mm: np.ndarray) -> float:
        """
        Assess homography quality via reprojection error.
        
        Args:
            H: Homography matrix
            detected_mm: Source points (N, 2)
            design_mm: Target points (N, 2)
            
        Returns:
            Quality score (0-1), where 1 is perfect
        """
        try:
            # Project detected to mm space
            detected_homog = np.hstack([detected_mm, np.ones((len(detected_mm), 1))])
            projected = (H @ detected_homog.T).T
            projected = projected[:, :2] / projected[:, 2:3]
            
            # Compute reprojection error
            errors = np.linalg.norm(projected - design_mm, axis=1)
            rmse = np.sqrt(np.mean(errors**2))
            
            # Quality: inversely related to error
            # 1mm error → quality 0.5, 0.1mm error → quality 0.99
            quality = np.exp(-rmse / 1.0)
            quality = np.clip(quality, 0.0, 1.0)
            
            logger.info(f"Homography RMSE: {rmse:.4f} mm, Quality: {quality:.3f}")
            
            return quality
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return 0.0
    
    def match_regular_dots_strict(self,
                                 detected_regular_px: np.ndarray,
                                 design_regular_mm: np.ndarray,
                                 H: np.ndarray,
                                 um_per_pixel: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match regular dots to design grid using strict 1-to-1 association.
        
        Args:
            detected_regular_px: Detected regular dots (M, 2) in pixels
            design_regular_mm: Design regular dots (N, 2) in mm
            H: Homography matrix (pixels → mm)
            um_per_pixel: Pixel-to-µm conversion factor
            
        Returns:
            (design_matched_mm, detected_matched_mm, discrepancies_um)
        """
        logger.info(f"Strict matching of {len(detected_regular_px)} detected regular dots...")
        
        if len(detected_regular_px) == 0:
            logger.warning("No detected regular dots to match")
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
        
        # Project detected dots to mm space
        detected_homog = np.hstack([detected_regular_px, np.ones((len(detected_regular_px), 1))])
        detected_mm_proj = (H @ detected_homog.T).T
        detected_mm_proj = detected_mm_proj[:, :2] / detected_mm_proj[:, 2:3]
        
        logger.info(f"Projected {len(detected_mm_proj)} dots to mm space")
        
        # Adaptive distance threshold based on dot pitch (1mm nominal)
        # Allow up to 2× pitch distance for matching
        max_distance_mm = 2.0
        
        # Strict 1-to-1 matching
        detected_matched, design_matched, quality = self.associate_detected_to_design_strict(
            detected_mm_proj, design_regular_mm, max_match_distance_mm=max_distance_mm
        )
        
        if len(detected_matched) == 0:
            logger.warning("No successful dot matches")
            return np.empty((0, 2)), np.empty((0, 2)), np.empty((0, 2))
        
        # Compute discrepancies
        discrepancies_mm = detected_matched - design_matched
        discrepancies_um = discrepancies_mm * 1000.0
        
        logger.info(f"Successfully matched {len(detected_matched)} dots")
        logger.info(f"Mean discrepancy: dy={discrepancies_um[:, 0].mean():.2f}µm, dx={discrepancies_um[:, 1].mean():.2f}µm")
        logger.info(f"Max discrepancy: {np.max(np.abs(discrepancies_um)):.2f}µm")
        
        return design_matched, detected_matched, discrepancies_um


# Integration function for DotDetector2D
def integrate_robust_association_into_detector(dot_detector):
    """
    Monkey-patch DotDetector2D to use robust association.
    
    Args:
        dot_detector: DotDetector2D instance
    """
    association_engine = RobustFeatureAssociation(dot_detector.geometry)
    
    # Replace the homography computation method
    original_compute_homography = dot_detector.compute_homography_from_anchors
    
    def compute_homography_robust_wrapper(detected_anchors_px, design_anchors_mm):
        """Wrapper for robust homography computation"""
        H, um_per_pixel, quality = association_engine.compute_homography_robust(
            detected_anchors_px, design_anchors_mm,
            detected_anchors_px, design_anchors_mm  # Anchors only for initial calibration
        )
        
        if H is None:
            raise ValueError("Robust homography computation failed")
        
        dot_detector.homography_matrix = H
        dot_detector.um_per_pixel = um_per_pixel
        dot_detector.homography_quality = quality
        
        # Compute RMS error for compatibility
        try:
            detected_homog = np.hstack([detected_anchors_px, np.ones((len(detected_anchors_px), 1))])
            projected_mm = (H @ detected_homog.T).T
            projected_mm = projected_mm[:, :2] / projected_mm[:, 2:3]
            errors = np.linalg.norm(projected_mm - design_anchors_mm, axis=1)
            error_residual = np.sqrt(np.mean(errors**2))
        except:
            error_residual = 0.0
        
        return H, error_residual
    
    dot_detector.compute_homography_from_anchors = compute_homography_robust_wrapper
    dot_detector.association_engine = association_engine
    
    logger.info("Integrated robust feature association into DotDetector2D")


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("✓ Robust feature association module ready")
