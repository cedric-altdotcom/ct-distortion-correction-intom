"""2D dot/circle detection for coded calibration plates (Dot-Plate 500-1000)

Handles macro-grid structured plates with:
- 8 rows × 10 columns of code blocks
- Regular background dots (500 µm, 1000 µm pitch)
- Larger orientation dots (800 µm) forming L-shaped markers
- Binary code encoding positional IDs (LSB to MSB)
"""
import numpy as np
import logging
from typing import Tuple, List, Dict, Optional
from scipy import ndimage
from scipy.ndimage import label as ndimage_label
from scipy.ndimage import center_of_mass
from skimage.measure import regionprops
from scipy.spatial import distance_matrix
import cv2

logger = logging.getLogger(__name__)


class CodedDotPlateGeometry:
    """Calibration plate CAD design geometry (Dot-Plate 500-1000)"""
    
    def __init__(self):
        # Physical dimensions (mm)
        self.plate_width_mm = 295.0
        self.plate_height_mm = 260.0
        
        # Macro-grid structure
        self.macro_rows = 8
        self.macro_cols = 10
        
        # Dot specifications
        self.regular_dot_diameter_um = 500
        self.code_dot_diameter_um = 800
        self.dot_pitch_um = 1000  # center-to-center
        
        # Per-block geometry
        self.dots_per_row_in_block = 10  # background grid within block
        self.dots_per_col_in_block = 8
        self.block_width_mm = self.plate_width_mm / self.macro_cols
        self.block_height_mm = self.plate_height_mm / self.macro_rows
        
        # L-shaped marker positions within each block (relative to block origin)
        # Corner dot at (col=1, row=1) forms the reference point (right angle)
        # LSB bits along horizontal from corner: (cols 2-7, row=1) - left-right sequence
        # MSB bits along vertical from corner: (col=1, rows 2-7) - top-bottom sequence
        self.corner_relative_pos = (1, 1)      # (col, row) - right angle anchor
        self.lsb_start_relative_pos = (2, 1)   # Start of horizontal LSB sequence
        self.msb_start_relative_pos = (1, 2)   # Start of vertical MSB sequence
        self.bits_per_axis = 6                 # 6 bits per axis = 7 positions per axis (including corner)
        
    
    def get_block_origin_mm(self, block_row: int, block_col: int) -> Tuple[float, float]:
        """
        Get the top-left corner of a block in mm.
        
        Args:
            block_row: Macro-grid row (0-7)
            block_col: Macro-grid column (0-9)
            
        Returns:
            (y_mm, x_mm) - top-left corner of block in plate coordinate system
        """
        y_mm = block_row * self.block_height_mm
        x_mm = block_col * self.block_width_mm
        return y_mm, x_mm
    
    def get_all_design_positions_mm(self) -> Dict:
        """
        Generate all design CAD positions for the plate.
        
        Returns:
            Dictionary with:
            {
                'regular_dots': (N, 2) array [y, x] in mm,
                'code_dots': (N, 2) array [y, x] in mm,
                'block_codes': (N,) array of block IDs (0-79),
                'anchor_points': (N, 2) array of L-anchor centers in mm,
                'block_geometry': {block_id: {'corner': pos, 'lsb_dots': [...], 'msb_dots': [...]}}
            }
        """
        regular_dots_list = []
        code_dots_list = []
        block_codes_list = []
        anchor_points_list = []
        block_geometry_dict = {}
        
        # Per-block spacing in mm
        block_dot_spacing_mm = self.dot_pitch_um / 1000.0  # 1.0 mm
        
        block_id = 0
        for block_row in range(self.macro_rows):
            for block_col in range(self.macro_cols):
                block_origin_y, block_origin_x = self.get_block_origin_mm(block_row, block_col)
                
                # Generate regular background dots within block
                for dot_row in range(self.dots_per_col_in_block):
                    for dot_col in range(self.dots_per_row_in_block):
                        y_mm = block_origin_y + dot_row * block_dot_spacing_mm
                        x_mm = block_origin_x + dot_col * block_dot_spacing_mm
                        regular_dots_list.append([y_mm, x_mm])
                
                # Generate L-shaped code marker dots
                # Corner dot (right-angle reference)
                corner_y = block_origin_y + self.corner_relative_pos[1] * block_dot_spacing_mm
                corner_x = block_origin_x + self.corner_relative_pos[0] * block_dot_spacing_mm
                code_dots_list.append([corner_y, corner_x])
                
                # LSB horizontal sequence (bit 0 to bit 5)
                lsb_dots = []
                for bit_idx in range(self.bits_per_axis):
                    lsb_y = block_origin_y + self.lsb_start_relative_pos[1] * block_dot_spacing_mm
                    lsb_x = block_origin_x + (self.lsb_start_relative_pos[0] + bit_idx) * block_dot_spacing_mm
                    lsb_dots.append([lsb_y, lsb_x])
                    code_dots_list.append([lsb_y, lsb_x])
                
                # MSB vertical sequence (bit 6 onwards, or full vertical encoding)
                msb_dots = []
                for bit_idx in range(self.bits_per_axis):
                    msb_y = block_origin_y + (self.msb_start_relative_pos[1] + bit_idx) * block_dot_spacing_mm
                    msb_x = block_origin_x + self.msb_start_relative_pos[0] * block_dot_spacing_mm
                    msb_dots.append([msb_y, msb_x])
                    code_dots_list.append([msb_y, msb_x])
                
                # Compute L-anchor center (corner + first lsb + first msb)
                anchor_y = (corner_y + lsb_dots[0][0] + msb_dots[0][0]) / 3.0
                anchor_x = (corner_x + lsb_dots[0][1] + msb_dots[0][1]) / 3.0
                anchor_points_list.append([anchor_y, anchor_x])
                
                block_geometry_dict[block_id] = {
                    'corner': np.array([corner_y, corner_x]),
                    'lsb_dots': np.array(lsb_dots),
                    'msb_dots': np.array(msb_dots),
                    'block_row': block_row,
                    'block_col': block_col
                }
                
                block_codes_list.append(block_id)
                block_id += 1
        
        return {
            'regular_dots': np.array(regular_dots_list),
            'code_dots': np.array(code_dots_list),
            'block_codes': np.array(block_codes_list),
            'anchor_points': np.array(anchor_points_list),
            'block_geometry': block_geometry_dict
        }


class DotDetector2D:
    """Detect and classify dots in coded calibration plate images"""
    
    def __init__(self, geometry: Optional[CodedDotPlateGeometry] = None):
        """
        Initialize 2D dot detector.
        
        Args:
            geometry: CodedDotPlateGeometry object (creates default if None)
        """
        self.geometry = geometry or CodedDotPlateGeometry()
        self.design_positions = self.geometry.get_all_design_positions_mm()
        self.homography_matrix = None
        self.um_per_pixel = None
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load a .tif image.
        
        Args:
            image_path: Path to .tif file
            
        Returns:
            2D numpy array (grayscale image)
        """
        try:
            from PIL import Image
            img = Image.open(image_path)
            img_array = np.array(img)
            if len(img_array.shape) == 3:
                # Convert RGB to grayscale
                img_array = np.mean(img_array, axis=2)
            logger.info(f"Loaded image from {image_path}: shape {img_array.shape}, dtype {img_array.dtype}")
            return img_array
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise
    
    def detect_and_classify_dots(self, image: np.ndarray,
                                 gaussian_sigma: float = 1.5,
                                 min_area_pixels: int = 30,
                                 max_area_pixels: int = 50000,
                                 regular_dot_area_range: Tuple[int, int] = (80, 500),
                                 code_dot_area_range: Tuple[int, int] = (200, 2000)) -> Dict[str, np.ndarray]:
        """
        Detect dots and classify them by size (regular vs code).
        
        Args:
            image: 2D grayscale image
            gaussian_sigma: Gaussian smoothing sigma (pixels)
            min_area_pixels: Minimum connected component area (pixels)
            max_area_pixels: Maximum connected component area (pixels)
            regular_dot_area_range: (min, max) area for 500 µm dots
            code_dot_area_range: (min, max) area for 800 µm dots
            
        Returns:
            Dictionary with:
            {
                'regular_dots': (N, 2) array [y, x] in pixels,
                'code_dots': (M, 2) array [y, x] in pixels,
                'all_dots': (N+M, 2) array of all dots,
                'dot_properties': regionprops list
            }
        """
        logger.info(f"Detecting and classifying dots in image shape {image.shape}...")
        
        # Normalize image
        img_norm = (image.astype(float) - image.min()) / (image.max() - image.min() + 1e-8)
        
        # Smooth
        img_smooth = ndimage.gaussian_filter(img_norm, sigma=gaussian_sigma)
        
        # Determine threshold (try both polarities)
        threshold_dark = np.percentile(img_smooth, 25)
        threshold_light = np.percentile(img_smooth, 75)
        
        binary_dark = img_smooth < threshold_dark
        binary_light = img_smooth > threshold_light
        
        # Label connected components for both
        labeled_dark, num_dark = ndimage_label(binary_dark)
        labeled_light, num_light = ndimage_label(binary_light)
        
        # Pick the interpretation with more components
        if num_dark > num_light:
            binary = binary_dark
            labeled = labeled_dark
            logger.info(f"Using dark dots: {num_dark} components detected")
        else:
            binary = binary_light
            labeled = labeled_light
            logger.info(f"Using light dots: {num_light} components detected")
        
        # Extract region properties
        regions = regionprops(labeled, intensity_image=img_smooth)
        
        # Filter by area range
        filtered_regions = [r for r in regions 
                           if min_area_pixels <= r.area <= max_area_pixels]
        logger.info(f"Filtered to {len(filtered_regions)} dots by area")
        
        # Classify by size
        regular_dots = []
        code_dots = []
        
        for region in filtered_regions:
            centroid = np.array(region.centroid)  # [y, x]
            area = region.area
            
            if regular_dot_area_range[0] <= area <= regular_dot_area_range[1]:
                regular_dots.append(centroid)
            elif code_dot_area_range[0] <= area <= code_dot_area_range[1]:
                code_dots.append(centroid)
        
        regular_dots = np.array(regular_dots) if regular_dots else np.empty((0, 2))
        code_dots = np.array(code_dots) if code_dots else np.empty((0, 2))
        
        logger.info(f"Classified: {len(regular_dots)} regular dots, {len(code_dots)} code dots")
        
        return {
            'regular_dots': regular_dots,
            'code_dots': code_dots,
            'all_dots': np.vstack([regular_dots, code_dots]) if len(regular_dots) > 0 and len(code_dots) > 0 else (regular_dots if len(regular_dots) > 0 else code_dots),
            'dot_properties': filtered_regions
        }
    
    def estimate_pixel_pitch(self, regular_dots_px: np.ndarray) -> float:
        """
        Estimate the average pixel spacing between regular dots.
        
        Uses pairwise distances to nearby neighbors to robustly estimate the dot pitch.
        
        Args:
            regular_dots_px: Detected regular dots (N, 2) in pixels
            
        Returns:
            Estimated pixel pitch (pixels between dot centers)
        """
        if len(regular_dots_px) < 10:
            logger.warning("Too few dots to estimate pixel pitch; using default 30px")
            return 30.0
        
        # Compute pairwise distances
        dists = distance_matrix(regular_dots_px, regular_dots_px)
        
        # Set diagonal to infinity (self-distance)
        np.fill_diagonal(dists, np.inf)
        
        # For each dot, find the 3 nearest neighbors
        nearest_dists = []
        for i in range(len(regular_dots_px)):
            neighbors = np.argsort(dists[i])[:3]
            nearest_dists.extend(dists[i, neighbors])
        
        # Filter out very small distances (noise) and very large distances
        nearest_dists = np.array(nearest_dists)
        nearest_dists = nearest_dists[nearest_dists > 1.0]  # Remove near-zero
        
        # Use median of nearest neighbor distances as the pitch estimate
        pixel_pitch = np.median(nearest_dists)
        
        logger.info(f"Estimated pixel pitch: {pixel_pitch:.2f} pixels")
        return pixel_pitch
    
    def identify_anchor_points(self, code_dots_px: np.ndarray,
                              regular_dots_px: np.ndarray,
                              pixel_pitch: Optional[float] = None) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """
        Identify L-shaped anchor points (block codes) from code dots.
        
        Clusters code dots spatially to find distinct L-markers, decodes their block IDs,
        and computes the anchor center for each cluster.
        
        Args:
            code_dots_px: Detected code dots (N, 2) in pixels
            regular_dots_px: Detected regular dots (M, 2) in pixels (for pitch estimation)
            pixel_pitch: Estimated pixel pitch (pixels). If None, auto-estimated from regular dots.
            
        Returns:
            (anchor_points_px, block_ids, block_decode_results)
            - anchor_points_px: (K, 2) detected anchor centers in pixels
            - block_ids: (K,) identified block IDs (0-79)
            - block_decode_results: List of decode result dicts with metadata
        """
        logger.info(f"Identifying anchor points from {len(code_dots_px)} code dots...")
        
        if len(code_dots_px) == 0:
            logger.warning("No code dots detected for anchor identification")
            return np.empty((0, 2)), np.empty((0,), dtype=int), []
        
        # Estimate or validate pixel pitch
        if pixel_pitch is None:
            pixel_pitch = self.estimate_pixel_pitch(regular_dots_px)
        else:
            logger.info(f"Using provided pixel pitch: {pixel_pitch:.2f} pixels")
        
        # Adaptive clustering threshold based on pixel pitch
        # Expect code dots within ~3-4 dot spacings (~3-4mm physically = code marker size)
        # L-marker cluster is typically 7×7 dot positions within a block
        cluster_threshold_px = pixel_pitch * 10.0  # ~10 dot spacings
        
        logger.info(f"Adaptive cluster threshold: {cluster_threshold_px:.2f} pixels")
        
        # Hierarchical clustering
        from scipy.cluster.hierarchy import fclusterdata
        
        clusters = fclusterdata(code_dots_px, t=cluster_threshold_px, criterion='distance', method='complete')
        num_clusters = len(np.unique(clusters))
        logger.info(f"Identified {num_clusters} L-marker clusters (code blocks)")
        
        anchor_points = []
        block_ids = []
        decode_results = []
        
        for cluster_id in np.unique(clusters):
            cluster_dots = code_dots_px[clusters == cluster_id]
            
            # Decode the block code from this L-marker cluster
            decode_result = self.decode_block_from_code_dots(cluster_dots, pixel_pitch)
            
            if decode_result is not None:
                anchor = decode_result['l_anchor_px']
                block_id = decode_result['block_code']
                anchor_points.append(anchor)
                block_ids.append(block_id)
                decode_results.append(decode_result)
                logger.info(f"Cluster {cluster_id}: Block ID={block_id}, Confidence={decode_result['confidence']:.3f}, "
                           f"Rotation={decode_result['rotation_angle']:.1f}°")
            else:
                logger.warning(f"Failed to decode cluster {cluster_id}")
        
        anchor_points = np.array(anchor_points) if anchor_points else np.empty((0, 2))
        block_ids = np.array(block_ids) if block_ids else np.empty((0,), dtype=int)
        
        return anchor_points, block_ids, decode_results
    
    def decode_block_from_code_dots(self, code_dots_cluster_px: np.ndarray, 
                                    pixel_pitch: float) -> Optional[Dict]:
        """
        Decode block ID from L-shaped code marker cluster.
        
        The L-marker consists of:
        1. Corner dot (forms right-angle reference)
        2. LSB horizontal sequence (6 bits along one axis)
        3. MSB vertical sequence (6 bits along perpendicular axis)
        
        Algorithm:
        1. Find the corner dot (forms L-angle with other two arms)
        2. Establish local axes from corner to endpoints
        3. Scan each axis to detect presence/absence of code dots (binary encoding)
        4. Decode the 12-bit code (or however the plate encodes it)
        
        Args:
            code_dots_cluster_px: Detected code dots in one L-cluster (N, 2) in pixels
            pixel_pitch: Pixel pitch (pixels between dot centers)
            
        Returns:
            Dictionary with:
            {
                'block_code': Decoded block ID (0-79),
                'l_anchor_px': L-anchor center in pixels,
                'corner_dot_px': Identified corner dot position,
                'lsb_sequence': List of 6 binary values (0=no dot, 1=dot present),
                'msb_sequence': List of 6 binary values,
                'rotation_angle': Rotation of L-marker in degrees,
                'confidence': Confidence score (0-1)
            }
            or None if decode fails
        """
        logger.debug(f"Decoding L-marker cluster with {len(code_dots_cluster_px)} dots")
        
        if len(code_dots_cluster_px) < 3:
            logger.warning(f"Cluster too small for L-marker ({len(code_dots_cluster_px)} < 3)")
            return None
        
        # Step 1: Find the corner dot (forms right angle)
        # The corner should be equidistant to endpoints of the two arms
        corner_candidate_idx = self._find_corner_dot(code_dots_cluster_px, pixel_pitch)
        
        if corner_candidate_idx is None:
            logger.warning("Could not identify corner dot in cluster")
            return None
        
        corner_dot = code_dots_cluster_px[corner_candidate_idx]
        remaining_dots = np.delete(code_dots_cluster_px, corner_candidate_idx, axis=0)
        
        logger.debug(f"Corner dot identified at {corner_dot}")
        
        # Step 2: Establish local axes
        # Partition remaining dots into two arms (LSB and MSB sequences)
        lsb_arm_dots, msb_arm_dots, lsb_direction, msb_direction = self._partition_arms(
            corner_dot, remaining_dots, pixel_pitch
        )
        
        if lsb_arm_dots is None or msb_arm_dots is None:
            logger.warning("Could not partition code dots into LSB/MSB arms")
            return None
        
        logger.debug(f"LSB arm: {len(lsb_arm_dots)} dots, direction {lsb_direction}")
        logger.debug(f"MSB arm: {len(msb_arm_dots)} dots, direction {msb_direction}")
        
        # Step 3: Scan each axis to extract binary sequences
        lsb_sequence = self._extract_binary_sequence(corner_dot, lsb_arm_dots, lsb_direction, pixel_pitch, num_bits=6)
        msb_sequence = self._extract_binary_sequence(corner_dot, msb_arm_dots, msb_direction, pixel_pitch, num_bits=6)
        
        logger.debug(f"LSB sequence: {lsb_sequence}")
        logger.debug(f"MSB sequence: {msb_sequence}")
        
        # Step 4: Decode the block code
        # Interpret binary sequences as a block ID
        # Format: block_id = row * cols + col (within macro-grid)
        # Or could be direct binary encoding; adjust based on your plate spec
        block_code = self._binary_to_block_id(lsb_sequence, msb_sequence)
        
        # Compute anchor center
        l_anchor = np.mean(code_dots_cluster_px, axis=0)
        
        # Compute rotation angle of L-marker
        rotation_angle = np.arctan2(msb_direction[0], msb_direction[1]) * 180.0 / np.pi
        
        # Confidence: based on how cleanly we can identify the arms and sequences
        confidence = self._compute_decode_confidence(lsb_sequence, msb_sequence)
        
        return {
            'block_code': block_code,
            'l_anchor_px': l_anchor,
            'corner_dot_px': corner_dot,
            'lsb_sequence': lsb_sequence,
            'msb_sequence': msb_sequence,
            'rotation_angle': rotation_angle,
            'confidence': confidence
        }
    
    def _find_corner_dot(self, dots: np.ndarray, pixel_pitch: float) -> Optional[int]:
        """
        Identify the corner dot (right-angle reference point) in an L-cluster.
        
        The corner is the dot with two arms extending at ~90° angles.
        Heuristic: compute angles between all triples, find the dot that participates
        in the most ~90° angle pairs.
        
        Args:
            dots: Cluster of code dots (N, 2)
            pixel_pitch: Expected spacing (pixels)
            
        Returns:
            Index of the corner dot, or None
        """
        if len(dots) < 3:
            return None
        
        best_corner_idx = None
        best_score = -1
        
        # Try each dot as a potential corner
        for i in range(len(dots)):
            corner = dots[i]
            other_dots = np.delete(dots, i, axis=0)
            
            # Compute vectors from corner to other dots
            vectors = other_dots - corner
            
            # Compute angles between all pairs of vectors
            angles = []
            for j in range(len(vectors)):
                for k in range(j + 1, len(vectors)):
                    v1 = vectors[j]
                    v2 = vectors[k]
                    
                    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180.0 / np.pi
                    angles.append(angle)
            
            # Score: how close the angles are to 90°
            angles = np.array(angles)
            score = -np.min(np.abs(angles - 90.0))  # Closer to 90° = higher (less negative) score
            
            if score > best_score:
                best_score = score
                best_corner_idx = i
        
        return best_corner_idx
    
    def _partition_arms(self, corner_dot: np.ndarray, other_dots: np.ndarray, 
                       pixel_pitch: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], 
                                                    Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Partition the non-corner dots into LSB and MSB arms.
        
        Args:
            corner_dot: Position of corner dot (2,)
            other_dots: Remaining dots (N, 2)
            pixel_pitch: Expected spacing
            
        Returns:
            (lsb_arm_dots, msb_arm_dots, lsb_direction, msb_direction)
            or (None, None, None, None) if partition fails
        """
        if len(other_dots) < 2:
            return None, None, None, None
        
        # Compute vectors from corner
        vectors = other_dots - corner_dot
        distances = np.linalg.norm(vectors, axis=1)
        
        # Find two distinct directions (arms should extend in different directions)
        # Use clustering on the angles
        angles = np.arctan2(vectors[:, 0], vectors[:, 1])
        
        # Partition by angle: expect ~90° separation
        # Try all pairs of angles and find the pair closest to 90°
        best_partition = None
        best_diff = np.inf
        
        for i in range(len(angles)):
            for j in range(i + 1, len(angles)):
                angle_diff = np.abs(angles[i] - angles[j])
                # Normalize to [0, 180]
                if angle_diff > 180:
                    angle_diff = 360 - angle_diff
                
                # Check if close to 90°
                dev_from_90 = np.abs(angle_diff - 90.0)
                if dev_from_90 < best_diff:
                    best_diff = dev_from_90
                    best_partition = (i, j)
        
        if best_partition is None:
            return None, None, None, None
        
        i, j = best_partition
        
        # Collect dots for each arm
        lsb_arm_dots = other_dots[i:i+1]  # Dots near angle i
        msb_arm_dots = other_dots[j:j+1]  # Dots near angle j
        
        # Add other dots to the nearest arm
        for k in range(len(other_dots)):
            if k != i and k != j:
                angle_to_i = np.abs(angles[k] - angles[i])
                angle_to_j = np.abs(angles[k] - angles[j])
                if angle_to_i < angle_to_j:
                    lsb_arm_dots = np.vstack([lsb_arm_dots, other_dots[k:k+1]])
                else:
                    msb_arm_dots = np.vstack([msb_arm_dots, other_dots[k:k+1]])
        
        # Compute direction vectors (normalize)
        lsb_direction = np.mean(vectors[i:i+1], axis=0)
        lsb_direction = lsb_direction / (np.linalg.norm(lsb_direction) + 1e-8)
        
        msb_direction = np.mean(vectors[j:j+1], axis=0)
        msb_direction = msb_direction / (np.linalg.norm(msb_direction) + 1e-8)
        
        return lsb_arm_dots, msb_arm_dots, lsb_direction, msb_direction
    
    def _extract_binary_sequence(self, corner_dot: np.ndarray, arm_dots: np.ndarray,
                                direction: np.ndarray, pixel_pitch: float, 
                                num_bits: int = 6) -> List[int]:
        """
        Extract binary sequence along an arm of the L-marker.
        
        Args:
            corner_dot: Position of corner (2,)
            arm_dots: Dots along this arm (N, 2)
            direction: Unit direction vector of arm (2,)
            pixel_pitch: Pixel pitch
            num_bits: Expected number of bits
            
        Returns:
            List of binary values (0 or 1) for each bit position
        """
        sequence = []
        
        # Generate expected positions along the arm
        for bit_idx in range(num_bits):
            expected_pos = corner_dot + direction * (bit_idx + 1) * pixel_pitch
            
            # Check if a dot exists near this position
            if len(arm_dots) == 0:
                sequence.append(0)
                continue
            
            dists = np.linalg.norm(arm_dots - expected_pos, axis=1)
            min_dist = np.min(dists)
            
            # Threshold: if a dot exists within ~half pitch, mark as 1
            if min_dist < pixel_pitch * 0.6:
                sequence.append(1)
            else:
                sequence.append(0)
        
        return sequence
    
    def _binary_to_block_id(self, lsb_sequence: List[int], msb_sequence: List[int]) -> int:
        """
        Decode block ID from LSB and MSB binary sequences.
        
        Interprets the sequences as a block code within the macro-grid.
        Assumes:
        - LSB sequence encodes column within macro-grid (0-9)
        - MSB sequence encodes row within macro-grid (0-7)
        
        Args:
            lsb_sequence: List of 6 bits (LSB position)
            msb_sequence: List of 6 bits (MSB position)
            
        Returns:
            Block ID (0-79)
        """
        # Convert binary sequences to decimals
        lsb_val = sum(bit * (2 ** idx) for idx, bit in enumerate(lsb_sequence))
        msb_val = sum(bit * (2 ** idx) for idx, bit in enumerate(msb_sequence))
        
        # Normalize to grid ranges (0-9 for cols, 0-7 for rows)
        col = lsb_val % self.geometry.macro_cols
        row = msb_val % self.geometry.macro_rows
        
        block_id = row * self.geometry.macro_cols + col
        
        return block_id
    
    def _compute_decode_confidence(self, lsb_sequence: List[int], msb_sequence: List[int]) -> float:
        """
        Compute confidence score for the block decode.
        
        Based on:
        - Whether sequences follow expected binary patterns
        - Presence of expected dots
        
        Args:
            lsb_sequence: Extracted LSB bits
            msb_sequence: Extracted MSB bits
            
        Returns:
            Confidence score (0-1)
        """
        # Simple heuristic: confidence based on number of detected bits
        # (more bits = cleaner L-marker)
        total_bits = sum(lsb_sequence) + sum(msb_sequence)
        max_bits = len(lsb_sequence) + len(msb_sequence)
        
        # Confidence: expect at least some bits to be present
        confidence = min(1.0, total_bits / max(1, max_bits * 0.5))
        
        return confidence
    
    def compute_homography_from_anchors(self, detected_anchors_px: np.ndarray,
                                       design_anchors_mm: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Compute homography matrix mapping detected image coordinates to design mm coordinates.
        
        Args:
            detected_anchors_px: Detected anchor centers (N, 2) in pixels
            design_anchors_mm: Design anchor positions (N, 2) in mm
            
        Returns:
            (homography_matrix, error_residual)
            - homography_matrix: 3×3 homography matrix
            - error_residual: RMS reprojection error in pixels
        """
        logger.info(f"Computing homography from {len(detected_anchors_px)} anchor pairs...")
        
        if len(detected_anchors_px) < 4:
            logger.error("Need at least 4 anchor pairs for homography")
            raise ValueError("Insufficient anchor pairs")
        
        # Compute homography using OpenCV
        H, status = cv2.findHomography(detected_anchors_px, design_anchors_mm, cv2.RANSAC, 5.0)
        
        if H is None:
            logger.error("Homography computation failed")
            raise ValueError("Homography computation failed")
        
        # Compute reprojection error
        detected_homog = np.hstack([detected_anchors_px, np.ones((len(detected_anchors_px), 1))])
        projected_mm = (H @ detected_homog.T).T
        projected_mm = projected_mm[:, :2] / projected_mm[:, 2:3]
        
        errors = np.linalg.norm(projected_mm - design_anchors_mm, axis=1)
        error_residual = np.sqrt(np.mean(errors**2))
        
        logger.info(f"Homography computed. RMS reprojection error: {error_residual:.4f} pixels")
        
        self.homography_matrix = H
        
        # Estimate µm per pixel from homography scale
        px_corner = np.array([[0, 0], [detected_anchors_px.max(axis=0)[0], 0],
                             [0, detected_anchors_px.max(axis=0)[1]]])
        mm_corner = (H @ np.hstack([px_corner, np.ones((3, 1))]).T).T
        mm_corner = mm_corner[:, :2] / mm_corner[:, 2:3]
        
        scale_y = np.abs(mm_corner[2, 0] - mm_corner[0, 0]) / np.abs(px_corner[2, 0] - px_corner[0, 0])
        scale_x = np.abs(mm_corner[1, 1] - mm_corner[0, 1]) / np.abs(px_corner[1, 1] - px_corner[0, 1])
        
        self.um_per_pixel = (scale_x + scale_y) / 2.0 * 1000.0
        logger.info(f"Estimated resolution: {self.um_per_pixel:.2f} µm/pixel")
        
        return H, error_residual
    
    def project_pixels_to_mm(self, points_px: np.ndarray) -> np.ndarray:
        """
        Project pixel coordinates to mm using homography matrix.
        
        Args:
            points_px: Points in pixels (N, 2)
            
        Returns:
            Points in mm (N, 2)
        """
        if self.homography_matrix is None:
            raise ValueError("Homography matrix not computed. Call compute_homography_from_anchors first.")
        
        points_homog = np.hstack([points_px, np.ones((len(points_px), 1))])
        points_mm_homog = (self.homography_matrix @ points_homog.T).T
        points_mm = points_mm_homog[:, :2] / points_mm_homog[:, 2:3]
        
        return points_mm
    
    def match_regular_dots_to_design(self, detected_regular_px: np.ndarray,
                                     design_regular_mm: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Match detected regular dots to design grid after homography alignment.
        
        Args:
            detected_regular_px: Detected regular dots (N, 2) in pixels
            design_regular_mm: Design regular dots (M, 2) in mm
            
        Returns:
            (design_matched_mm, detected_matched_mm, discrepancies_um)
            - design_matched_mm: Design positions matched to detected (N, 2) in mm
            - detected_matched_mm: Detected positions (N, 2) in mm
            - discrepancies_um: Spatial discrepancies (N, 2) in µm
        """
        logger.info(f"Matching {len(detected_regular_px)} detected dots to {len(design_regular_mm)} design dots...")
        
        # Project detected dots to mm using homography
        detected_mm = self.project_pixels_to_mm(detected_regular_px)
        
        # Compute nearest-neighbor matching
        dists = distance_matrix(detected_mm, design_regular_mm)
        matched_design_idx = np.argmin(dists, axis=1)
        
        design_matched = design_regular_mm[matched_design_idx]
        
        # Compute discrepancies in µm
        discrepancies_mm = detected_mm - design_matched
        discrepancies_um = discrepancies_mm * 1000.0
        
        logger.info(f"Matched {len(detected_mm)} regular dots")
        logger.info(f"Mean discrepancy: dy={discrepancies_um[:, 0].mean():.2f}µm, dx={discrepancies_um[:, 1].mean():.2f}µm")
        logger.info(f"Std discrepancy: dy={discrepancies_um[:, 0].std():.2f}µm, dx={discrepancies_um[:, 1].std():.2f}µm")
        logger.info(f"Max discrepancy: {np.max(np.abs(discrepancies_um)):.2f}µm")
        
        return design_matched, detected_mm, discrepancies_um
    
    def process_image_file(self, image_path: str, 
                          output_dir: str = None,
                          pixel_pitch: Optional[float] = None) -> Dict:
        """
        Complete pipeline: load image, detect, decode blocks, compute homography, match dots.
        
        Args:
            image_path: Path to .tif image
            output_dir: Output directory for results
            pixel_pitch: Optional pre-computed pixel pitch for clustering
            
        Returns:
            Dictionary with complete analysis results
        """
        import os
        
        # Load image
        image = self.load_image(image_path)
        
        # Detect and classify dots
        detections = self.detect_and_classify_dots(image)
        regular_dots_px = detections['regular_dots']
        code_dots_px = detections['code_dots']
        
        # Identify anchor points from code dots with adaptive clustering
        detected_anchors_px, block_ids, decode_results = self.identify_anchor_points(
            code_dots_px, regular_dots_px, pixel_pitch=pixel_pitch
        )
        
        # Get design anchor positions
        design_anchors_mm = self.design_positions['anchor_points']
        
        # Match detected anchors to design
        if len(detected_anchors_px) > 0:
            dists = distance_matrix(detected_anchors_px, design_anchors_mm)
            matched_design_idx = np.argmin(dists, axis=1)
            design_anchors_matched = design_anchors_mm[matched_design_idx]
            
            # Compute homography
            H, error = self.compute_homography_from_anchors(detected_anchors_px, design_anchors_matched)
            
            # Match regular dots to design grid
            design_matched, detected_mm, discrepancies_um = self.match_regular_dots_to_design(
                regular_dots_px, self.design_positions['regular_dots']
            )
        else:
            logger.warning("No anchor points detected; skipping homography and dot matching")
            design_matched = None
            detected_mm = None
            discrepancies_um = None
        
        results = {
            'image': image,
            'regular_dots_px': regular_dots_px,
            'code_dots_px': code_dots_px,
            'detected_anchors_px': detected_anchors_px,
            'block_ids': block_ids,
            'decode_results': decode_results,
            'design_anchors_mm': design_anchors_mm,
            'homography_matrix': self.homography_matrix,
            'detected_regular_mm': detected_mm,
            'design_regular_mm': design_matched,
            'discrepancies_um': discrepancies_um,
            'um_per_pixel': self.um_per_pixel
        }
        
        # Visualize
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.visualize_results(image, regular_dots_px, code_dots_px, detected_anchors_px,
                                  os.path.join(output_dir, 'detection_visualization.png'))
            
            # Save results
            results_path = os.path.join(output_dir, 'analysis_results.npz')
            np.savez(results_path, 
                    regular_dots_px=regular_dots_px,
                    code_dots_px=code_dots_px,
                    detected_anchors_px=detected_anchors_px,
                    block_ids=block_ids,
                    discrepancies_um=discrepancies_um)
            logger.info(f"Saved results to {results_path}")
            
            # Save summary
            summary_path = os.path.join(output_dir, 'summary.txt')
            self.write_summary(summary_path, results)
        
        return results
    
    def visualize_results(self, image: np.ndarray, 
                         regular_dots_px: np.ndarray,
                         code_dots_px: np.ndarray,
                         anchors_px: np.ndarray,
                         output_path: str) -> None:
        """Visualize detection results"""
        try:
            import matplotlib.pyplot as plt
            
            fig, axes = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left: Image with all detections
            ax = axes[0]
            ax.imshow(image, cmap='gray', origin='upper')
            if len(regular_dots_px) > 0:
                ax.plot(regular_dots_px[:, 1], regular_dots_px[:, 0], 'go', 
                       markersize=6, label='Regular Dots (500µm)', alpha=0.6)
            if len(code_dots_px) > 0:
                ax.plot(code_dots_px[:, 1], code_dots_px[:, 0], 'r^', 
                       markersize=8, label='Code Dots (800µm)', alpha=0.7)
            if len(anchors_px) > 0:
                ax.plot(anchors_px[:, 1], anchors_px[:, 0], 'b*', 
                       markersize=15, label='Block Anchors', alpha=0.8)
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            ax.set_title('Dot Detection: Regular + Code Markers')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Right: Magnified view of one anchor region
            if len(anchors_px) > 0:
                ax = axes[1]
                anchor = anchors_px[0].astype(int)
                region_size = 200
                y_min = max(0, anchor[0] - region_size)
                y_max = min(image.shape[0], anchor[0] + region_size)
                x_min = max(0, anchor[1] - region_size)
                x_max = min(image.shape[1], anchor[1] + region_size)
                
                region = image[y_min:y_max, x_min:x_max]
                ax.imshow(region, cmap='gray', origin='upper')
                
                # Plot dots in this region
                mask_reg = ((regular_dots_px[:, 0] >= y_min) & (regular_dots_px[:, 0] < y_max) &
                           (regular_dots_px[:, 1] >= x_min) & (regular_dots_px[:, 1] < x_max))
                mask_code = ((code_dots_px[:, 0] >= y_min) & (code_dots_px[:, 0] < y_max) &
                            (code_dots_px[:, 1] >= x_min) & (code_dots_px[:, 1] < x_max))
                
                if mask_reg.any():
                    rel_reg = regular_dots_px[mask_reg] - np.array([y_min, x_min])
                    ax.plot(rel_reg[:, 1], rel_reg[:, 0], 'go', markersize=5, alpha=0.6)
                if mask_code.any():
                    rel_code = code_dots_px[mask_code] - np.array([y_min, x_min])
                    ax.plot(rel_code[:, 1], rel_code[:, 0], 'r^', markersize=7, alpha=0.7)
                
                ax.set_xlabel('X (pixels)')
                ax.set_ylabel('Y (pixels)')
                ax.set_title(f'Magnified view: Block anchor @ ({anchor[0]}, {anchor[1]})')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved visualization to {output_path}")
            plt.close()
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
    
    def write_summary(self, output_path: str, results: Dict) -> None:
        """Write analysis summary to text file"""
        try:
            with open(output_path, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("CODED CALIBRATION PLATE ANALYSIS SUMMARY\n")
                f.write("Dot-Plate 500-1000 (8×10 macro-grid)\n")
                f.write("=" * 80 + "\n\n")
                
                f.write("DETECTION RESULTS:\n")
                f.write(f"  Regular dots detected: {len(results['regular_dots_px'])}\n")
                f.write(f"  Code dots detected: {len(results['code_dots_px'])}\n")
                f.write(f"  Block anchors identified: {len(results['detected_anchors_px'])}\n")
                f.write(f"  Block IDs decoded: {list(results['block_ids'])}\n\n")
                
                if results['decode_results']:
                    f.write("BLOCK DECODE DETAILS:\n")
                    for i, decode in enumerate(results['decode_results']):
                        f.write(f"  Block {i}: ID={decode['block_code']}, ")
                        f.write(f"Rotation={decode['rotation_angle']:.1f}°, ")
                        f.write(f"Confidence={decode['confidence']:.3f}\n")
                        f.write(f"    LSB: {decode['lsb_sequence']}\n")
                        f.write(f"    MSB: {decode['msb_sequence']}\n")
                    f.write("\n")
                
                if results['homography_matrix'] is not None:
                    f.write("HOMOGRAPHY CALIBRATION:\n")
                    f.write(f"  Resolution: {results['um_per_pixel']:.2f} µm/pixel\n")
                    f.write(f"  Homography matrix:\n")
                    f.write(f"  {results['homography_matrix']}\n\n")
                
                if results['discrepancies_um'] is not None:
                    discr = results['discrepancies_um']
                    f.write("SPATIAL DISCREPANCIES (µm):\n")
                    f.write(f"  Mean: dy={discr[:, 0].mean():.2f}µm, dx={discr[:, 1].mean():.2f}µm\n")
                    f.write(f"  Std:  dy={discr[:, 0].std():.2f}µm, dx={discr[:, 1].std():.2f}µm\n")
                    f.write(f"  Max:  {np.max(np.abs(discr)):.2f}µm\n")
                    f.write(f"  Min:  {np.min(np.abs(discr)):.2f}µm\n\n")
                
                f.write("=" * 80 + "\n")
            
            logger.info(f"Saved summary to {output_path}")
        except Exception as e:
            logger.error(f"Summary write failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                       format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    logger.info("Code-aware Dot Detector 2D initialized")
    detector = DotDetector2D()
    logger.info(f"Design geometry: {detector.geometry.macro_rows}×{detector.geometry.macro_cols} macro-grid")
    logger.info(f"Total design regular dots: {len(detector.design_positions['regular_dots'])}")
    logger.info(f"Total design code dots: {len(detector.design_positions['code_dots'])}")
    logger.info("✓ Ready for image processing")
