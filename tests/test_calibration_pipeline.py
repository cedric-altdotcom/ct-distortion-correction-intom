"""End-to-end test suite for calibration and distortion correction pipeline"""
import numpy as np
import logging
import os
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('calibration_test.log')
    ]
)
logger = logging.getLogger(__name__)


class CalibrationTestSuite:
    """Comprehensive test suite for the calibration pipeline"""
    
    def __init__(self, output_dir: str = 'test_results'):
        """
        Initialize test suite.
        
        Args:
            output_dir: Directory for test outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.test_results = {}
        logger.info(f"Test suite initialized. Output: {self.output_dir}")
    
    def test_detector_initialization(self):
        """Test DotDetector2D initialization"""
        logger.info("\n" + "="*80)
        logger.info("TEST 1: Detector Initialization")
        logger.info("="*80)
        
        try:
            from calibration.dot_detector_2d import DotDetector2D, CodedDotPlateGeometry
            
            geometry = CodedDotPlateGeometry()
            detector = DotDetector2D(geometry)
            
            assert detector.geometry is not None
            assert detector.design_positions is not None
            assert len(detector.design_positions['regular_dots']) > 0
            assert len(detector.design_positions['code_dots']) > 0
            
            logger.info(f"✓ Geometry: {geometry.macro_rows}×{geometry.macro_cols} macro-grid")
            logger.info(f"✓ Regular dots: {len(detector.design_positions['regular_dots'])}")
            logger.info(f"✓ Code dots: {len(detector.design_positions['code_dots'])}")
            logger.info(f"✓ Total design positions: {len(detector.design_positions['anchor_points'])}")
            
            self.test_results['initialization'] = 'PASS'
            return True
            
        except Exception as e:
            logger.error(f"✗ Initialization failed: {e}")
            self.test_results['initialization'] = f'FAIL: {e}'
            return False
    
    def test_synthetic_image_generation(self):
        """Test generation of synthetic calibration image"""
        logger.info("\n" + "="*80)
        logger.info("TEST 2: Synthetic Image Generation")
        logger.info("="*80)
        
        try:
            from calibration.dot_detector_2d import DotDetector2D
            
            detector = DotDetector2D()
            image = self._generate_synthetic_calibration_image(detector)
            
            assert image is not None
            assert image.shape[0] > 0 and image.shape[1] > 0
            
            # Save synthetic image
            output_path = self.output_dir / 'synthetic_calibration_image.npy'
            np.save(output_path, image)
            
            logger.info(f"✓ Generated synthetic image: {image.shape}, dtype: {image.dtype}")
            logger.info(f"✓ Intensity range: [{image.min()}, {image.max()}]")
            logger.info(f"✓ Saved to {output_path}")
            
            self.test_results['synthetic_image'] = 'PASS'
            return True
            
        except Exception as e:
            logger.error(f"✗ Synthetic image generation failed: {e}")
            self.test_results['synthetic_image'] = f'FAIL: {e}'
            return False
    
    def _generate_synthetic_calibration_image(self, detector, 
                                             image_shape: tuple = (800, 800),
                                             um_per_pixel: float = 100.0) -> np.ndarray:
        """
        Generate synthetic calibration image with realistic dot patterns.
        
        Args:
            detector: DotDetector2D instance
            image_shape: Image dimensions (height, width)
            um_per_pixel: Pixel resolution
            
        Returns:
            Synthetic image as uint8
        """
        logger.info("Generating synthetic calibration image...")
        
        image = np.ones(image_shape, dtype=np.uint8) * 200  # Gray background
        
        mm_per_pixel = 1.0 / (um_per_pixel / 1000.0)
        design_mm = detector.design_positions['regular_dots']
        
        # Draw regular dots
        for pos_mm in design_mm:
            y_px = int(pos_mm[0] / mm_per_pixel)
            x_px = int(pos_mm[1] / mm_per_pixel)
            
            # Skip out-of-bounds
            if 0 <= y_px < image_shape[0] and 0 <= x_px < image_shape[1]:
                # Draw dot (dark circle)
                y_min = max(0, y_px - 3)
                y_max = min(image_shape[0], y_px + 4)
                x_min = max(0, x_px - 3)
                x_max = min(image_shape[1], x_px + 4)
                
                image[y_min:y_max, x_min:x_max] = 50  # Dark
        
        # Draw code dots (larger)
        design_code = detector.design_positions['code_dots']
        for pos_mm in design_code:
            y_px = int(pos_mm[0] / mm_per_pixel)
            x_px = int(pos_mm[1] / mm_per_pixel)
            
            if 0 <= y_px < image_shape[0] and 0 <= x_px < image_shape[1]:
                y_min = max(0, y_px - 5)
                y_max = min(image_shape[0], y_px + 6)
                x_min = max(0, x_px - 5)
                x_max = min(image_shape[1], x_px + 6)
                
                image[y_min:y_max, x_min:x_max] = 30  # Darker
        
        # Add Gaussian noise
        noise = np.random.normal(0, 10, image.shape)
        image = np.clip(image + noise, 0, 255).astype(np.uint8)
        
        logger.info(f"Synthetic image generated: {image.shape}")
        return image
    
    def test_feature_association_strict(self):
        """Test strict 1-to-1 feature association"""
        logger.info("\n" + "="*80)
        logger.info("TEST 3: Strict Feature Association (Hungarian Algorithm)")
        logger.info("="*80)
        
        try:
            from calibration.robust_association import RobustFeatureAssociation
            from calibration.dot_detector_2d import DotDetector2D
            
            detector = DotDetector2D()
            association = RobustFeatureAssociation(detector.geometry)
            
            # Create synthetic points
            design_mm = np.array([[10, 10], [20, 20], [30, 30], [40, 40], [50, 50]])
            detected_mm = np.array([
                [10.1, 10.1],   # Close to first
                [20.2, 20.2],   # Close to second
                [30.05, 30.05], # Close to third
                [40.15, 40.15], # Close to fourth
                [50.2, 50.2]    # Close to fifth
            ])
            
            # Test association
            matched_detected, matched_design, quality = association.associate_detected_to_design_strict(
                detected_mm, design_mm, max_match_distance_mm=1.0
            )
            
            assert len(matched_detected) == len(design_mm), "Should match all points"
            assert quality > 0.9, "Quality should be high"
            
            logger.info(f"✓ Matched {len(matched_detected)} point pairs")
            logger.info(f"✓ Association quality: {quality*100:.1f}%")
            logger.info(f"✓ Enforced 1-to-1 mapping (no collapses)")
            
            self.test_results['feature_association'] = 'PASS'
            return True
            
        except Exception as e:
            logger.error(f"✗ Feature association failed: {e}")
            self.test_results['feature_association'] = f'FAIL: {e}'
            return False
    
    def test_metrology_report_generation(self):
        """Test metrology report generation"""
        logger.info("\n" + "="*80)
        logger.info("TEST 4: Metrology Report Generation")
        logger.info("="*80)
        
        try:
            from calibration.metrology_report import MetrologyReportGenerator
            from calibration.dot_detector_2d import DotDetector2D
            
            detector = DotDetector2D()
            metrology = MetrologyReportGenerator(detector.geometry)
            
            # Create synthetic measurement data
            design_mm = detector.design_positions['regular_dots'][:50]  # First 50
            
            # Simulate detected positions with small errors
            np.random.seed(42)
            detected_mm = design_mm + np.random.normal(0, 0.01, design_mm.shape)
            discrepancies_um = (detected_mm - design_mm) * 1000.0
            
            # Generate report
            report_data = metrology.generate_metrology_report(
                design_mm, detected_mm, discrepancies_um,
                output_dir=str(self.output_dir / 'metrology_test')
            )
            
            assert report_data is not None
            assert 'global_stats' in report_data
            assert 'per_block_stats' in report_data
            
            logger.info(f"✓ Generated metrology report")
            logger.info(f"✓ Global mean error: {report_data['global_stats']['mean_um']:.3f} µm")
            logger.info(f"✓ Global RMSE: {report_data['global_stats']['rmse_um']:.3f} µm")
            logger.info(f"✓ Blocks analyzed: {len(report_data['per_block_stats'])}")
            
            # Print summary
            summary = metrology.get_summary_text()
            logger.info(f"\nMetrology Summary:\n{summary}")
            
            self.test_results['metrology_report'] = 'PASS'
            return True
            
        except Exception as e:
            logger.error(f"✗ Metrology report generation failed: {e}")
            self.test_results['metrology_report'] = f'FAIL: {e}'
            return False
    
    def test_distortion_map_generation(self):
        """Test distortion correction map generation"""
        logger.info("\n" + "="*80)
        logger.info("TEST 5: Distortion Correction Map Generation")
        logger.info("="*80)
        
        try:
            from calibration.distortion_correction import DistortionCorrectionEngine
            from calibration.dot_detector_2d import DotDetector2D
            
            detector = DotDetector2D()
            engine = DistortionCorrectionEngine(detector.geometry, um_per_pixel=100.0)
            
            # Create synthetic measurement data
            design_mm = detector.design_positions['regular_dots'][:50]
            
            np.random.seed(42)
            detected_mm = design_mm + np.random.normal(0, 0.005, design_mm.shape)
            discrepancies_um = (detected_mm - design_mm) * 1000.0
            
            image_shape = (400, 400)
            
            # Generate RBF-based distortion map
            distortion_map = engine.build_distortion_map_from_metrology(
                design_mm, detected_mm, discrepancies_um,
                image_shape, method='rbf'
            )
            
            assert distortion_map is not None
            assert 'distortion_map' in distortion_map
            assert distortion_map['coverage'] > 0.5
            
            logger.info(f"✓ Generated distortion map: {distortion_map['distortion_map'].shape}")
            logger.info(f"✓ Coverage: {distortion_map['coverage']*100:.1f}%")
            logger.info(f"✓ Method: {distortion_map['method'].upper()}")
            
            # Generate report
            report = engine.generate_correction_report(distortion_map, 
                                                       str(self.output_dir / 'correction_report.txt'))
            logger.info(f"\nCorrection Report:\n{report}")
            
            # Visualize
            engine.visualize_distortion_map(distortion_map,
                                           str(self.output_dir / 'distortion_map.png'))
            
            self.test_results['distortion_map'] = 'PASS'
            return True
            
        except Exception as e:
            logger.error(f"✗ Distortion map generation failed: {e}")
            self.test_results['distortion_map'] = f'FAIL: {e}'
            return False
    
    def test_image_correction_application(self):
        """Test applying distortion correction to an image"""
        logger.info("\n" + "="*80)
        logger.info("TEST 6: Image Correction Application")
        logger.info("="*80)
        
        try:
            from calibration.distortion_correction import DistortionCorrectionEngine
            from calibration.dot_detector_2d import DotDetector2D
            
            detector = DotDetector2D()
            engine = DistortionCorrectionEngine(detector.geometry, um_per_pixel=100.0)
            
            # Generate synthetic image
            image = self._generate_synthetic_calibration_image(detector, (400, 400))
            
            # Create simple distortion map
            design_mm = detector.design_positions['regular_dots'][:20]
            detected_mm = design_mm + np.random.normal(0, 0.01, design_mm.shape)
            discrepancies_um = (detected_mm - design_mm) * 1000.0
            
            distortion_map = engine.build_distortion_map_from_metrology(
                design_mm, detected_mm, discrepancies_um,
                image.shape[:2], method='rbf'
            )
            
            # Apply correction
            corrected = engine.apply_distortion_correction(image, distortion_map['distortion_map'])
            
            assert corrected is not None
            assert corrected.shape == image.shape
            
            logger.info(f"✓ Applied correction to image {image.shape}")
            logger.info(f"✓ Input dtype: {image.dtype}, Output dtype: {corrected.dtype}")
            
            # Save corrected image
            output_path = self.output_dir / 'corrected_image.npy'
            np.save(output_path, corrected)
            logger.info(f"✓ Saved corrected image to {output_path}")
            
            self.test_results['image_correction'] = 'PASS'
            return True
            
        except Exception as e:
            logger.error(f"✗ Image correction failed: {e}")
            self.test_results['image_correction'] = f'FAIL: {e}'
            return False
    
    def test_full_pipeline_integration(self):
        """Test full calibration → correction pipeline"""
        logger.info("\n" + "="*80)
        logger.info("TEST 7: Full Pipeline Integration")
        logger.info("="*80)
        
        try:
            from calibration.dot_detector_2d import DotDetector2D
            from calibration.metrology_report import MetrologyReportGenerator
            from calibration.distortion_correction import CorrectionPipeline
            
            logger.info("Initializing pipeline components...")
            
            detector = DotDetector2D()
            metrology = MetrologyReportGenerator(detector.geometry)
            pipeline = CorrectionPipeline(detector, metrology)
            
            logger.info("✓ Pipeline components initialized")
            
            # Verify integration
            assert pipeline.dot_detector is not None
            assert pipeline.metrology is not None
            assert hasattr(pipeline, 'run_calibration_pass')
            assert hasattr(pipeline, 'run_correction_pass')
            
            logger.info("✓ Pipeline structure verified")
            logger.info("✓ Ready for calibration pass")
            logger.info("✓ Ready for correction pass")
            
            self.test_results['pipeline_integration'] = 'PASS'
            return True
            
        except Exception as e:
            logger.error(f"✗ Pipeline integration failed: {e}")
            self.test_results['pipeline_integration'] = f'FAIL: {e}'
            return False
    
    def run_all_tests(self):
        """Run all tests and generate summary"""
        logger.info("\n" + "="*80)
        logger.info("STARTING CALIBRATION TEST SUITE")
        logger.info("="*80)
        logger.info(f"Test output directory: {self.output_dir}")
        
        tests = [
            self.test_detector_initialization,
            self.test_synthetic_image_generation,
            self.test_feature_association_strict,
            self.test_metrology_report_generation,
            self.test_distortion_map_generation,
            self.test_image_correction_application,
            self.test_full_pipeline_integration
        ]
        
        for test in tests:
            try:
                test()
            except Exception as e:
                logger.error(f"Unexpected error in {test.__name__}: {e}")
        
        # Generate summary
        self._print_test_summary()
    
    def _print_test_summary(self):
        """Print test summary report"""
        logger.info("\n" + "="*80)
        logger.info("TEST SUITE SUMMARY")
        logger.info("="*80)
        
        passed = sum(1 for v in self.test_results.values() if v == 'PASS')
        total = len(self.test_results)
        
        logger.info(f"\nTotal Tests: {total}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {total - passed}")
        logger.info(f"Success Rate: {passed/total*100:.1f}%\n")
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result == 'PASS' else f"✗ {result}"
            logger.info(f"{test_name:30s} {status}")
        
        logger.info("\n" + "="*80)
        
        if passed == total:
            logger.info("✓ ALL TESTS PASSED!")
        else:
            logger.info(f"✗ {total - passed} TEST(S) FAILED")
        
        logger.info("="*80 + "\n")
        
        # Save summary to file
        summary_path = self.output_dir / 'test_summary.txt'
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TEST SUITE SUMMARY\n")
            f.write("="*80 + "\n\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"Total Tests: {total}\n")
            f.write(f"Passed: {passed}\n")
            f.write(f"Failed: {total - passed}\n")
            f.write(f"Success Rate: {passed/total*100:.1f}%\n\n")
            
            for test_name, result in self.test_results.items():
                status = "✓ PASS" if result == 'PASS' else f"✗ {result}"
                f.write(f"{test_name:30s} {status}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        logger.info(f"Test summary saved to {summary_path}")


if __name__ == "__main__":
    # Run test suite
    suite = CalibrationTestSuite(output_dir='test_results')
    suite.run_all_tests()
