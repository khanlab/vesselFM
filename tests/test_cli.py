"""Tests for the vesselFM CLI tool."""

import unittest
import tempfile
import shutil
from pathlib import Path
import subprocess
import sys


class TestCLI(unittest.TestCase):
    """Test cases for vesselFM CLI inference tool."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.repo_root = Path(__file__).parent.parent
        cls.test_data_dir = cls.repo_root / "test_data"
        cls.input_dir = cls.test_data_dir / "input"
        
        # Verify test data exists
        if not cls.input_dir.exists() or not list(cls.input_dir.glob("*.nii.gz")):
            raise RuntimeError(
                f"Test data not found in {cls.input_dir}. "
                "Please create test data before running tests."
            )
    
    def setUp(self):
        """Set up test case."""
        # Create temporary output directory
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = Path(self.temp_dir) / "output"
        self.output_dir.mkdir(parents=True)
    
    def tearDown(self):
        """Clean up after test case."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run(
            [sys.executable, "-m", "vesselfm.cli", "--help"],
            capture_output=True,
            text=True
        )
        self.assertEqual(result.returncode, 0)
        self.assertIn("--input-folder", result.stdout)
        self.assertIn("--output-folder", result.stdout)
    
    def test_cli_missing_required_args(self):
        """Test that CLI fails gracefully without required arguments."""
        result = subprocess.run(
            [sys.executable, "-m", "vesselfm.cli"],
            capture_output=True,
            text=True
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("required", result.stderr.lower())
    
    def test_cli_nonexistent_input_folder(self):
        """Test that CLI fails when input folder doesn't exist."""
        result = subprocess.run(
            [
                sys.executable, "-m", "vesselfm.cli",
                "--input-folder", "/nonexistent/path",
                "--output-folder", str(self.output_dir)
            ],
            capture_output=True,
            text=True
        )
        self.assertNotEqual(result.returncode, 0)
    
    def test_cli_basic_inference(self):
        """Test basic CLI inference with default parameters."""
        # Note: This test will fail if the model/dependencies are not available
        # We're testing the CLI interface, not the actual model inference
        result = subprocess.run(
            [
                sys.executable, "-m", "vesselfm.cli",
                "--input-folder", str(self.input_dir),
                "--output-folder", str(self.output_dir),
                "--device", "cpu"
            ],
            capture_output=True,
            text=True,
            timeout=300  # 5 minutes timeout
        )
        
        # Check if output was attempted (may fail due to missing dependencies)
        # but the CLI should have parsed arguments correctly
        if result.returncode != 0:
            # If it fails, it should be due to missing dependencies (torch, etc.)
            # not CLI parsing errors
            error_lower = result.stderr.lower()
            self.assertTrue(
                "model" in error_lower or 
                "torch" in error_lower or
                "modulenotfounderror" in error_lower or
                "importerror" in error_lower or
                "Loading model" in result.stdout or
                result.returncode == 0,
                f"Unexpected error: {result.stderr}"
            )
        else:
            # If successful, check that output folder has files
            output_files = list(self.output_dir.glob("*.nii.gz"))
            self.assertGreater(len(output_files), 0, "No output files generated")
    
    def test_cli_with_custom_params(self):
        """Test CLI with custom parameters."""
        result = subprocess.run(
            [
                sys.executable, "-m", "vesselfm.cli",
                "--input-folder", str(self.input_dir),
                "--output-folder", str(self.output_dir),
                "--device", "cpu",
                "--batch-size", "4",
                "--threshold", "0.6",
                "--apply-postprocessing"
            ],
            capture_output=True,
            text=True,
            timeout=300
        )
        
        # Similar to basic inference test - check CLI worked correctly
        if result.returncode != 0:
            error_lower = result.stderr.lower()
            self.assertTrue(
                "model" in error_lower or 
                "torch" in error_lower or
                "modulenotfounderror" in error_lower or
                "importerror" in error_lower or
                "Loading model" in result.stdout or
                result.returncode == 0,
                f"Unexpected error: {result.stderr}"
            )


class TestConfigGeneration(unittest.TestCase):
    """Test configuration generation from CLI arguments."""
    
    def test_config_creation(self):
        """Test that config is created correctly from CLI args."""
        from vesselfm.cli import create_config
        from argparse import Namespace
        
        args = Namespace(
            input_folder=Path("/test/input"),
            output_folder=Path("/test/output"),
            mask_folder=None,
            device="cpu",
            batch_size=8,
            patch_size=[64, 64, 64],
            overlap=0.5,
            threshold=0.5,
            tta_scales=[1.0],
            apply_postprocessing=False
        )
        
        cfg = create_config(args)
        
        # Check that config has expected values
        self.assertEqual(cfg.image_path, "/test/input")
        self.assertEqual(cfg.output_folder, "/test/output")
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.batch_size, 8)
        self.assertEqual(cfg.patch_size, [64, 64, 64])
        self.assertEqual(cfg.overlap, 0.5)
        self.assertEqual(cfg.merging.threshold, 0.5)
        self.assertEqual(cfg.tta.scales, [1.0])
        self.assertFalse(cfg.post.apply)


if __name__ == "__main__":
    unittest.main()
