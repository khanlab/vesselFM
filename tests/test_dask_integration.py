"""Tests for Dask integration in vesselFM."""

import unittest
from pathlib import Path
from argparse import Namespace


def create_test_args(**overrides):
    """Helper function to create test arguments with all required fields."""
    defaults = {
        'input_folder': Path("/test/input"),
        'output_folder': Path("/test/output"),
        'mask_folder': None,
        'device': "cpu",
        'batch_size': None,
        'patch_size': None,
        'overlap': None,
        'threshold': None,
        'tta_scales': None,
        'apply_postprocessing': False,
        'disable_dask': False,
        'dask_workers': None,
        'dask_threads_per_worker': None,
        'enable_dask_chunking': False,
        'disable_dask_chunking': False,
    }
    defaults.update(overrides)
    return Namespace(**defaults)


class TestDaskConfig(unittest.TestCase):
    """Test cases for Dask configuration."""
    
    def test_dask_config_exists_in_default_config(self):
        """Test that Dask configuration is present in the default config."""
        from vesselfm.cli import create_config
        
        args = create_test_args()
        
        cfg = create_config(args)
        
        # Check that Dask configuration exists
        self.assertIn("dask", cfg)
        self.assertIn("enabled", cfg.dask)
        
    def test_dask_config_default_values(self):
        """Test that Dask configuration has correct default values."""
        from vesselfm.cli import create_config
        
        args = create_test_args()
        cfg = create_config(args)
        
        # Check default values
        self.assertTrue(cfg.dask.enabled)
        self.assertIsNone(cfg.dask.n_workers)
        self.assertEqual(cfg.dask.threads_per_worker, 2)
        self.assertEqual(cfg.dask.memory_limit, "auto")
        # Check new chunk_images default
        self.assertTrue(cfg.dask.chunk_images)
    
    def test_dask_disabled_via_cli(self):
        """Test that Dask can be disabled via CLI argument."""
        from vesselfm.cli import create_config
        
        args = create_test_args(disable_dask=True)
        
        cfg = create_config(args)
        
        # Check that Dask is disabled
        self.assertFalse(cfg.dask.enabled)
    
    def test_dask_workers_override(self):
        """Test that Dask workers can be overridden via CLI."""
        from vesselfm.cli import create_config
        
        args = create_test_args(
            dask_workers=4,
            dask_threads_per_worker=3
        )
        
        cfg = create_config(args)
        
        # Check overridden values
        self.assertEqual(cfg.dask.n_workers, 4)
        self.assertEqual(cfg.dask.threads_per_worker, 3)


class TestDaskImports(unittest.TestCase):
    """Test that Dask modules can be imported."""
    
    def test_dask_import(self):
        """Test that Dask can be imported."""
        import dask
        import dask.bag
    
    def test_distributed_import(self):
        """Test that Dask distributed can be imported."""
        from dask.diagnostics import ProgressBar


if __name__ == "__main__":
    unittest.main()
