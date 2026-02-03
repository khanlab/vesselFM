"""Tests for the dask-based inference module."""

import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np

try:
    import dask.array as da
    import torch
    from omegaconf import OmegaConf
    DEPS_AVAILABLE = True
except ImportError:
    DEPS_AVAILABLE = False


@unittest.skipIf(not DEPS_AVAILABLE, "Required dependencies (dask, torch, omegaconf) not available")
class TestDaskInference(unittest.TestCase):
    """Test cases for dask-based inference."""
    
    def setUp(self):
        """Set up test case."""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up after test case."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    def test_import_inference_dask(self):
        """Test that inference_dask module can be imported."""
        try:
            from vesselfm.seg import inference_dask
            self.assertTrue(hasattr(inference_dask, 'run_inference_dask'))
            self.assertTrue(hasattr(inference_dask, 'inference_chunk'))
        except ImportError as e:
            self.fail(f"Failed to import inference_dask: {e}")
    
    def test_inference_chunk_shape_3d(self):
        """Test that inference_chunk handles 3D input correctly."""
        from vesselfm.seg.inference_dask import inference_chunk
        from vesselfm.seg.utils.data import generate_transforms
        from monai.inferers import SlidingWindowInfererAdapt
        
        # Create a simple mock model
        class MockModel(torch.nn.Module):
            def forward(self, x):
                # Return same shape with single channel
                return torch.sigmoid(x)
        
        model = MockModel()
        model.eval()
        
        # Create mock config
        cfg = OmegaConf.create({
            'tta': {
                'scales': [1.0],
                'invert': False,
                'equalize_hist': False,
            },
            'merging': {
                'max': False,
                'threshold': 0.5,
            },
            'post': {
                'apply': False,
            }
        })
        
        # Create transforms
        transforms_config = [{'ScaleIntensityRange': {'a_min': 0, 'a_max': 1, 'b_min': 0, 'b_max': 1, 'clip': True}}]
        transforms = generate_transforms(transforms_config)
        
        # Create inferer
        inferer = SlidingWindowInfererAdapt(
            roi_size=(32, 32, 32),
            sw_batch_size=1,
            overlap=0.25,
        )
        
        # Create 3D test data
        test_chunk = np.random.rand(32, 32, 32).astype(np.float32)
        
        # Run inference
        result = inference_chunk(
            test_chunk,
            model=model,
            inferer=inferer,
            transforms=transforms,
            device='cpu',
            cfg=cfg
        )
        
        # Check output shape matches input
        self.assertEqual(result.shape, test_chunk.shape)
        self.assertEqual(result.dtype, np.uint8)
    
    def test_run_inference_dask_basic(self):
        """Test basic dask inference workflow."""
        from vesselfm.seg.inference_dask import run_inference_dask
        
        # Create a simple mock model
        class MockModel(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)
        
        model = MockModel()
        
        # Create a small dask array for testing
        test_array = da.from_array(
            np.random.rand(64, 64, 64).astype(np.float32),
            chunks=(32, 32, 32)
        )
        
        # Create minimal config
        cfg = OmegaConf.create({
            'device': 'cpu',
            'transforms_config': [
                {'ScaleIntensityRange': {'a_min': 0, 'a_max': 1, 'b_min': 0, 'b_max': 1, 'clip': True}}
            ],
            'patch_size': (16, 16, 16),
            'batch_size': 1,
            'overlap': 0.25,
            'mode': 'gaussian',
            'sigma_scale': 0.125,
            'padding_mode': 'constant',
            'tta': {
                'scales': [1.0],
                'invert': False,
                'equalize_hist': False,
            },
            'merging': {
                'max': False,
                'threshold': 0.5,
            },
            'post': {
                'apply': False,
            }
        })
        
        # Run inference (don't compute, just check graph creation)
        result = run_inference_dask(test_array, cfg, model=model)
        
        # Check that result is a dask array
        self.assertIsInstance(result, da.Array)
        
        # Check output shape matches input
        self.assertEqual(result.shape, test_array.shape)
        
        # Check dtype is uint8
        self.assertEqual(result.dtype, np.uint8)
    
    def test_run_inference_dask_compute_small(self):
        """Test that dask inference can be computed on a very small array."""
        from vesselfm.seg.inference_dask import run_inference_dask
        
        # Create a simple mock model
        class MockModel(torch.nn.Module):
            def forward(self, x):
                return torch.sigmoid(x)
        
        model = MockModel()
        
        # Create a very small dask array
        test_array = da.from_array(
            np.random.rand(16, 16, 16).astype(np.float32),
            chunks=(16, 16, 16)
        )
        
        # Create minimal config
        cfg = OmegaConf.create({
            'device': 'cpu',
            'transforms_config': [
                {'ScaleIntensityRange': {'a_min': 0, 'a_max': 1, 'b_min': 0, 'b_max': 1, 'clip': True}}
            ],
            'patch_size': (8, 8, 8),
            'batch_size': 1,
            'overlap': 0.0,
            'mode': 'constant',
            'sigma_scale': 0.125,
            'padding_mode': 'constant',
            'tta': {
                'scales': [1.0],
                'invert': False,
                'equalize_hist': False,
            },
            'merging': {
                'max': False,
                'threshold': 0.5,
            },
            'post': {
                'apply': False,
            }
        })
        
        # Run inference and compute
        result = run_inference_dask(test_array, cfg, model=model)
        computed_result = result.compute()
        
        # Check output
        self.assertEqual(computed_result.shape, test_array.shape)
        self.assertEqual(computed_result.dtype, np.uint8)
        self.assertTrue(np.all((computed_result == 0) | (computed_result == 1)))


if __name__ == "__main__":
    unittest.main()
