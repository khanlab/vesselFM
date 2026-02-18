"""Tests for multi-GPU processing functionality."""

import pytest
import numpy as np
import torch
import torch.nn as nn

from vesselfm.seg.utils.dask_patching import (
    get_available_gpus,
    setup_gpu_workers,
    process_patch_on_gpu,
    process_image_with_dask_chunks,
    extract_patch,
    compute_patch_coords,
)


class DummyModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(1, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class TestMultiGPUFunctionality:
    """Test suite for multi-GPU processing."""
    
    def test_get_available_gpus(self):
        """Test GPU detection."""
        gpus = get_available_gpus()
        
        if torch.cuda.is_available():
            assert len(gpus) > 0
            assert all(gpu.startswith("cuda:") for gpu in gpus)
        else:
            assert len(gpus) == 0
    
    def test_setup_gpu_workers_no_gpus(self):
        """Test setup when no GPUs are available."""
        if torch.cuda.is_available():
            pytest.skip("GPUs are available, skipping this test")
        
        devices, client = setup_gpu_workers()
        assert devices == ["cpu"]
        assert client is None
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_setup_gpu_workers_with_gpus(self):
        """Test setup with available GPUs."""
        devices, client = setup_gpu_workers()
        
        assert len(devices) > 0
        assert all(device.startswith("cuda:") for device in devices)
        
        # Clean up client if created
        if client is not None:
            client.close()
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_setup_gpu_workers_specific_ids(self):
        """Test setup with specific GPU IDs."""
        n_gpus = torch.cuda.device_count()
        
        if n_gpus >= 2:
            # Test with specific GPU IDs
            devices, client = setup_gpu_workers(gpu_ids=[0, 1])
            assert "cuda:0" in devices
            assert "cuda:1" in devices
            
            if client is not None:
                client.close()
        else:
            # Test with single GPU
            devices, client = setup_gpu_workers(gpu_ids=[0])
            assert devices == ["cuda:0"]
            
            if client is not None:
                client.close()
    
    def test_process_patch_on_gpu_cpu(self):
        """Test processing a patch on CPU."""
        # Create a small test image
        image = torch.randn(1, 1, 20, 20, 20)
        coords = (0, 10, 0, 10, 0, 10)
        
        # Create a simple model
        model = DummyModel()
        model.eval()
        
        # Process patch
        result = process_patch_on_gpu(coords, image, model, "cpu")
        
        assert "logits" in result
        assert "coords" in result
        assert result["coords"] == coords
        assert result["logits"].shape == (10, 10, 10)
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_process_patch_on_gpu_cuda(self):
        """Test processing a patch on GPU."""
        # Create a small test image
        image = torch.randn(1, 1, 20, 20, 20)
        coords = (0, 10, 0, 10, 0, 10)
        
        # Create a simple model
        model = DummyModel()
        model.eval()
        
        # Process patch on GPU
        result = process_patch_on_gpu(coords, image, model, "cuda:0")
        
        assert "logits" in result
        assert "coords" in result
        assert result["coords"] == coords
        assert result["logits"].shape == (10, 10, 10)
        # Result should be on CPU after processing
        assert result["logits"].device == torch.device("cpu")
    
    def test_process_image_single_gpu(self):
        """Test processing an image with single GPU or CPU."""
        # Create a small test image
        image = torch.randn(1, 1, 32, 32, 32)
        
        # Create a simple model
        model = DummyModel()
        model.eval()
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Process with Dask disabled (sequential)
        output = process_image_with_dask_chunks(
            image=image,
            model=model,
            device=device,
            patch_size=(16, 16, 16),
            overlap=0.5,
            batch_size=2,
            use_dask=False,
            use_multi_gpu=False
        )
        
        assert output.shape == image.shape
        assert output.dtype == torch.float32
    
    def test_process_image_with_dask(self):
        """Test processing an image with Dask enabled."""
        # Create a small test image
        image = torch.randn(1, 1, 32, 32, 32)
        
        # Create a simple model
        model = DummyModel()
        model.eval()
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Process with Dask enabled
        output = process_image_with_dask_chunks(
            image=image,
            model=model,
            device=device,
            patch_size=(16, 16, 16),
            overlap=0.5,
            batch_size=2,
            use_dask=True,
            use_multi_gpu=False
        )
        
        assert output.shape == image.shape
        assert output.dtype == torch.float32
    
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs not available")
    def test_process_image_multi_gpu(self):
        """Test processing an image with multiple GPUs."""
        # Create a test image
        image = torch.randn(1, 1, 64, 64, 64)
        
        # Create a simple model
        model = DummyModel()
        model.eval()
        
        # Process with multi-GPU enabled
        output = process_image_with_dask_chunks(
            image=image,
            model=model,
            device="cuda:0",
            patch_size=(32, 32, 32),
            overlap=0.5,
            batch_size=1,
            use_dask=True,
            use_multi_gpu=True,
            gpu_ids=[0, 1]
        )
        
        assert output.shape == image.shape
        assert output.dtype == torch.float32
    
    def test_multi_gpu_fallback_to_single(self):
        """Test that multi-GPU falls back to single GPU when only one is available."""
        # Create a small test image
        image = torch.randn(1, 1, 32, 32, 32)
        
        # Create a simple model
        model = DummyModel()
        model.eval()
        
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
        # Try to enable multi-GPU (should fall back if only 1 GPU available)
        output = process_image_with_dask_chunks(
            image=image,
            model=model,
            device=device,
            patch_size=(16, 16, 16),
            overlap=0.5,
            batch_size=2,
            use_dask=True,
            use_multi_gpu=True,
            gpu_ids=None  # Use all available
        )
        
        assert output.shape == image.shape
        assert output.dtype == torch.float32


class TestMultiGPUConsistency:
    """Test that multi-GPU produces consistent results."""
    
    @pytest.mark.skipif(torch.cuda.device_count() < 2, reason="Multiple GPUs not available")
    def test_multi_gpu_vs_single_gpu_consistency(self):
        """Test that multi-GPU produces the same results as single GPU."""
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        # Create a test image
        image = torch.randn(1, 1, 64, 64, 64)
        
        # Create a simple model with fixed weights
        model = DummyModel()
        model.eval()
        
        # Process with single GPU
        torch.manual_seed(42)
        output_single = process_image_with_dask_chunks(
            image=image,
            model=model,
            device="cuda:0",
            patch_size=(32, 32, 32),
            overlap=0.5,
            batch_size=1,
            use_dask=True,
            use_multi_gpu=False
        )
        
        # Process with multi-GPU
        torch.manual_seed(42)
        output_multi = process_image_with_dask_chunks(
            image=image,
            model=model,
            device="cuda:0",
            patch_size=(32, 32, 32),
            overlap=0.5,
            batch_size=1,
            use_dask=True,
            use_multi_gpu=True,
            gpu_ids=[0, 1]
        )
        
        # Results should be very close (small numerical differences are acceptable)
        torch.testing.assert_close(output_single, output_multi, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
