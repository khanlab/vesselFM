"""Tests for Dask-based chunking functionality."""

import unittest
import numpy as np
import torch
from vesselfm.seg.utils.dask_patching import (
    compute_patch_coords,
    extract_patch,
    compute_gaussian_importance_map,
    stitch_patches,
    process_image_with_dask_chunks
)


class TestPatchCoordinates(unittest.TestCase):
    """Test patch coordinate computation."""
    
    def test_compute_patch_coords_no_overlap(self):
        """Test patch coordinate computation without overlap."""
        image_shape = (1, 1, 64, 64, 64)
        patch_size = (32, 32, 32)
        overlap = 0.0
        
        coords = compute_patch_coords(image_shape, patch_size, overlap)
        
        # With no overlap and image size 64, patch size 32, we expect 2^3 = 8 patches
        self.assertEqual(len(coords), 8)
        
        # Check that first patch starts at origin
        self.assertIn((0, 32, 0, 32, 0, 32), coords)
    
    def test_compute_patch_coords_with_overlap(self):
        """Test patch coordinate computation with overlap."""
        image_shape = (1, 1, 64, 64, 64)
        patch_size = (32, 32, 32)
        overlap = 0.5
        
        coords = compute_patch_coords(image_shape, patch_size, overlap)
        
        # With 50% overlap, we expect more patches
        self.assertGreater(len(coords), 8)
        
        # Check that coordinates are within bounds
        for coord in coords:
            start_d, end_d, start_h, end_h, start_w, end_w = coord
            self.assertGreaterEqual(start_d, 0)
            self.assertLessEqual(end_d, 64)
            self.assertGreaterEqual(start_h, 0)
            self.assertLessEqual(end_h, 64)
            self.assertGreaterEqual(start_w, 0)
            self.assertLessEqual(end_w, 64)
    
    def test_compute_patch_coords_small_image(self):
        """Test patch coordinate computation when image is smaller than patch."""
        image_shape = (1, 1, 16, 16, 16)
        patch_size = (32, 32, 32)
        overlap = 0.0
        
        coords = compute_patch_coords(image_shape, patch_size, overlap)
        
        # Should have exactly 1 patch
        self.assertEqual(len(coords), 1)
        
        # Patch should cover the entire image
        self.assertEqual(coords[0], (0, 16, 0, 16, 0, 16))


class TestPatchExtraction(unittest.TestCase):
    """Test patch extraction."""
    
    def test_extract_patch(self):
        """Test extracting a patch from an image."""
        image = torch.randn(1, 1, 64, 64, 64)
        coords = (0, 32, 0, 32, 0, 32)
        
        patch = extract_patch(image, coords)
        
        self.assertEqual(patch.shape, (1, 1, 32, 32, 32))
        
        # Verify that extracted data matches
        self.assertTrue(torch.allclose(patch, image[:, :, 0:32, 0:32, 0:32]))
    
    def test_extract_patch_non_aligned(self):
        """Test extracting a patch with non-aligned coordinates."""
        image = torch.randn(1, 1, 64, 64, 64)
        coords = (10, 42, 10, 42, 10, 42)
        
        patch = extract_patch(image, coords)
        
        self.assertEqual(patch.shape, (1, 1, 32, 32, 32))


class TestGaussianImportanceMap(unittest.TestCase):
    """Test Gaussian importance map computation."""
    
    def test_compute_gaussian_importance_map(self):
        """Test Gaussian importance map generation."""
        patch_size = (32, 32, 32)
        sigma_scale = 0.125
        
        importance_map = compute_gaussian_importance_map(patch_size, sigma_scale)
        
        # Check shape
        self.assertEqual(importance_map.shape, (32, 32, 32))
        
        # Check that center has highest value
        center_value = importance_map[16, 16, 16]
        corner_value = importance_map[0, 0, 0]
        self.assertGreater(center_value, corner_value)
        
        # Check that values are positive
        self.assertTrue(np.all(importance_map > 0))
    
    def test_gaussian_symmetry(self):
        """Test that Gaussian importance map is symmetric."""
        patch_size = (32, 32, 32)
        importance_map = compute_gaussian_importance_map(patch_size)
        
        # Check symmetry
        self.assertAlmostEqual(
            importance_map[0, 16, 16],
            importance_map[31, 16, 16],
            places=5
        )


class TestPatchStitching(unittest.TestCase):
    """Test patch stitching."""
    
    def test_stitch_patches_simple(self):
        """Test stitching patches without overlap."""
        image_shape = (1, 1, 64, 64, 64)
        patch_size = (32, 32, 32)
        
        # Create dummy patch results
        patch_results = [
            {
                'logits': torch.ones(1, 1, 32, 32, 32),
                'coords': (0, 32, 0, 32, 0, 32)
            },
            {
                'logits': torch.ones(1, 1, 32, 32, 32) * 2,
                'coords': (32, 64, 0, 32, 0, 32)
            },
        ]
        
        output = stitch_patches(patch_results, image_shape, patch_size)
        
        # Check output shape
        self.assertEqual(output.shape, image_shape)
        
        # Check that output is not all zeros
        self.assertTrue(torch.any(output > 0))
    
    def test_stitch_patches_with_overlap(self):
        """Test stitching overlapping patches."""
        image_shape = (1, 1, 64, 64, 64)
        patch_size = (32, 32, 32)
        
        # Create overlapping patch results
        patch_results = [
            {
                'logits': torch.ones(1, 1, 32, 32, 32),
                'coords': (0, 32, 0, 32, 0, 32)
            },
            {
                'logits': torch.ones(1, 1, 32, 32, 32),
                'coords': (16, 48, 0, 32, 0, 32)
            },
        ]
        
        output = stitch_patches(patch_results, image_shape, patch_size)
        
        # Check output shape
        self.assertEqual(output.shape, image_shape)
        
        # Overlapping regions should be blended
        self.assertTrue(torch.any(output > 0))


class MockModel(torch.nn.Module):
    """Mock model for testing."""
    
    def forward(self, x):
        # Simple passthrough that maintains shape
        return x


class TestDaskChunking(unittest.TestCase):
    """Test end-to-end Dask chunking."""
    
    def test_process_image_without_dask(self):
        """Test processing image without Dask."""
        image = torch.randn(1, 1, 64, 64, 64)
        model = MockModel()
        device = "cpu"
        patch_size = (32, 32, 32)
        overlap = 0.25
        batch_size = 4
        
        output = process_image_with_dask_chunks(
            image=image,
            model=model,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
            use_dask=False
        )
        
        # Check output shape matches input
        self.assertEqual(output.shape, image.shape)
    
    def test_process_image_with_dask(self):
        """Test processing image with Dask."""
        image = torch.randn(1, 1, 64, 64, 64)
        model = MockModel()
        device = "cpu"
        patch_size = (32, 32, 32)
        overlap = 0.25
        batch_size = 4
        
        output = process_image_with_dask_chunks(
            image=image,
            model=model,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
            use_dask=True,
            n_workers=2
        )
        
        # Check output shape matches input
        self.assertEqual(output.shape, image.shape)
    
    def test_dask_vs_sequential_consistency(self):
        """Test that Dask and sequential processing give similar results."""
        torch.manual_seed(42)
        image = torch.randn(1, 1, 64, 64, 64)
        model = MockModel()
        device = "cpu"
        patch_size = (32, 32, 32)
        overlap = 0.5
        batch_size = 4
        
        # Process with Dask
        output_dask = process_image_with_dask_chunks(
            image=image,
            model=model,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
            use_dask=True,
            n_workers=2
        )
        
        # Process sequentially
        output_sequential = process_image_with_dask_chunks(
            image=image,
            model=model,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
            use_dask=False
        )
        
        # Results should be very close (allowing for numerical differences)
        self.assertTrue(torch.allclose(output_dask, output_sequential, rtol=1e-4, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
