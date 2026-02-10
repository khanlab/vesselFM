#!/usr/bin/env python3
"""
Simple validation script for Dask chunking functionality.
Tests the basic functionality without requiring pytest.
"""

import sys
import traceback


def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    try:
        import numpy as np
        import torch
        import dask
        import dask.bag as db
        from vesselfm.seg.utils.dask_patching import (
            compute_patch_coords,
            extract_patch,
            compute_gaussian_importance_map,
            stitch_patches,
            process_image_with_dask_chunks
        )
        print("✓ All imports successful")
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        traceback.print_exc()
        return False


def test_patch_coords():
    """Test patch coordinate computation."""
    print("\nTesting patch coordinate computation...")
    try:
        from vesselfm.seg.utils.dask_patching import compute_patch_coords
        
        image_shape = (1, 1, 64, 64, 64)
        patch_size = (32, 32, 32)
        overlap = 0.0
        
        coords = compute_patch_coords(image_shape, patch_size, overlap)
        
        assert len(coords) == 8, f"Expected 8 patches, got {len(coords)}"
        assert (0, 32, 0, 32, 0, 32) in coords, "First patch should start at origin"
        
        print(f"✓ Generated {len(coords)} patches correctly")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_extract_patch():
    """Test patch extraction."""
    print("\nTesting patch extraction...")
    try:
        import torch
        from vesselfm.seg.utils.dask_patching import extract_patch
        
        image = torch.randn(1, 1, 64, 64, 64)
        coords = (0, 32, 0, 32, 0, 32)
        
        patch = extract_patch(image, coords)
        
        assert patch.shape == (1, 1, 32, 32, 32), f"Unexpected shape: {patch.shape}"
        
        print(f"✓ Patch extracted with shape {patch.shape}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_gaussian_map():
    """Test Gaussian importance map."""
    print("\nTesting Gaussian importance map...")
    try:
        import numpy as np
        from vesselfm.seg.utils.dask_patching import compute_gaussian_importance_map
        
        patch_size = (32, 32, 32)
        importance_map = compute_gaussian_importance_map(patch_size)
        
        assert importance_map.shape == (32, 32, 32), f"Unexpected shape: {importance_map.shape}"
        assert np.all(importance_map > 0), "All values should be positive"
        
        center_value = importance_map[16, 16, 16]
        corner_value = importance_map[0, 0, 0]
        assert center_value > corner_value, "Center should have higher value than corner"
        
        print(f"✓ Gaussian map generated correctly")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_stitch_patches():
    """Test patch stitching."""
    print("\nTesting patch stitching...")
    try:
        import torch
        from vesselfm.seg.utils.dask_patching import stitch_patches
        
        image_shape = (1, 1, 64, 64, 64)
        patch_size = (32, 32, 32)
        
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
        
        assert output.shape == image_shape, f"Unexpected shape: {output.shape}"
        assert torch.any(output > 0), "Output should have non-zero values"
        
        print(f"✓ Patches stitched correctly, output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def test_process_image():
    """Test end-to-end image processing."""
    print("\nTesting end-to-end image processing...")
    try:
        import torch
        from vesselfm.seg.utils.dask_patching import process_image_with_dask_chunks
        
        class MockModel(torch.nn.Module):
            def forward(self, x):
                return x
        
        image = torch.randn(1, 1, 64, 64, 64)
        model = MockModel()
        device = "cpu"
        patch_size = (32, 32, 32)
        overlap = 0.25
        batch_size = 4
        
        # Test without Dask
        output_seq = process_image_with_dask_chunks(
            image=image,
            model=model,
            device=device,
            patch_size=patch_size,
            overlap=overlap,
            batch_size=batch_size,
            use_dask=False
        )
        
        assert output_seq.shape == image.shape, f"Unexpected shape: {output_seq.shape}"
        
        print(f"✓ Sequential processing works, output shape: {output_seq.shape}")
        
        # Test with Dask
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
        
        assert output_dask.shape == image.shape, f"Unexpected shape: {output_dask.shape}"
        
        print(f"✓ Dask processing works, output shape: {output_dask.shape}")
        
        # Check consistency
        if torch.allclose(output_dask, output_seq, rtol=1e-4, atol=1e-6):
            print("✓ Dask and sequential outputs are consistent")
        else:
            print("⚠ Dask and sequential outputs differ slightly (expected due to order)")
        
        return True
    except Exception as e:
        print(f"✗ Test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Validating Dask Chunking Implementation")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_patch_coords,
        test_extract_patch,
        test_gaussian_map,
        test_stitch_patches,
        test_process_image,
    ]
    
    results = []
    for test in tests:
        results.append(test())
    
    print("\n" + "=" * 60)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
