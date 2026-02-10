#!/usr/bin/env python3
"""
Test CLI argument parsing for Dask chunking options.
"""

from argparse import Namespace
from pathlib import Path


def test_default_config():
    """Test that default config includes chunk_images setting."""
    from vesselfm.cli import create_config
    
    args = Namespace(
        input_folder=Path("/test/input"),
        output_folder=Path("/test/output"),
        mask_folder=None,
        device="cpu",
        batch_size=None,
        patch_size=None,
        overlap=None,
        threshold=None,
        tta_scales=None,
        apply_postprocessing=False,
        disable_dask=False,
        dask_workers=None,
        dask_threads_per_worker=None,
        enable_dask_chunking=False,
        disable_dask_chunking=False
    )
    
    cfg = create_config(args)
    
    print("Default config created successfully")
    print(f"  dask.enabled: {cfg.dask.enabled}")
    print(f"  dask.chunk_images: {cfg.dask.chunk_images}")
    print(f"  dask.n_workers: {cfg.dask.n_workers}")
    
    assert cfg.dask.enabled == True, "Dask should be enabled by default"
    assert cfg.dask.chunk_images == True, "Chunk images should be enabled by default"
    print("✓ Default config test passed\n")


def test_enable_dask_chunking():
    """Test enabling Dask chunking via CLI."""
    from vesselfm.cli import create_config
    
    args = Namespace(
        input_folder=Path("/test/input"),
        output_folder=Path("/test/output"),
        mask_folder=None,
        device="cpu",
        batch_size=None,
        patch_size=None,
        overlap=None,
        threshold=None,
        tta_scales=None,
        apply_postprocessing=False,
        disable_dask=False,
        dask_workers=None,
        dask_threads_per_worker=None,
        enable_dask_chunking=True,
        disable_dask_chunking=False
    )
    
    cfg = create_config(args)
    
    print("Config with --enable-dask-chunking:")
    print(f"  dask.enabled: {cfg.dask.enabled}")
    print(f"  dask.chunk_images: {cfg.dask.chunk_images}")
    
    assert cfg.dask.chunk_images == True, "Chunk images should be enabled"
    print("✓ Enable dask chunking test passed\n")


def test_disable_dask_chunking():
    """Test disabling Dask chunking via CLI."""
    from vesselfm.cli import create_config
    
    args = Namespace(
        input_folder=Path("/test/input"),
        output_folder=Path("/test/output"),
        mask_folder=None,
        device="cpu",
        batch_size=None,
        patch_size=None,
        overlap=None,
        threshold=None,
        tta_scales=None,
        apply_postprocessing=False,
        disable_dask=False,
        dask_workers=None,
        dask_threads_per_worker=None,
        enable_dask_chunking=False,
        disable_dask_chunking=True
    )
    
    cfg = create_config(args)
    
    print("Config with --disable-dask-chunking:")
    print(f"  dask.enabled: {cfg.dask.enabled}")
    print(f"  dask.chunk_images: {cfg.dask.chunk_images}")
    
    assert cfg.dask.chunk_images == False, "Chunk images should be disabled"
    print("✓ Disable dask chunking test passed\n")


def test_disable_dask_completely():
    """Test disabling Dask completely via CLI."""
    from vesselfm.cli import create_config
    
    args = Namespace(
        input_folder=Path("/test/input"),
        output_folder=Path("/test/output"),
        mask_folder=None,
        device="cpu",
        batch_size=None,
        patch_size=None,
        overlap=None,
        threshold=None,
        tta_scales=None,
        apply_postprocessing=False,
        disable_dask=True,
        dask_workers=None,
        dask_threads_per_worker=None,
        enable_dask_chunking=False,
        disable_dask_chunking=False
    )
    
    cfg = create_config(args)
    
    print("Config with --disable-dask:")
    print(f"  dask.enabled: {cfg.dask.enabled}")
    print(f"  dask.chunk_images: {cfg.dask.chunk_images}")
    
    assert cfg.dask.enabled == False, "Dask should be disabled"
    print("✓ Disable dask completely test passed\n")


def test_dask_workers():
    """Test setting number of Dask workers via CLI."""
    from vesselfm.cli import create_config
    
    args = Namespace(
        input_folder=Path("/test/input"),
        output_folder=Path("/test/output"),
        mask_folder=None,
        device="cpu",
        batch_size=None,
        patch_size=None,
        overlap=None,
        threshold=None,
        tta_scales=None,
        apply_postprocessing=False,
        disable_dask=False,
        dask_workers=8,
        dask_threads_per_worker=None,
        enable_dask_chunking=False,
        disable_dask_chunking=False
    )
    
    cfg = create_config(args)
    
    print("Config with --dask-workers 8:")
    print(f"  dask.n_workers: {cfg.dask.n_workers}")
    
    assert cfg.dask.n_workers == 8, "Should have 8 workers"
    print("✓ Dask workers test passed\n")


if __name__ == "__main__":
    print("=" * 60)
    print("Testing CLI Configuration for Dask Chunking")
    print("=" * 60 + "\n")
    
    try:
        test_default_config()
        test_enable_dask_chunking()
        test_disable_dask_chunking()
        test_disable_dask_completely()
        test_dask_workers()
        
        print("=" * 60)
        print("✓ All CLI configuration tests passed!")
        print("=" * 60)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
