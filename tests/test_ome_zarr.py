"""Tests for OME.ZARR reader/writer functionality."""

import os
import tempfile
import shutil
import pytest
import numpy as np

# Try to import zarr and ome-zarr
try:
    import zarr
    import ome_zarr
    from vesselfm.seg.utils.io import OmeZarrReaderWriter
    ZARR_AVAILABLE = True
except ImportError:
    ZARR_AVAILABLE = False


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="ome-zarr package not available")
class TestOmeZarrReaderWriter:
    """Test suite for OME.ZARR I/O functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.reader_writer = OmeZarrReaderWriter()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def test_write_and_read_3d_volume(self):
        """Test writing and reading a 3D volume in OME.ZARR format."""
        # Create a small 3D volume
        volume = np.random.randint(0, 255, size=(10, 20, 30), dtype=np.uint8)
        
        # Define metadata
        metadata = {
            "spacing": [1.0, 0.5, 0.5],  # z, y, x spacing
            "origin": [0.0, 0.0, 0.0],
            "direction": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        }
        
        # Write to OME.ZARR
        output_path = os.path.join(self.temp_dir, "test_volume.zarr")
        self.reader_writer.write_seg(volume, output_path, metadata)
        
        # Verify the file was created
        assert os.path.exists(output_path)
        assert os.path.isdir(output_path)
        
        # Read back the volume
        read_volume, read_metadata = self.reader_writer.read_images(output_path)
        
        # Verify the data matches
        np.testing.assert_array_equal(read_volume, volume)
        
        # Verify metadata (spacing should be preserved)
        assert "spacing" in read_metadata
        np.testing.assert_array_almost_equal(read_metadata["spacing"], metadata["spacing"], decimal=5)
    
    def test_write_without_metadata(self):
        """Test writing without metadata (should use defaults)."""
        volume = np.random.randint(0, 255, size=(5, 10, 15), dtype=np.uint8)
        
        output_path = os.path.join(self.temp_dir, "test_no_metadata.zarr")
        self.reader_writer.write_seg(volume, output_path, metadata=None)
        
        # Should still create the file successfully
        assert os.path.exists(output_path)
        
        # Read back and verify
        read_volume, read_metadata = self.reader_writer.read_images(output_path)
        np.testing.assert_array_equal(read_volume, volume)
    
    def test_read_nonexistent_file(self):
        """Test reading a non-existent file raises appropriate error."""
        with pytest.raises((ValueError, FileNotFoundError)):
            self.reader_writer.read_images("/nonexistent/path.zarr")
    
    def test_supported_formats(self):
        """Test that the correct file formats are supported."""
        assert "zarr" in self.reader_writer.supported_file_formats
        assert "ome.zarr" in self.reader_writer.supported_file_formats
    
    def test_read_segs(self):
        """Test that read_segs works the same as read_images."""
        volume = np.random.randint(0, 2, size=(8, 16, 24), dtype=np.uint8)
        
        output_path = os.path.join(self.temp_dir, "test_seg.zarr")
        self.reader_writer.write_seg(volume, output_path)
        
        # Read as segmentation
        read_volume, _ = self.reader_writer.read_segs(output_path)
        np.testing.assert_array_equal(read_volume, volume)
    
    def test_different_data_types(self):
        """Test writing and reading different data types."""
        # Test with float32
        volume_float = np.random.rand(5, 10, 15).astype(np.float32)
        output_path = os.path.join(self.temp_dir, "test_float.zarr")
        self.reader_writer.write_seg(volume_float, output_path)
        
        read_volume, _ = self.reader_writer.read_images(output_path)
        np.testing.assert_array_almost_equal(read_volume, volume_float, decimal=5)
        
        # Test with uint16
        volume_uint16 = np.random.randint(0, 65535, size=(5, 10, 15), dtype=np.uint16)
        output_path = os.path.join(self.temp_dir, "test_uint16.zarr")
        self.reader_writer.write_seg(volume_uint16, output_path)
        
        read_volume, _ = self.reader_writer.read_images(output_path)
        np.testing.assert_array_equal(read_volume, volume_uint16)


def test_zarr_not_available():
    """Test that appropriate error is raised when ome-zarr is not available."""
    # This test will only run if zarr is not available
    if ZARR_AVAILABLE:
        pytest.skip("ome-zarr is available, skipping this test")
    else:
        from vesselfm.seg.utils.io import OmeZarrReaderWriter
        with pytest.raises(ImportError):
            OmeZarrReaderWriter()


@pytest.mark.skipif(not ZARR_AVAILABLE, reason="ome-zarr package not available")
def test_determine_reader_writer_with_zarr():
    """Test that determine_reader_writer correctly identifies zarr files."""
    from vesselfm.seg.utils.io import determine_reader_writer, OmeZarrReaderWriter
    
    # Test .zarr extension
    reader_writer_class = determine_reader_writer("zarr")
    assert reader_writer_class == OmeZarrReaderWriter
    
    # Test .ome.zarr extension
    reader_writer_class = determine_reader_writer("ome.zarr")
    assert reader_writer_class == OmeZarrReaderWriter


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
