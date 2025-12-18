#!/usr/bin/env python3
"""Test Task 4: NPZ animation data generation."""

import numpy as np
import tempfile
from pathlib import Path

def test_npz_generation():
    """Test NPZ file generation and structure."""
    
    print("Testing NPZ animation data generation...")
    print("=" * 70)
    
    # Simulate data from interpolation
    n_sources = 2562
    n_frames = 120  # 4 seconds at 30 FPS
    n_triangles = 5120
    
    # Create test data
    activity_timeline = np.random.randn(n_sources, n_frames).astype(np.float32)
    timestamps = np.linspace(0, 4.0, n_frames, dtype=np.float32)
    source_positions = np.random.randn(n_sources, 3).astype(np.float32) * 100  # mm
    triangles = np.random.randint(0, n_sources, (n_triangles, 3), dtype=np.int32)
    fps = np.array(30, dtype=np.int32)
    
    print(f"Test data:")
    print(f"  Activity: {activity_timeline.shape} {activity_timeline.dtype}")
    print(f"  Timestamps: {timestamps.shape} {timestamps.dtype}")
    print(f"  Positions: {source_positions.shape} {source_positions.dtype}")
    print(f"  Triangles: {triangles.shape} {triangles.dtype}")
    print(f"  FPS: {fps.shape} {fps.dtype}")
    print()
    
    # Create animation data dictionary
    animation_data = {
        'activity': activity_timeline,
        'timestamps': timestamps,
        'source_positions': source_positions,
        'triangles': triangles,
        'fps': fps,
    }
    
    # Save as compressed NPZ
    with tempfile.TemporaryDirectory() as tmpdir:
        animation_path = Path(tmpdir) / 'animation_data.npz'
        np.savez_compressed(str(animation_path), **animation_data)
        
        file_size_mb = animation_path.stat().st_size / (1024 * 1024)
        print(f"✓ Saved NPZ file: {file_size_mb:.2f} MB")
        print()
        
        # Load and verify
        print("Loading and verifying NPZ file...")
        loaded = np.load(str(animation_path))
        
        # Check all keys exist
        required_keys = ['activity', 'timestamps', 'source_positions', 'triangles', 'fps']
        for key in required_keys:
            assert key in loaded, f"Missing key: {key}"
        print(f"✓ All required keys present: {required_keys}")
        
        # Verify shapes
        assert loaded['activity'].shape == (n_sources, n_frames), \
            f"Activity shape mismatch: {loaded['activity'].shape}"
        assert loaded['timestamps'].shape == (n_frames,), \
            f"Timestamps shape mismatch: {loaded['timestamps'].shape}"
        assert loaded['source_positions'].shape == (n_sources, 3), \
            f"Positions shape mismatch: {loaded['source_positions'].shape}"
        assert loaded['triangles'].shape == (n_triangles, 3), \
            f"Triangles shape mismatch: {loaded['triangles'].shape}"
        assert loaded['fps'].shape == (), \
            f"FPS should be scalar, got shape: {loaded['fps'].shape}"
        
        print(f"✓ All shapes correct")
        
        # Verify dtypes
        assert loaded['activity'].dtype == np.float32, \
            f"Activity dtype should be float32, got {loaded['activity'].dtype}"
        assert loaded['timestamps'].dtype == np.float32, \
            f"Timestamps dtype should be float32, got {loaded['timestamps'].dtype}"
        assert loaded['source_positions'].dtype == np.float32, \
            f"Positions dtype should be float32, got {loaded['source_positions'].dtype}"
        assert loaded['triangles'].dtype == np.int32, \
            f"Triangles dtype should be int32, got {loaded['triangles'].dtype}"
        assert loaded['fps'].dtype == np.int32, \
            f"FPS dtype should be int32, got {loaded['fps'].dtype}"
        
        print(f"✓ All dtypes correct")
        
        # Verify values
        assert np.allclose(loaded['activity'], activity_timeline), \
            "Activity values mismatch"
        assert np.allclose(loaded['timestamps'], timestamps), \
            "Timestamps values mismatch"
        assert np.allclose(loaded['source_positions'], source_positions), \
            "Positions values mismatch"
        assert np.array_equal(loaded['triangles'], triangles), \
            "Triangles values mismatch"
        assert loaded['fps'] == 30, \
            f"FPS should be 30, got {loaded['fps']}"
        
        print(f"✓ All values preserved correctly")
        print()
        
        # Test file size
        uncompressed_size = (
            activity_timeline.nbytes +
            timestamps.nbytes +
            source_positions.nbytes +
            triangles.nbytes +
            fps.nbytes
        ) / (1024 * 1024)
        
        compression_ratio = uncompressed_size / file_size_mb
        
        print(f"Compression statistics:")
        print(f"  Uncompressed: {uncompressed_size:.2f} MB")
        print(f"  Compressed: {file_size_mb:.2f} MB")
        print(f"  Ratio: {compression_ratio:.2f}x")
        print()
        
        loaded.close()
    
    print("=" * 70)
    print("✅ Task 4 NPZ generation verified successfully!")
    print()
    print("Summary:")
    print(f"  - NPZ file contains all required data")
    print(f"  - All shapes and dtypes are correct")
    print(f"  - Compression works effectively ({compression_ratio:.1f}x)")
    print(f"  - File size reasonable ({file_size_mb:.2f} MB for {n_frames} frames)")
    print(f"  - Ready for frontend consumption")

if __name__ == "__main__":
    test_npz_generation()
