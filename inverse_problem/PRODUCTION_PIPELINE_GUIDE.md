# Production Pipeline Usage Guide

## Quick Start

### For Frontend Visualization (Recommended)
```bash
python run_edf_inference.py <input.edf> \
  --use_global_norm \
  --smoothing_alpha 0.6 \
  --overlap_fraction 0.5
```

**Output:**
- `animation_data.npz` - Smoothed version for visualization (recommended for frontend)
- `animation_data_raw.npz` - Raw version preserved for analysis
- `<filename>_window<XXXX>_t<XXX>_interactive.html` - Best window visualization

**Parameters:**
- `--use_global_norm`: Uses 99th percentile normalization across entire recording (eliminates 6.89× artificial variations)
- `--smoothing_alpha 0.6`: Light temporal smoothing (57.8% jump reduction, minimal accuracy loss)
- `--overlap_fraction 0.5`: 50% window overlap (produces ~4 FPS output)

### For Research/Analysis
```bash
python run_edf_inference.py <input.edf> \
  --use_global_norm \
  --overlap_fraction 0.5
```

**Output:**
- `animation_data.npz` - Raw version only (no smoothing applied)
- Preserves full temporal dynamics and localization accuracy

## Parameters Reference

### Required
- `edf_path`: Path to EDF file

### Recommended
- `--use_global_norm`: Enable global normalization (recommended for all use cases)
- `--overlap_fraction 0.5`: 50% window overlap (default: 0.0)

### Optional
- `--smoothing_alpha <0.0-1.0>`: EMA smoothing strength
  - **0.7**: Very light smoothing (~40% jump reduction)
  - **0.6**: Light smoothing (~58% jump reduction) - **RECOMMENDED for visualization**
  - **0.5**: Moderate smoothing (~70% jump reduction)
  - **0.3**: Heavy smoothing (~90% jump reduction, but 60% accuracy loss) - **NOT RECOMMENDED**
  - **<0.3**: Extreme smoothing (destroys 80% of localization accuracy) - **NEVER USE**

- `--max_windows <N>`: Limit number of windows (default: process entire file)
- `--window_seconds <float>`: Window duration (default: 0.5 seconds)
- `--output_dir <path>`: Custom output directory

### Legacy (Not Recommended)
- `--no_pad_last`: Drop last window instead of padding (default: pad)

## NPZ File Format

Both `animation_data.npz` and `animation_data_raw.npz` contain:

```python
{
    'activity': np.float32,      # Shape: (n_sources, n_frames)
                                 # Source activity at each timepoint
    
    'timestamps': np.float32,    # Shape: (n_frames,)
                                 # Timestamp in seconds for each frame
    
    'source_positions': np.float32,  # Shape: (n_sources, 3)
                                     # 3D positions (x, y, z) in mm
    
    'triangles': np.int32,       # Shape: (n_triangles, 3)
                                 # Triangle mesh connectivity
    
    'fps': np.int32,             # Scalar
                                 # Actual frames per second (~4 for 50% overlap)
}
```

## Loading in Frontend

```javascript
// Example: Load NPZ in JavaScript using numpy.js or similar
const data = await loadNPZ('animation_data.npz');

const activity = data.activity;         // (n_sources, n_frames)
const timestamps = data.timestamps;     // (n_frames,)
const positions = data.source_positions; // (n_sources, 3)
const triangles = data.triangles;       // (n_triangles, 3)
const fps = data.fps;                   // ~4 FPS

// For each frame:
for (let frame = 0; frame < activity.shape[1]; frame++) {
    const frameActivity = activity.slice([null, frame]);  // (n_sources,)
    const time = timestamps[frame];
    
    // Update 3D brain visualization with frameActivity at positions
    updateBrain(positions, triangles, frameActivity);
    
    // Wait for next frame
    await sleep(1000 / fps);
}
```

## Frontend Integration Example

```typescript
// Example: React component for EEG animation playback
import React, { useState, useEffect } from 'react';
import { loadNPZ } from 'numpy-loader';
import BrainVisualization from './BrainVisualization';

export const EEGAnimationPlayer = ({ npzPath }) => {
    const [data, setData] = useState(null);
    const [frame, setFrame] = useState(0);
    const [isPlaying, setIsPlaying] = useState(false);
    
    useEffect(() => {
        loadNPZ(npzPath).then(setData);
    }, [npzPath]);
    
    useEffect(() => {
        if (!isPlaying || !data) return;
        
        const interval = setInterval(() => {
            setFrame(prev => (prev + 1) % data.timestamps.length);
        }, 1000 / data.fps);
        
        return () => clearInterval(interval);
    }, [isPlaying, data]);
    
    if (!data) return <div>Loading...</div>;
    
    return (
        <div>
            <BrainVisualization
                positions={data.source_positions}
                triangles={data.triangles}
                activity={data.activity.slice([null, frame])}
            />
            <div>
                <button onClick={() => setIsPlaying(!isPlaying)}>
                    {isPlaying ? 'Pause' : 'Play'}
                </button>
                <span>Frame {frame + 1} / {data.timestamps.length}</span>
                <span>Time: {data.timestamps[frame].toFixed(2)}s</span>
            </div>
        </div>
    );
};
```

## Performance Considerations

### File Size
- **Typical:** 2-3 MB per 50 windows (~12.5 seconds)
- **Full recording (300 windows):** ~15-20 MB
- **Compression:** Already uses `np.savez_compressed` (no further compression needed)

### Frame Rate
- **Window overlap 50%:** ~4 FPS (smooth playback)
- **Window overlap 75%:** ~8 FPS (very smooth, 2× file size)
- **Window overlap 25%:** ~2 FPS (choppy, smaller file)

**Recommended:** 50% overlap balances smoothness and file size.

### Memory Usage
- **Load time:** <100ms for typical file
- **Memory footprint:** ~10-20 MB in browser
- **Rendering:** Use WebGL for 1284-2562 sources (excellent performance)

## Troubleshooting

### "Activity values are very small (1e-10 range)"
**This is expected.** Source activity is in units of current density (A/m²). Always normalize for visualization:

```python
# Normalize activity to 0-1 range for each frame
activity_normalized = activity / np.max(np.abs(activity), axis=0, keepdims=True)
```

### "Flashes/jumps still visible with smoothing"
**This is normal.** Smoothing reduces jumps by ~58% (α=0.6) but doesn't eliminate them. Flashes represent genuine EEG variations. If further smoothing needed:
- Try α=0.5 (70% reduction) - acceptable for visualization
- DO NOT use α<0.3 (destroys localization accuracy)

### "Different results between runs"
Check:
1. Same `--use_global_norm` flag (creates consistency)
2. Same `--overlap_fraction` (affects frame rate and smoothness)
3. Same `--smoothing_alpha` (affects temporal dynamics)

### "Missing channels error"
Pipeline automatically interpolates missing channels. If error persists:
1. Check EDF channel labels match standard_1020 naming
2. Verify EDF contains EEG channels (not just ECG/EMG)

## Example Workflows

### 1. Quick Preview (5 seconds)
```bash
python run_edf_inference.py input.edf \
  --use_global_norm \
  --smoothing_alpha 0.6 \
  --overlap_fraction 0.5 \
  --max_windows 20
```

### 2. Full Recording (Production)
```bash
python run_edf_inference.py input.edf \
  --use_global_norm \
  --smoothing_alpha 0.6 \
  --overlap_fraction 0.5 \
  --output_dir /path/to/output
```

### 3. Research Analysis (No Smoothing)
```bash
python run_edf_inference.py input.edf \
  --use_global_norm \
  --overlap_fraction 0.5 \
  --output_dir /path/to/output
```

### 4. Batch Processing
```bash
for edf in *.edf; do
    python run_edf_inference.py "$edf" \
      --use_global_norm \
      --smoothing_alpha 0.6 \
      --overlap_fraction 0.5 \
      --output_dir "results/$(basename $edf .edf)"
done
```

## API Integration

If integrating into a web service:

```python
# Example: Flask API endpoint
from flask import Flask, request, send_file
import subprocess
import os

app = Flask(__name__)

@app.route('/api/process_eeg', methods=['POST'])
def process_eeg():
    # Save uploaded EDF
    edf_file = request.files['edf']
    edf_path = f'/tmp/{edf_file.filename}'
    edf_file.save(edf_path)
    
    # Run inference with global norm + smoothing
    output_dir = f'/tmp/output_{os.path.basename(edf_path)}'
    subprocess.run([
        'conda', 'run', '-n', 'inv_solver',
        'python', 'run_edf_inference.py',
        edf_path,
        '--use_global_norm',
        '--smoothing_alpha', '0.6',
        '--overlap_fraction', '0.5',
        '--output_dir', output_dir
    ])
    
    # Return animation data
    return send_file(f'{output_dir}/animation_data.npz')
```

## Migration from Old Pipeline

### Old Approach
```bash
python run_edf_inference.py input.edf --overlap_fraction 0.5
```
- Used per-window normalization (6.89× artificial variations)
- No smoothing option
- Only one output file

### New Approach
```bash
python run_edf_inference.py input.edf \
  --use_global_norm \
  --smoothing_alpha 0.6 \
  --overlap_fraction 0.5
```
- Uses global normalization (consistent across recording)
- Optional smoothing (preserves raw data)
- Two output files (raw + smoothed)

**Migration steps:**
1. Add `--use_global_norm` to all scripts
2. Add `--smoothing_alpha 0.6` for visualization endpoints
3. Use `animation_data.npz` (smoothed) in frontend
4. Keep `animation_data_raw.npz` for analysis tools

---

**Questions?** See `FLASH_ARTIFACTS_REPORT.md` for full technical details.
