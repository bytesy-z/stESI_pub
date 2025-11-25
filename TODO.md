# Implementation TODO List: Animated Brain Visualization

This document outlines the step-by-step changes needed to transform the current single-window EEG source localization visualization into a full temporal animation system with playable 3D brain activity over time.

## Backend Changes (Python)

### 1. Update Python inference script to save all window predictions
**File**: `inverse_problem/run_edf_inference.py`  
**Location**: Lines 262-299 in `main()` function

**Task**: Modify the main loop to store predictions from ALL windows instead of just tracking the best one.

**Changes**:
- Create a list `all_window_predictions = []` before the segmentation loop
- For each window processed, append a dictionary containing:
  - `window_idx`: segment index
  - `start_time`: start time in seconds (start_sample / fs)
  - `end_time`: end time in seconds ((start_sample + window_samples) / fs)
  - `predictions`: the full prediction tensor as numpy array (n_sources × n_timepoints)
  - `max_abs`: the normalization factor used
- Keep the existing "best window" tracking for backward compatibility
- This provides the foundation for temporal animation by preserving all temporal information

---

### 2. Add overlap parameter to Python argument parser
**File**: `inverse_problem/run_edf_inference.py`  
**Location**: Around line 230 in `parse_args()` function

**Task**: Add the `--overlap_fraction` argument to argparse (currently referenced but not defined).

**Changes**:
```python
parser.add_argument(
    "--overlap_fraction",
    type=float,
    default=0.5,
    help="Overlap fraction between consecutive windows (0.0=no overlap, 0.5=50% overlap, max=0.95).",
)
```

**Note**: This parameter is already used in line 309 but needs to be properly defined in the argument parser.

---

### 3. Create window interpolation function for smooth transitions
**File**: `inverse_problem/run_edf_inference.py`  
**Location**: After line 145 (after `_load_model()` function)

**Task**: Add new function to interpolate overlapping window predictions into a smooth timeline.

**Function signature**:
```python
def _interpolate_sliding_windows(
    all_predictions: List[dict],
    target_fps: int = 30,
) -> Tuple[np.ndarray, np.ndarray]:
```

**Implementation details**:
- Determine total duration from max end_time across all predictions
- Calculate number of frames: `n_frames = int(total_duration * target_fps)`
- Create timestamp array: `np.linspace(0, total_duration, n_frames)`
- Initialize output arrays: `activity_timeline` (n_sources × n_frames), `weights` (n_frames)
- For each prediction window:
  - Find overlapping frames using timestamp matching
  - Interpolate within window prediction for each frame
  - Apply Gaussian weighting (higher weight at window center, lower at edges)
  - Accumulate weighted predictions: `activity_timeline[:, frame_idx] += pred * weight`
  - Track total weights: `weights[frame_idx] += weight`
- Normalize by total weights: `activity_timeline /= (weights + 1e-8)`
- Return `(activity_timeline, timestamps)` as float32 arrays

**Purpose**: This eliminates boundary artifacts and creates smooth temporal transitions between overlapping windows.

---

### 4. Generate and save animation data as NPZ format
**File**: `inverse_problem/run_edf_inference.py`  
**Location**: After line 410 in `main()` function (after best window processing)

**Task**: Call the interpolation function and save comprehensive animation data.

**Changes**:
- After collecting all window predictions, call:
  ```python
  activity_timeline, timestamps = _interpolate_sliding_windows(
      all_window_predictions,
      target_fps=30
  )
  ```
- Prepare animation data dictionary:
  ```python
  animation_data = {
      'activity': activity_timeline.astype(np.float32),         # (n_sources, n_frames)
      'timestamps': timestamps.astype(np.float32),              # (n_frames,)
      'source_positions': geom.positions.astype(np.float32),   # (n_sources, 3)
      'triangles': geom.triangles.astype(np.int32),            # (n_triangles, 3)
      'fps': np.array(30, dtype=np.int32),                     # scalar
  }
  ```
- Save using: `np.savez_compressed(output_dir / 'animation_data.npz', **animation_data)`
- Log file size and frame count for user feedback

**Output**: Single compressed NPZ file containing all data needed for frontend visualization.

---

## API Integration (TypeScript)

### 5. Update API route to pass overlap parameter to Python
**File**: `frontend/app/api/analyze-eeg/route.ts`  
**Location**: Around lines 50-60 (where `args` array is constructed)

**Task**: Add overlap parameter to Python script arguments.

**Changes**:
- In the `args` array passed to `spawn("python3", args)`, add:
  ```typescript
  "--overlap_fraction", "0.5",
  ```
- This should be added after existing parameters like `--source_space` and before the filepath
- Enables 50% window overlap for smoother temporal resolution

---

### 6. Update API route response to include animation file
**File**: `frontend/app/api/analyze-eeg/route.ts`  
**Location**: Around line 120 (NextResponse.json() return statement)

**Task**: Add animation file reference to API response.

**Changes**:
- In the returned JSON object, add new field:
  ```typescript
  animationFile: "animation_data.npz",
  ```
- This tells the frontend where to find the animation data file
- Frontend will construct full path as `/${outputDir}/animation_data.npz`

---

## Frontend Infrastructure (TypeScript/React)

### 7. Install Three.js dependencies in frontend
**File**: N/A (terminal command)  
**Location**: `frontend/` directory

**Task**: Install required npm packages for 3D rendering and NPZ parsing.

**Commands**:
```bash
cd frontend
pnpm add three @types/three
pnpm add fflate
```

**Packages**:
- `three`: Three.js library for WebGL 3D rendering
- `@types/three`: TypeScript type definitions
- `fflate`: Fast JavaScript compression library for parsing .npz files

---

### 8. Create NPZ parser utility for browser
**File**: `frontend/lib/npz-parser.ts` (new file)

**Task**: Implement browser-side NPZ file parser.

**Function signature**:
```typescript
export async function parseNPZ(
  buffer: ArrayBuffer
): Promise<Record<string, Float32Array | Int32Array>>
```

**Implementation outline**:
- Use `fflate.unzipSync()` to decompress the ZIP archive
- For each file in the archive:
  - Parse NPY header to determine dtype, shape, byte order
  - Extract data buffer and create typed array (Float32Array or Int32Array)
  - Handle both C-order and Fortran-order arrays
  - Store in result object with filename (without .npy extension) as key
- Return object with all arrays

**NPY format reference**: Header is ASCII dict followed by binary data, starts with magic bytes `\x93NUMPY`

---

### 9. Create inferno colormap utility function
**File**: `frontend/lib/colormaps.ts` (new file)

**Task**: Implement inferno colormap for activity visualization.

**Function signature**:
```typescript
export function infernoColormap(value: number): [number, number, number]
```

**Implementation**:
- Pre-define 256-entry lookup table with RGB triplets from matplotlib's inferno colormap
- Input `value` should be in range [0, 1] (0=lowest activity, 1=highest)
- Clamp value to [0, 1]
- Interpolate between nearest two entries in lookup table
- Return `[r, g, b]` with values in range [0, 1]

**Inferno characteristics**: 
- Low values: dark purple/black
- Mid values: dark red/orange
- High values: bright yellow/white
- Perceptually uniform and colorblind-friendly

---

## Visualization Components (TypeScript/React)

### 10. Create AnimatedBrainVisualization component
**File**: `frontend/components/animated-brain-visualization.tsx` (new file)

**Task**: Build main component for animated 3D brain rendering.

**Component props**:
```typescript
interface Props {
  dataUrl: string;  // Path to animation_data.npz
}
```

**Implementation outline**:
- **State management**:
  - `data: AnimationData | null` - parsed NPZ data
  - `currentFrame: number` - current frame index
  - `isPlaying: boolean` - playback state
  - `error: string | null` - error messages
  
- **Data loading** (useEffect on mount):
  - Fetch `dataUrl` as ArrayBuffer
  - Parse using `parseNPZ()`
  - Reshape arrays if needed (handle 1D flattened data)
  - Set state when complete
  
- **Three.js setup** (useEffect when data loads):
  - Create Scene, PerspectiveCamera, WebGLRenderer
  - Build BufferGeometry from source_positions and triangles
  - Add color attribute to geometry for per-vertex coloring
  - Create Mesh with MeshBasicMaterial (vertexColors: true)
  - Add OrbitControls for user interaction (rotate, zoom, pan)
  - Set up animation loop with `requestAnimationFrame`
  - Return cleanup function to dispose resources
  
- **Color updates** (useEffect on currentFrame change):
  - Extract activity for current frame: `data.activity[:, currentFrame]`
  - Normalize to [0, 1] range
  - Map each source activity to RGB using `infernoColormap()`
  - Update geometry color attribute
  - Set `needsUpdate = true` to trigger re-render

**Render**: Container div with ref for Three.js canvas injection

---

### 11. Add playback controls to AnimatedBrainVisualization
**File**: `frontend/components/animated-brain-visualization.tsx`  
**Location**: Below the 3D canvas container

**Task**: Add UI controls for animation playback.

**Controls to implement**:
1. **Play/Pause button**:
   - Import `Button` from `@/components/ui/button`
   - Toggle `isPlaying` state on click
   - Show "Pause" when playing, "Play" when paused
   - Icon: Use play/pause icons from lucide-react

2. **Timeline slider**:
   - `<input type="range" min={0} max={data?.timestamps.length ?? 0} value={currentFrame} />`
   - On change: update `currentFrame` state
   - Pauses playback automatically when user scrubs
   - Full width styling with Tailwind CSS

3. **Timestamp display**:
   - Show current time: `{data?.timestamps[currentFrame].toFixed(2)}s`
   - Also show total duration: ` / {data?.timestamps[data.timestamps.length - 1].toFixed(2)}s`

**Animation playback** (useEffect on isPlaying):
- When `isPlaying` is true:
  - Use `setInterval()` with period `1000 / data.fps` ms
  - Increment `currentFrame` each tick
  - Loop back to 0 when reaching end
  - Clear interval in cleanup function
- When false: no interval

**Layout**: Flexbox row with gap between controls, responsive spacing

---

### 12. Update OutputWindow to conditionally render animation
**File**: `frontend/components/output-window.tsx`  
**Location**: Around lines 20-40 (main render section)

**Task**: Add conditional rendering for animation visualization.

**Changes**:
- Import: `import { AnimatedBrainVisualization } from './animated-brain-visualization'`
- Check if `result.animationFile` exists
- If yes, render:
  ```tsx
  <AnimatedBrainVisualization 
    dataUrl={`/${result.outputDir}/${result.animationFile}`} 
  />
  ```
- Optionally keep the static HTML plot as a fallback or secondary view
- Add tabs or toggle to switch between animation and static plot
- Use conditional rendering: `{result.animationFile ? <Animated... /> : <iframe... />}`

**Backward compatibility**: If `animationFile` is not in result, fall back to existing iframe visualization.

---

### 13. Update main page to pass animation data to OutputWindow
**File**: `frontend/app/page.tsx`  
**Location**: Around lines 30-50 (state management and result handling)

**Task**: Ensure animation file reference propagates through the component tree.

**Changes**:
- Verify that `result` state includes all fields from API response
- When setting result state after API call, spread entire response: `setResult(data)`
- Ensure TypeScript interface for result includes optional `animationFile?: string` field
- Pass complete result to `<OutputWindow result={result} />`
- No filtering or transformation should remove the `animationFile` field

**Type safety**: Update result interface if needed to include new field.

---

## Testing & Polish

### 14. Test end-to-end pipeline with sample EDF
**File**: Multiple files  
**Location**: Full system test

**Task**: Comprehensive testing of the complete pipeline.

**Test steps**:
1. Start development server: `cd frontend && pnpm dev`
2. Open browser to `http://localhost:3000`
3. Upload `sample/0001082.edf` through the UI
4. Monitor browser console and network tab for errors
5. Wait for Python processing to complete
6. Verify outputs:
   - Check `results/edf_inference/.../animation_data.npz` exists
   - Verify file size is reasonable (5-10 MB for 60s recording)
   - Confirm 3D visualization renders
   - Test animation playback (play/pause, scrubbing)
   - Verify smooth transitions between frames
   - Check activity patterns make sense
7. Test edge cases:
   - Very short EEG files (< 2 seconds)
   - Files with missing channels
   - Non-standard sampling rates
8. Performance testing:
   - Monitor browser memory usage during playback
   - Check frame rate stays at target FPS
   - Verify no memory leaks on repeated uploads

**Debug checklist**:
- Python script errors: Check terminal output
- NPZ parsing errors: Check browser console
- WebGL errors: Verify browser support
- Missing data: Verify all NPZ arrays are present
- Visual artifacts: Check color normalization and interpolation

---

### 15. Add error handling for animation loading failures
**File**: `frontend/components/animated-brain-visualization.tsx`  
**Location**: Throughout component

**Task**: Implement comprehensive error handling with user feedback.

**Error scenarios to handle**:
1. **Network errors**: Animation file fails to fetch (404, network down)
2. **Parse errors**: NPZ file is corrupted or wrong format
3. **WebGL errors**: Browser doesn't support WebGL or context creation fails
4. **Data validation**: NPZ contains unexpected array shapes or missing fields
5. **Memory errors**: File too large to load in browser

**Implementation**:
- Wrap fetch and parse in try-catch block
- Validate NPZ contents before creating geometry:
  ```typescript
  if (!data.activity || !data.source_positions || !data.triangles) {
    throw new Error('Invalid animation data format');
  }
  ```
- Check WebGL support before creating renderer
- Use `error` state to store error messages
- Import and render `<ErrorAlert>` component when error exists
- Provide helpful error messages:
  - "Failed to load animation data. Please try again."
  - "Your browser doesn't support WebGL. Please use a modern browser."
  - "Animation data is corrupted. Please re-run the analysis."

**User experience**: Show error inline instead of breaking entire page.

---

### 16. Optimize animation data size and loading
**File**: Multiple files  
**Location**: Backend and frontend

**Task**: Performance optimization to handle large recordings efficiently.

**Optimization strategies**:

1. **Temporal downsampling** (Python):
   - If recording > 60 seconds, reduce FPS automatically
   - Example: 30 FPS for <60s, 15 FPS for >60s
   - Add parameter: `--target_fps` with auto-calculation

2. **Spatial downsampling** (Python):
   - For preview mode, reduce number of sources
   - Cluster nearby sources and average activity
   - Save both full-res and low-res versions

3. **Progressive loading** (Frontend):
   - Load low-res preview first for quick feedback
   - Stream full-res data in background
   - Switch to full-res when loaded

4. **Compression tuning** (Python):
   - Test different compression levels in `np.savez_compressed()`
   - Consider using fewer bits (float16 instead of float32)
   - Profile: `compression_level` parameter in fflate

5. **Lazy frame generation** (Frontend):
   - Don't pre-compute all frame colors
   - Compute colors on-demand for current frame
   - Reduces memory usage

**Monitoring**:
- Log file sizes in Python script
- Use `performance.memory` API in browser
- Set maximum file size threshold (e.g., 50 MB)
- Warn user if file is large before loading

---

### 17. Add animation export functionality
**File**: `frontend/components/animated-brain-visualization.tsx`  
**Location**: Additional UI control (optional enhancement)

**Task**: Allow users to export animation as video or GIF.

**Implementation options**:

1. **MediaRecorder API** (native browser):
   - Capture canvas frames using `canvas.captureStream()`
   - Record with `MediaRecorder` to WebM format
   - Download as video file
   - Pros: Native, no dependencies
   - Cons: Limited format support, quality control

2. **CCapture.js library**:
   - Frame-by-frame capture with precise control
   - Export as WebM, animated GIF, or frame sequence
   - Better quality and format options
   - Add dependency: `pnpm add ccapture.js`

**UI**:
- Add "Export Animation" button below playback controls
- Show modal/dialog with export options:
  - Format: MP4/WebM/GIF
  - Quality: Low/Medium/High
  - Frame rate: 15/30/60 FPS
  - Duration: Full/Selection
- Progress bar during export
- Download link when complete

**User experience**:
- Disable controls during export
- Show estimated file size
- Allow cancellation
- Handle export errors gracefully

---

## Summary of Changes

| Category | Files Modified | Files Created | New Dependencies |
|----------|---------------|---------------|------------------|
| **Python Backend** | `run_edf_inference.py` | None | None |
| **API Integration** | `app/api/analyze-eeg/route.ts` | None | None |
| **Frontend Utils** | None | `lib/npz-parser.ts`<br>`lib/colormaps.ts` | `fflate` |
| **Frontend Components** | `components/output-window.tsx`<br>`app/page.tsx` | `components/animated-brain-visualization.tsx` | `three`<br>`@types/three` |
| **Documentation** | None | `frontend/README.md`<br>`TODO.md` (this file) | None |

## Estimated Implementation Time

- **Backend changes** (Tasks 1-4): 4-6 hours
- **API integration** (Tasks 5-6): 30 minutes
- **Frontend infrastructure** (Tasks 7-9): 2-3 hours
- **Visualization components** (Tasks 10-13): 6-8 hours
- **Testing & polish** (Tasks 14-17): 4-6 hours

**Total**: ~20-25 hours of development time

## Dependencies & Prerequisites

**Python**:
- numpy >= 1.20
- scipy >= 1.7
- torch >= 1.9
- mne >= 0.24
- plotly >= 5.0

**Node.js**:
- Node.js >= 18
- pnpm >= 8
- Next.js 14+
- React 18+

**Data**:
- Pre-trained 1D-CNN model checkpoint
- fsaverage head model files
- Sample EDF file for testing

## Notes

- All line numbers are approximate and may shift as code is modified
- Test each change incrementally rather than implementing everything at once
- Consider creating a feature branch for this work
- Keep the existing single-window visualization as fallback for compatibility
- Monitor performance throughout development, especially browser memory usage
- Document any deviations from this plan in commit messages

## Success Criteria

✅ User can upload an EDF file and see animated 3D brain activity  
✅ Animation plays smoothly at 30 FPS  
✅ Playback controls (play/pause/scrub) work intuitively  
✅ Overlapping windows create smooth temporal transitions  
✅ Output file size is reasonable (< 20 MB for 60s recording)  
✅ Browser doesn't crash or lag during playback  
✅ Error handling provides helpful feedback  
✅ Backward compatible with existing static visualization
