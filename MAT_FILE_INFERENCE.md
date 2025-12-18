# MAT File Inference

This document describes how to run inference on simulation MAT files and view evaluation metrics.

## Overview

The MAT file inference pipeline supports:
- **Sliding window approach**: Process long recordings with configurable overlap
- **3D brain visualization**: Interactive heatmaps with playback animation
- **Evaluation metrics**: When ground truth is available (simulation files), compute:
  - Mean nMSE (normalized Mean Squared Error)
  - Mean AUC (Area Under ROC Curve)
  - Localization Error (mm)
  - Time Error (ms)

## Command Line Usage

### Basic Usage

```bash
cd /home/zik/UniStuff/FYP/stESI_pub
conda run -n inv_solver python3 inverse_problem/run_mat_inference.py <mat_file_path> [options]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--simu_name` | `mes_debug` | Simulation name for head model config |
| `--subject` | `fsaverage` | Subject folder name |
| `--orientation` | `constrained` | Source orientation |
| `--electrode_montage` | `standard_1020` | Electrode montage |
| `--source_space` | `ico3` | Source space name |
| `--model_path` | Auto-detect | Path to 1dCNN checkpoint |
| `--source_file` | Auto-detect | Ground truth source file |
| `--inter_layer` | `4096` | Model intermediate layer size |
| `--kernel_size` | `5` | Model kernel size |
| `--train_loss` | `cosine` | Loss used during training |
| `--window_samples` | From config | Window size in samples |
| `--overlap_fraction` | `0.5` | Window overlap (0-0.95) |
| `--max_windows` | None | Maximum windows to process |
| `--output_dir` | Auto | Output directory |
| `--open_plot` | False | Open plot in browser |

### Example with Ground Truth

```bash
conda run -n inv_solver python3 inverse_problem/run_mat_inference.py \
  simulation/fsaverage/constrained/standard_1020/ico3/simu/mes_debug/eeg/infdb/1_eeg.mat \
  --source_file simulation/fsaverage/constrained/standard_1020/ico3/simu/mes_debug/sources/Jact/1_src_act.mat \
  --output_dir results/mat_inference/test_1
```

## Output Files

The inference script generates the following files in the output directory:

| File | Description |
|------|-------------|
| `inference_summary.json` | Main summary with metrics and settings |
| `*_interactive.html` | Interactive 3D brain heatmap |
| `animation_data.npz` | Animation data for playback |
| `segments/` | Individual window MAT files |

### Summary JSON Structure

```json
{
  "mat_file": "/path/to/input.mat",
  "window_samples": 256,
  "overlap_fraction": 0.5,
  "n_windows_processed": 2,
  "best_window": {
    "window_index": 0,
    "start_time_seconds": 0.0,
    "peak_time_index": 214,
    "score": 2491.60
  },
  "interactive_plot": "1_eeg_window0000_interactive.html",
  "has_ground_truth": true,
  "source_file": "path/to/source.mat",
  "metrics": {
    "mean_nmse": 0.0101,
    "mean_auc": 0.7337,
    "mean_localization_error_mm": 17.25,
    "mean_time_error_ms": 0.0,
    "n_windows": 2
  }
}
```

## Frontend Integration

### Upload MAT Files

The frontend now accepts both EDF and MAT files:

1. Go to the VESL web interface
2. Upload a `.mat` file (simulation) or `.edf` file (real recording)
3. Wait for processing
4. View results in 3D Brain Map tab
5. For MAT files with ground truth, view metrics in Analysis Details tab

### API Endpoint

**POST** `/api/analyze-mat`

Form data:
- `file`: The MAT file to analyze

Response includes:
- `plotHtml`: Interactive plot HTML
- `animationFile`: Path to animation NPZ
- `hasGroundTruth`: Whether metrics are available
- `metrics`: Evaluation metrics (if ground truth found)

## MAT File Format

### EEG Data Format
```matlab
% Expected structure
eeg_data.EEG = [n_electrodes x n_times] matrix
```

### Source Data Format (Ground Truth)
```matlab
% Expected structure  
Jact.Jact = [n_active_sources x n_times] matrix
```

With metadata JSON file:
```json
{
  "seeds": [125, 1246, 1004],
  "orders": [4, 5, 1],
  "n_patch": 3,
  "act_src": {
    "patch_1": [10, 26, 27, ...],
    "patch_2": [653, 679, 680, ...],
    "patch_3": [654, 684, 865, ...]
  }
}
```

## Metrics Explanation

| Metric | Description | Good Value |
|--------|-------------|------------|
| **nMSE** | Normalized mean squared error between predicted and actual source activity | Lower is better (< 0.1) |
| **AUC** | Area under ROC curve for source detection | Higher is better (> 0.7) |
| **Localization Error** | Distance between true and estimated source locations | Lower is better (< 20 mm) |
| **Time Error** | Difference in peak activity timing | Lower is better (< 10 ms) |
