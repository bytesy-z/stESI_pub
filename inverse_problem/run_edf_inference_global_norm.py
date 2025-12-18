#!/usr/bin/env python3
"""
Patch to apply global normalization to EDF inference.

This module provides a modified inference pipeline that uses a single global
normalization factor instead of per-window normalization, which reduces
artificial amplitude swings ("flashes") in the output.

Usage:
    python run_edf_inference_global_norm.py path/to/file.edf [options]
    
The key difference from run_edf_inference.py:
- Computes max_abs once for the entire recording
- All windows use the same normalization factor
- Eliminates per-window amplitude artifacts
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mne
import numpy as np
import torch
from scipy.io import savemat
from scipy.signal import windows as scipy_windows

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_data import HeadModel
from load_data.FolderStructure import FolderStructure
from models.cnn_1d import CNN1Dpl
from plot_1dcnn_inference_heatmap import (
    BrainGeometry,
    _ensure_within_repo,
    _find_default_model_checkpoint,
    _infer_model,
    _load_brain_geometry,
    _save_interactive_heatmap,
)


@dataclass
class SegmentSummary:
    index: int
    start_sample: int
    start_time_seconds: float
    eeg_max_abs: float
    mat_path: Path


def _canonical_lookup(names):
    return {name.lower(): name for name in names}


def _rename_channels_to_target(raw, target_names):
    lookup = _canonical_lookup(target_names)
    rename_map = {}
    for ch in raw.ch_names:
        key = ch.lower()
        if key in lookup:
            rename_map[ch] = lookup[key]
        else:
            key = ch.replace(" ", "").lower()
            if key in lookup:
                rename_map[ch] = lookup[key]
    if rename_map:
        raw.rename_channels(rename_map)


def _build_full_montage_raw(source_raw, target_info, target_names):
    data = np.zeros((len(target_names), source_raw.n_times), dtype=np.float64)
    missing = []
    for idx, name in enumerate(target_names):
        if name in source_raw.ch_names:
            data[idx] = source_raw.get_data(picks=[name])[0]
        else:
            missing.append(name)
    full_info = target_info.copy()
    full_raw = mne.io.RawArray(data, full_info, verbose=False)
    if missing:
        full_raw.info["bads"] = missing.copy()
        full_raw.interpolate_bads(reset_bads=True, verbose=False, mode="accurate")
    return full_raw, missing


def _segment_signal(data, window_samples, step_samples, pad, max_windows):
    n_samples = data.shape[1]
    if n_samples == 0:
        return
    start = 0
    n_generated = 0
    while start < n_samples and (max_windows is None or n_generated < max_windows):
        end = start + window_samples
        segment = data[:, start:end]
        if segment.shape[1] < window_samples:
            if not pad:
                break
            pad_width = window_samples - segment.shape[1]
            segment = np.pad(segment, ((0, 0), (0, pad_width)), mode="edge")
        yield start, segment
        n_generated += 1
        start += step_samples
        if step_samples <= 0:
            break


def _load_model(checkpoint_path, n_electrodes, n_sources, inter_layer=4096, kernel_size=5):
    net_params = {
        "channels": [n_electrodes, inter_layer, n_sources],
        "kernel_size": kernel_size,
        "bias": False,
        "optimizer": None,
        "lr": 1e-3,
        "criterion": None,
    }
    model = CNN1Dpl(**net_params)
    state_dict = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _prepare_animation_timeline_with_smoothing(
    all_predictions: List[dict],
    overlap_fraction: float = 0.5,
    temporal_smoothing: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare animation timeline with optional temporal smoothing.
    
    Args:
        all_predictions: List of prediction dictionaries
        overlap_fraction: Overlap between windows (used for Hann weighting)
        temporal_smoothing: Exponential smoothing factor (0 = no smoothing, 0.9 = heavy smoothing)
        
    Returns:
        (activity_timeline, timestamps) arrays
    """
    if not all_predictions:
        raise ValueError("all_predictions list is empty")
    
    n_windows = len(all_predictions)
    n_sources = all_predictions[0]['predictions'].shape[0]
    n_timepoints = all_predictions[0]['predictions'].shape[1]
    
    activity_timeline = np.zeros((n_sources, n_windows), dtype=np.float32)
    timestamps = np.zeros(n_windows, dtype=np.float32)
    
    # Hann window for temporal weighting within windows
    hann = scipy_windows.hann(n_timepoints)
    
    for i, window in enumerate(all_predictions):
        win_pred = window['predictions']
        
        # Find peak activity timepoint
        total_activity = np.abs(win_pred).sum(axis=0)
        peak_idx = np.argmax(total_activity)
        
        # Get activity at peak
        current_activity = win_pred[:, peak_idx]
        
        if overlap_fraction > 0 and i > 0:
            # Blend with previous frame using overlap-weighted average
            prev_activity = activity_timeline[:, i-1]
            blend_weight = overlap_fraction * 0.5  # Moderate blending
            current_activity = (1 - blend_weight) * current_activity + blend_weight * prev_activity
        
        activity_timeline[:, i] = current_activity
        timestamps[i] = (window['start_time'] + window['end_time']) / 2.0
    
    # Apply temporal smoothing if requested
    if temporal_smoothing > 0:
        for i in range(1, n_windows):
            activity_timeline[:, i] = (
                temporal_smoothing * activity_timeline[:, i-1] +
                (1 - temporal_smoothing) * activity_timeline[:, i]
            )
    
    return activity_timeline, timestamps


def _prepare_output_dir(base_dir, repo_root, edf_path):
    if base_dir:
        out_dir = Path(base_dir).expanduser()
    else:
        out_dir = repo_root / "results" / "edf_inference_global_norm" / edf_path.stem
    out_dir = _ensure_within_repo(out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "segments").mkdir(exist_ok=True)
    return out_dir


def main():
    parser = argparse.ArgumentParser(description="Run 1dCNN inference with GLOBAL normalization.")
    parser.add_argument("edf_path", type=str)
    parser.add_argument("--simu_name", default="mes_debug")
    parser.add_argument("--subject", default="fsaverage")
    parser.add_argument("--orientation", default="constrained")
    parser.add_argument("--electrode_montage", default="standard_1020")
    parser.add_argument("--source_space", default="ico3")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--inter_layer", type=int, default=4096)
    parser.add_argument("--kernel_size", type=int, default=5)
    parser.add_argument("--train_loss", default="cosine")
    parser.add_argument("--window_seconds", type=float, default=None)
    parser.add_argument("--overlap_fraction", type=float, default=0.5)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--temporal_smoothing", type=float, default=0.0,
                        help="Exponential smoothing factor (0-1). Higher = smoother.")
    parser.add_argument("--normalization_mode", choices=["global", "robust_global", "percentile"],
                        default="robust_global",
                        help="How to compute the global normalization factor")
    parser.add_argument("--open_plot", action="store_true")
    parser.add_argument("--no_pad_last", action="store_true")
    
    args = parser.parse_args()
    
    repo_root = REPO_ROOT
    edf_path = Path(args.edf_path).expanduser().resolve()
    
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    
    # Setup paths
    simu_root = repo_root / "simulation" / args.subject
    config_path = (
        simu_root / args.orientation / args.electrode_montage / args.source_space
        / "simu" / args.simu_name / f"{args.simu_name}{args.source_space}_config.json"
    )
    
    with config_path.open() as f:
        general_config = json.load(f)
    
    general_config["simu_name"] = args.simu_name
    general_config["electrode_space"]["electrode_montage"] = args.electrode_montage
    general_config.setdefault("eeg_snr", "infdb")
    
    # Load head model
    folders = FolderStructure(str(simu_root), general_config)
    electrode_space = HeadModel.ElectrodeSpace(folders, general_config)
    source_space_obj = HeadModel.SourceSpace(folders, general_config)
    head_model = HeadModel.HeadModel(electrode_space, source_space_obj, folders, args.subject)
    
    # Setup window parameters
    fs = float(general_config["rec_info"]["fs"])
    if args.window_seconds is None:
        window_samples = int(general_config["rec_info"]["n_times"])
        window_seconds = window_samples / fs
    else:
        window_seconds = float(args.window_seconds)
        window_samples = max(1, int(round(window_seconds * fs)))
    
    overlap_fraction = min(max(args.overlap_fraction, 0.0), 0.95)
    step_samples = max(1, int(round(window_samples * (1.0 - overlap_fraction))))
    
    output_dir = _prepare_output_dir(args.output_dir, repo_root, edf_path)
    
    # Load model
    if args.model_path:
        model_path = Path(args.model_path)
    else:
        token = f"{args.simu_name}{args.source_space}_"
        model_path = _find_default_model_checkpoint(token, args.inter_layer)
        if model_path is None:
            raise FileNotFoundError("Cannot find model checkpoint")
    
    model = _load_model(
        model_path,
        head_model.electrode_space.n_electrodes,
        head_model.source_space.n_sources,
        args.inter_layer,
        args.kernel_size,
    )
    
    # Load EDF
    print(f"Loading EDF: {edf_path}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    raw.pick(picks="eeg")
    _rename_channels_to_target(raw, head_model.electrode_space.info.ch_names)
    
    try:
        raw.set_montage("standard_1020", match_case=False, verbose=False, on_missing="warn")
    except RuntimeError:
        pass
    
    raw.set_eeg_reference("average", projection=False, verbose=False)
    
    if not math.isclose(raw.info["sfreq"], fs, rel_tol=1e-6):
        raw.resample(fs, npad="auto", verbose=False)
    
    full_raw, missing = _build_full_montage_raw(
        raw, head_model.electrode_space.info, head_model.electrode_space.info.ch_names
    )
    if missing:
        print(f"Interpolated {len(missing)} missing channels")
    
    full_raw.set_eeg_reference("average", projection=False, verbose=False)
    data = full_raw.get_data()
    
    # ==========================================================================
    # KEY CHANGE: Compute GLOBAL normalization factor
    # ==========================================================================
    if args.normalization_mode == "global":
        # Simple global max
        global_max_abs = float(np.abs(data).max())
    elif args.normalization_mode == "robust_global":
        # Robust: use 99th percentile to reduce artifact sensitivity
        global_max_abs = float(np.percentile(np.abs(data), 99))
    elif args.normalization_mode == "percentile":
        # Very robust: use 95th percentile
        global_max_abs = float(np.percentile(np.abs(data), 95))
    
    print(f"\n{'='*60}")
    print(f"GLOBAL NORMALIZATION")
    print(f"{'='*60}")
    print(f"Mode: {args.normalization_mode}")
    print(f"Global max_abs: {global_max_abs:.6f}")
    print(f"This value will be used for ALL windows (no per-window variation)")
    print(f"{'='*60}\n")
    
    leadfield = torch.from_numpy(head_model.fwd["sol"]["data"]).float()
    all_predictions = []
    segment_summaries = []
    
    for seg_idx, (start_sample, segment) in enumerate(
        _segment_signal(
            data, window_samples, step_samples,
            pad=not args.no_pad_last, max_windows=args.max_windows
        )
    ):
        local_max_abs = float(np.abs(segment).max())
        if local_max_abs <= 0:
            print(f"Skipping window {seg_idx} - zero variance")
            continue
        
        # Use GLOBAL normalization instead of per-window
        eeg_tensor = torch.from_numpy((segment / global_max_abs).astype(np.float32))
        
        pred = _infer_model(
            model, eeg_tensor, args.train_loss,
            max_src_val=1.0,
            max_eeg_val=global_max_abs,  # Use global factor
            leadfield=leadfield,
        )
        
        all_predictions.append({
            'window_idx': seg_idx,
            'start_time': start_sample / fs,
            'end_time': (start_sample + window_samples) / fs,
            'predictions': pred.detach().cpu().numpy(),
            'max_abs': global_max_abs,  # Store global value for reference
            'local_max_abs': local_max_abs,  # Keep track of local for analysis
        })
        
        segment_summaries.append(SegmentSummary(
            index=seg_idx,
            start_sample=start_sample,
            start_time_seconds=start_sample / fs,
            eeg_max_abs=global_max_abs,
            mat_path=output_dir / "segments" / f"segment_{seg_idx:04d}.mat"
        ))
    
    if not all_predictions:
        raise RuntimeError("No valid windows processed")
    
    print(f"Processed {len(all_predictions)} windows with global normalization")
    
    # Prepare animation timeline
    activity_timeline, timestamps = _prepare_animation_timeline_with_smoothing(
        all_predictions, args.overlap_fraction, args.temporal_smoothing
    )
    
    # Compare to what per-window would have been
    local_max_abs_values = [p['local_max_abs'] for p in all_predictions]
    local_ratio = max(local_max_abs_values) / min(local_max_abs_values)
    print(f"\nLocal max_abs would have varied by {local_ratio:.2f}x")
    print(f"Global normalization eliminates this variation!")
    
    # Compute source power for reporting
    source_power = np.sum(activity_timeline ** 2, axis=0)
    power_ratio = np.max(source_power) / np.min(source_power)
    print(f"\nSource power ratio (with global norm): {power_ratio:.2f}x")
    
    # Save animation data
    geom = _load_brain_geometry(head_model)
    
    triangles_list = []
    vertex_offset = 0
    for surf in geom.surfaces:
        adjusted_faces = surf.faces + vertex_offset
        triangles_list.append(adjusted_faces)
        vertex_offset += len(surf.vertices_mm)
    triangles = np.vstack(triangles_list).astype(np.int32)
    
    if len(timestamps) > 1:
        avg_time_between_frames = np.mean(np.diff(timestamps))
        actual_fps = int(1.0 / avg_time_between_frames) if avg_time_between_frames > 0 else 1
    else:
        actual_fps = 1
    
    animation_data = {
        'activity': activity_timeline.astype(np.float32),
        'timestamps': timestamps.astype(np.float32),
        'source_positions': np.ascontiguousarray(geom.positions_mm.astype(np.float32)),
        'triangles': triangles,
        'fps': np.array(actual_fps, dtype=np.int32),
        'normalization_mode': args.normalization_mode,
        'global_max_abs': global_max_abs,
    }
    
    animation_path = output_dir / 'animation_data.npz'
    np.savez_compressed(str(animation_path), **animation_data)
    
    file_size_mb = animation_path.stat().st_size / (1024 * 1024)
    print(f"\nSaved animation data to {animation_path}")
    print(f"  - {len(timestamps)} frames at ~{actual_fps} FPS ({timestamps[-1]:.2f}s duration)")
    print(f"  - File size: {file_size_mb:.2f} MB")
    
    # Save metadata
    metadata = {
        'normalization_mode': args.normalization_mode,
        'global_max_abs': float(global_max_abs),
        'n_windows': len(all_predictions),
        'window_seconds': float(window_seconds),
        'overlap_fraction': float(overlap_fraction),
        'temporal_smoothing': float(args.temporal_smoothing),
        'power_ratio': float(power_ratio),
        'local_max_abs_would_have_been': {
            'min': float(min(local_max_abs_values)),
            'max': float(max(local_max_abs_values)),
            'ratio': float(local_ratio),
        }
    }
    
    with open(output_dir / 'global_norm_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n{'='*60}")
    print("DONE - Global normalization applied successfully!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
