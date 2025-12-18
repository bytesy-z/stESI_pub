#!/usr/bin/env python3
"""
Diagnostic script for EEG inverse solver "flash" artifacts.

This script:
1. Computes Global Field Power (GFP) on raw EEG to check if flashes exist at sensor level
2. Runs inference and detects "flash" frames (high-power outliers in source space)
3. Parses neurologist annotations and compares timestamps
4. Implements Hann-window overlap-add re-weighting to test if stitching causes flashes
5. Reports findings with visualizations

Usage:
    python diagnose_flashes.py /path/to/file.edf --annotation_csv /path/to/annotations.csv
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import mne
import numpy as np
import torch
from scipy.signal import windows as scipy_windows

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_data import HeadModel
from load_data.FolderStructure import FolderStructure
from models.cnn_1d import CNN1Dpl
from plot_1dcnn_inference_heatmap import (
    _ensure_within_repo,
    _find_default_model_checkpoint,
    _infer_model,
)


@dataclass
class FlashEvent:
    """Represents a detected flash in the source activity."""
    frame_idx: int
    timestamp: float  # seconds from recording start
    z_score: float
    power: float
    matched_annotation: Optional[str] = None
    annotation_distance: Optional[float] = None  # seconds


@dataclass
class AnnotationEvent:
    """Represents a neurologist annotation."""
    start_time: float  # seconds from recording start
    end_time: float
    channels: str
    comment: str


def parse_annotation_time(time_str: str, file_start_str: str) -> float:
    """
    Parse annotation time string (HH:MM:SS:mmm format) to seconds from file start.
    
    Args:
        time_str: Time string like "17:12:59:434" (HH:MM:SS:milliseconds)
        file_start_str: File start time like "17:12:33"
    
    Returns:
        Seconds from file start
    """
    # Parse file start (HH:MM:SS)
    parts = file_start_str.strip().split(':')
    file_start = timedelta(
        hours=int(parts[0]),
        minutes=int(parts[1]),
        seconds=int(parts[2])
    )
    
    # Parse event time (HH:MM:SS:mmm or HH:MM:SS)
    # Handle various formats: "17:12:59:434", "17:13:00:4", "17:13:28"
    time_parts = time_str.strip().split(':')
    
    if len(time_parts) >= 3:
        hours = int(time_parts[0])
        minutes = int(time_parts[1])
        
        # Third part could be "SS" or "SS" with optional milliseconds
        seconds_part = time_parts[2]
        
        if len(time_parts) == 4:
            # Format: HH:MM:SS:mmm
            seconds = int(seconds_part)
            ms_str = time_parts[3]
            # Handle variable length milliseconds (e.g., "4" -> 400ms, "434" -> 434ms)
            if len(ms_str) == 1:
                milliseconds = int(ms_str) * 100
            elif len(ms_str) == 2:
                milliseconds = int(ms_str) * 10
            else:
                milliseconds = int(ms_str)
        else:
            # Format: HH:MM:SS
            seconds = int(seconds_part)
            milliseconds = 0
        
        event_time = timedelta(
            hours=hours,
            minutes=minutes,
            seconds=seconds,
            milliseconds=milliseconds
        )
    else:
        raise ValueError(f"Cannot parse time string: {time_str}")
    
    # Return seconds from file start
    return (event_time - file_start).total_seconds()


def load_annotations(csv_path: Path) -> Tuple[List[AnnotationEvent], str]:
    """
    Load neurologist annotations from CSV file.
    
    Returns:
        Tuple of (list of AnnotationEvent, file_start_time_str)
    """
    import csv
    
    annotations = []
    file_start = None
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Get file start from first row
            if file_start is None and row.get('File Start'):
                file_start = row['File Start'].strip()
            
            start_time_str = row.get('Start time', '').strip()
            end_time_str = row.get('End time', '').strip()
            channels = row.get('Channel names', '').strip()
            comment = row.get('Comment', '').strip()
            
            if not start_time_str or not file_start:
                continue
            
            try:
                start_time = parse_annotation_time(start_time_str, file_start)
                end_time = parse_annotation_time(end_time_str, file_start) if end_time_str else start_time
                
                annotations.append(AnnotationEvent(
                    start_time=start_time,
                    end_time=end_time,
                    channels=channels,
                    comment=comment
                ))
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse annotation row: {row} - {e}")
                continue
    
    return annotations, file_start


def compute_gfp(eeg_data: np.ndarray) -> np.ndarray:
    """
    Compute Global Field Power (GFP) from EEG data.
    
    GFP(t) = sqrt(sum_channels(V_i(t) - mean(V(t)))^2)
    
    Args:
        eeg_data: (n_channels, n_samples) array
        
    Returns:
        (n_samples,) array of GFP values
    """
    mean_across_channels = np.mean(eeg_data, axis=0, keepdims=True)
    deviations = eeg_data - mean_across_channels
    gfp = np.sqrt(np.sum(deviations ** 2, axis=0))
    return gfp


def compute_source_power(source_maps: np.ndarray) -> np.ndarray:
    """
    Compute total source power per frame.
    
    S(t) = sum_voxels(source_map(t)^2)
    
    Args:
        source_maps: (n_frames, n_sources) or (n_sources, n_frames) array
        
    Returns:
        (n_frames,) array of power values
    """
    if source_maps.shape[0] > source_maps.shape[1]:
        # Assume (n_sources, n_frames), need to transpose
        source_maps = source_maps.T
    
    # Now (n_frames, n_sources)
    power = np.sum(source_maps ** 2, axis=1)
    return power


def detect_flashes(
    power: np.ndarray,
    timestamps: np.ndarray,
    z_threshold: float = 3.0
) -> List[FlashEvent]:
    """
    Detect flash frames where power exceeds z_threshold standard deviations.
    
    Args:
        power: (n_frames,) array of power values
        timestamps: (n_frames,) array of timestamps in seconds
        z_threshold: Z-score threshold for flash detection
        
    Returns:
        List of FlashEvent objects
    """
    mean_power = np.mean(power)
    std_power = np.std(power)
    
    if std_power < 1e-10:
        return []
    
    z_scores = (power - mean_power) / std_power
    flash_indices = np.where(z_scores > z_threshold)[0]
    
    flashes = []
    for idx in flash_indices:
        flashes.append(FlashEvent(
            frame_idx=int(idx),
            timestamp=float(timestamps[idx]),
            z_score=float(z_scores[idx]),
            power=float(power[idx])
        ))
    
    return flashes


def match_flashes_to_annotations(
    flashes: List[FlashEvent],
    annotations: List[AnnotationEvent],
    tolerance: float = 0.5
) -> Tuple[List[FlashEvent], int]:
    """
    Match detected flashes to neurologist annotations.
    
    Args:
        flashes: List of detected FlashEvent objects
        annotations: List of AnnotationEvent objects
        tolerance: Time tolerance in seconds for matching
        
    Returns:
        Tuple of (updated flashes with matches, number of matches)
    """
    # Get unique annotation times (since same event may be annotated on multiple channels)
    unique_annotation_times = set()
    for ann in annotations:
        # Use the midpoint of the annotation
        midpoint = (ann.start_time + ann.end_time) / 2
        unique_annotation_times.add(midpoint)
    
    annotation_times = np.array(sorted(unique_annotation_times))
    
    n_matches = 0
    for flash in flashes:
        if len(annotation_times) == 0:
            continue
            
        distances = np.abs(annotation_times - flash.timestamp)
        min_idx = np.argmin(distances)
        min_distance = distances[min_idx]
        
        if min_distance <= tolerance:
            n_matches += 1
            # Find the annotation comment
            closest_time = annotation_times[min_idx]
            for ann in annotations:
                mid = (ann.start_time + ann.end_time) / 2
                if abs(mid - closest_time) < 0.01:
                    flash.matched_annotation = ann.comment
                    flash.annotation_distance = min_distance
                    break
    
    return flashes, n_matches


def apply_hann_window_weighting(
    all_predictions: List[dict],
    overlap_fraction: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply Hann window overlap-add weighting to merge overlapping window predictions.
    
    This implements smooth temporal blending instead of abrupt window transitions.
    
    Args:
        all_predictions: List of dicts with 'predictions', 'start_time', 'end_time'
        overlap_fraction: Fractional overlap between windows (0-1)
        
    Returns:
        Tuple of (weighted_activity, timestamps) arrays
    """
    if not all_predictions:
        raise ValueError("No predictions to process")
    
    n_windows = len(all_predictions)
    n_sources = all_predictions[0]['predictions'].shape[0]
    n_timepoints_per_window = all_predictions[0]['predictions'].shape[1]
    
    # Create Hann window for temporal weighting within each window
    hann_window = scipy_windows.hann(n_timepoints_per_window)
    
    # For frame-level output, we'll still produce one frame per window
    # but weight contributions from overlapping windows
    
    # Simple approach: for each output frame, blend with neighboring windows
    # using Hann-based weights
    
    weighted_activity = np.zeros((n_sources, n_windows), dtype=np.float32)
    timestamps = np.zeros(n_windows, dtype=np.float32)
    
    for i, window in enumerate(all_predictions):
        win_pred = window['predictions']  # (n_sources, n_timepoints)
        
        # Find peak activity timepoint
        total_activity = np.abs(win_pred).sum(axis=0)
        peak_idx = np.argmax(total_activity)
        
        # Get activity at peak
        current_activity = win_pred[:, peak_idx]
        
        # Weight from Hann window at the peak position
        current_weight = hann_window[peak_idx]
        
        # Accumulate weighted contributions
        weighted_sum = current_activity * current_weight
        weight_sum = current_weight
        
        # If there's overlap, blend with neighboring windows
        if overlap_fraction > 0:
            # Previous window contribution
            if i > 0:
                prev_pred = all_predictions[i-1]['predictions']
                prev_total = np.abs(prev_pred).sum(axis=0)
                prev_peak_idx = np.argmax(prev_total)
                # Use end of Hann window (fading out)
                fade_weight = hann_window[-1] * overlap_fraction
                weighted_sum += prev_pred[:, prev_peak_idx] * fade_weight
                weight_sum += fade_weight
            
            # Next window contribution  
            if i < n_windows - 1:
                next_pred = all_predictions[i+1]['predictions']
                next_total = np.abs(next_pred).sum(axis=0)
                next_peak_idx = np.argmax(next_total)
                # Use start of Hann window (fading in)
                fade_weight = hann_window[0] * overlap_fraction
                weighted_sum += next_pred[:, next_peak_idx] * fade_weight
                weight_sum += fade_weight
        
        # Normalize by total weight
        if weight_sum > 0:
            weighted_activity[:, i] = weighted_sum / weight_sum
        else:
            weighted_activity[:, i] = current_activity
        
        timestamps[i] = (window['start_time'] + window['end_time']) / 2.0
    
    return weighted_activity, timestamps


def _load_model(
    checkpoint_path: Path,
    n_electrodes: int,
    n_sources: int,
    inter_layer: int = 4096,
    kernel_size: int = 5,
) -> CNN1Dpl:
    """Load 1dCNN model from checkpoint."""
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


def run_diagnostic(
    edf_path: Path,
    annotation_csv: Optional[Path] = None,
    simu_name: str = "mes_debug",
    subject: str = "fsaverage",
    orientation: str = "constrained",
    electrode_montage: str = "standard_1020",
    source_space: str = "ico3",
    model_path: Optional[Path] = None,
    inter_layer: int = 4096,
    kernel_size: int = 5,
    train_loss: str = "cosine",
    window_seconds: Optional[float] = None,
    overlap_fraction: float = 0.5,
    max_windows: Optional[int] = None,
    z_threshold: float = 3.0,
    annotation_tolerance: float = 0.5,
    output_dir: Optional[Path] = None,
) -> dict:
    """
    Run comprehensive flash diagnostic on EDF file.
    
    Returns:
        Dictionary with diagnostic results
    """
    repo_root = REPO_ROOT
    
    if not edf_path.exists():
        raise FileNotFoundError(f"EDF file not found: {edf_path}")
    
    # Setup paths
    simu_root = repo_root / "simulation" / subject
    config_path = (
        simu_root / orientation / electrode_montage / source_space
        / "simu" / simu_name / f"{simu_name}{source_space}_config.json"
    )
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with config_path.open() as f:
        general_config = json.load(f)
    
    general_config["simu_name"] = simu_name
    general_config["electrode_space"]["electrode_montage"] = electrode_montage
    general_config.setdefault("eeg_snr", "infdb")
    
    # Load head model
    folders = FolderStructure(str(simu_root), general_config)
    electrode_space = HeadModel.ElectrodeSpace(folders, general_config)
    source_space_obj = HeadModel.SourceSpace(folders, general_config)
    head_model = HeadModel.HeadModel(electrode_space, source_space_obj, folders, subject)
    
    # Setup window parameters
    fs = float(general_config["rec_info"]["fs"])
    if window_seconds is None:
        window_samples = int(general_config["rec_info"]["n_times"])
        window_seconds = window_samples / fs
    else:
        window_samples = max(1, int(round(window_seconds * fs)))
    
    step_samples = max(1, int(round(window_samples * (1.0 - overlap_fraction))))
    
    # Load model
    if model_path:
        model_checkpoint = Path(model_path)
    else:
        token = f"{simu_name}{source_space}_"
        model_checkpoint = _find_default_model_checkpoint(token, inter_layer)
        if model_checkpoint is None:
            raise FileNotFoundError("Cannot find model checkpoint")
    
    model = _load_model(
        model_checkpoint,
        head_model.electrode_space.n_electrodes,
        head_model.source_space.n_sources,
        inter_layer,
        kernel_size,
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
    # DIAGNOSTIC 1: Compute GFP on raw sensor data
    # ==========================================================================
    print("\n" + "="*60)
    print("DIAGNOSTIC 1: Global Field Power (GFP) Analysis")
    print("="*60)
    
    gfp = compute_gfp(data)
    gfp_times = np.arange(len(gfp)) / fs
    
    # Detect GFP spikes
    gfp_mean = np.mean(gfp)
    gfp_std = np.std(gfp)
    gfp_z = (gfp - gfp_mean) / gfp_std
    gfp_spike_samples = np.where(gfp_z > z_threshold)[0]
    gfp_spike_times = gfp_spike_samples / fs
    
    print(f"GFP Statistics:")
    print(f"  Mean: {gfp_mean:.6f}")
    print(f"  Std:  {gfp_std:.6f}")
    print(f"  Max:  {np.max(gfp):.6f}")
    print(f"  Number of GFP spikes (z > {z_threshold}): {len(gfp_spike_times)}")
    
    # ==========================================================================
    # DIAGNOSTIC 2: Run inference and detect source-space flashes
    # ==========================================================================
    print("\n" + "="*60)
    print("DIAGNOSTIC 2: Source Space Flash Detection")
    print("="*60)
    
    leadfield = torch.from_numpy(head_model.fwd["sol"]["data"]).float()
    all_predictions = []
    
    for seg_idx, (start_sample, segment) in enumerate(
        _segment_signal(data, window_samples, step_samples, pad=True, max_windows=max_windows)
    ):
        max_abs = float(np.abs(segment).max())
        if max_abs <= 0:
            continue
        
        eeg_tensor = torch.from_numpy((segment / max_abs).astype(np.float32))
        pred = _infer_model(
            model, eeg_tensor, train_loss,
            max_src_val=1.0, max_eeg_val=max_abs, leadfield=leadfield
        )
        
        all_predictions.append({
            'window_idx': seg_idx,
            'start_time': start_sample / fs,
            'end_time': (start_sample + window_samples) / fs,
            'predictions': pred.detach().cpu().numpy(),
            'max_abs': max_abs,
        })
    
    print(f"Processed {len(all_predictions)} windows")
    
    # Extract activity timeline (original method - no Hann weighting)
    n_windows = len(all_predictions)
    n_sources = all_predictions[0]['predictions'].shape[0]
    
    activity_original = np.zeros((n_sources, n_windows), dtype=np.float32)
    timestamps = np.zeros(n_windows, dtype=np.float32)
    
    for i, window in enumerate(all_predictions):
        win_pred = window['predictions']
        total_activity = np.abs(win_pred).sum(axis=0)
        peak_idx = np.argmax(total_activity)
        activity_original[:, i] = win_pred[:, peak_idx]
        timestamps[i] = (window['start_time'] + window['end_time']) / 2.0
    
    # Compute source power per frame
    source_power_original = compute_source_power(activity_original)
    
    # Detect flashes in original
    flashes_original = detect_flashes(source_power_original, timestamps, z_threshold)
    
    print(f"\nOriginal Method (no Hann weighting):")
    print(f"  Number of detected flashes (z > {z_threshold}): {len(flashes_original)}")
    
    if flashes_original:
        print(f"  Flash timestamps (first 10):")
        for flash in flashes_original[:10]:
            print(f"    {flash.timestamp:.2f}s (z={flash.z_score:.2f})")
    
    # ==========================================================================
    # DIAGNOSTIC 3: Apply Hann window weighting and re-detect
    # ==========================================================================
    print("\n" + "="*60)
    print("DIAGNOSTIC 3: Hann Window Weighted Re-merge")
    print("="*60)
    
    activity_hann, timestamps_hann = apply_hann_window_weighting(
        all_predictions, overlap_fraction
    )
    
    source_power_hann = compute_source_power(activity_hann)
    flashes_hann = detect_flashes(source_power_hann, timestamps_hann, z_threshold)
    
    print(f"Hann-weighted Method:")
    print(f"  Number of detected flashes (z > {z_threshold}): {len(flashes_hann)}")
    
    if flashes_hann:
        print(f"  Flash timestamps (first 10):")
        for flash in flashes_hann[:10]:
            print(f"    {flash.timestamp:.2f}s (z={flash.z_score:.2f})")
    
    # ==========================================================================
    # DIAGNOSTIC 4: Compare to neurologist annotations
    # ==========================================================================
    annotations = []
    n_matches_original = 0
    n_matches_hann = 0
    
    if annotation_csv and annotation_csv.exists():
        print("\n" + "="*60)
        print("DIAGNOSTIC 4: Comparison with Neurologist Annotations")
        print("="*60)
        
        annotations, file_start = load_annotations(annotation_csv)
        
        # Get unique annotation events (same event may appear on multiple channels)
        unique_times = set()
        unique_annotations = []
        for ann in annotations:
            mid = (ann.start_time + ann.end_time) / 2
            key = f"{mid:.2f}"
            if key not in unique_times:
                unique_times.add(key)
                unique_annotations.append(ann)
        
        print(f"Loaded {len(unique_annotations)} unique annotation events")
        print(f"Recording starts at: {file_start}")
        print(f"Annotation time range: {min(a.start_time for a in annotations):.1f}s - {max(a.end_time for a in annotations):.1f}s")
        
        # Match flashes to annotations
        flashes_original, n_matches_original = match_flashes_to_annotations(
            flashes_original, unique_annotations, annotation_tolerance
        )
        flashes_hann, n_matches_hann = match_flashes_to_annotations(
            flashes_hann, unique_annotations, annotation_tolerance
        )
        
        print(f"\nOriginal method flashes matching annotations: {n_matches_original}/{len(flashes_original)}")
        print(f"Hann-weighted flashes matching annotations: {n_matches_hann}/{len(flashes_hann)}")
        
        # Show matched flashes
        if n_matches_original > 0:
            print(f"\nMatched flashes (original method):")
            for flash in flashes_original:
                if flash.matched_annotation:
                    print(f"  {flash.timestamp:.2f}s -> {flash.matched_annotation} (Δ={flash.annotation_distance:.3f}s)")
        
        # Check GFP spikes against annotations
        gfp_matches = 0
        for spike_time in gfp_spike_times:
            for ann in unique_annotations:
                if abs((ann.start_time + ann.end_time) / 2 - spike_time) <= annotation_tolerance:
                    gfp_matches += 1
                    break
        
        print(f"\nGFP spikes matching annotations: {gfp_matches}/{len(gfp_spike_times)}")
    
    # ==========================================================================
    # DIAGNOSTIC 5: Cross-check GFP vs Source Power correlation
    # ==========================================================================
    print("\n" + "="*60)
    print("DIAGNOSTIC 5: GFP vs Source Power Correlation")
    print("="*60)
    
    # Resample GFP to match frame timestamps
    gfp_at_frames = np.interp(timestamps, gfp_times, gfp)
    
    # Compute correlation
    corr_original = np.corrcoef(gfp_at_frames, source_power_original)[0, 1]
    corr_hann = np.corrcoef(gfp_at_frames, source_power_hann)[0, 1]
    
    print(f"Correlation between GFP and source power:")
    print(f"  Original method: {corr_original:.4f}")
    print(f"  Hann-weighted:   {corr_hann:.4f}")
    
    # Check if source flashes occur when GFP is elevated
    source_flash_times = [f.timestamp for f in flashes_original]
    gfp_at_source_flashes = np.interp(source_flash_times, gfp_times, gfp_z) if source_flash_times else []
    
    if len(gfp_at_source_flashes) > 0:
        gfp_elevated_at_flashes = np.sum(np.array(gfp_at_source_flashes) > 1.0)
        print(f"\nOf {len(flashes_original)} source flashes:")
        print(f"  {gfp_elevated_at_flashes} occur when GFP is elevated (z > 1)")
        print(f"  {len(flashes_original) - gfp_elevated_at_flashes} occur when GFP is normal")
        
        if gfp_elevated_at_flashes < len(flashes_original) * 0.5:
            print("\n⚠️  WARNING: Many source flashes occur when GFP is normal!")
            print("   This suggests the inverse solver or window stitching may be causing artifacts.")
    
    # ==========================================================================
    # Generate plots
    # ==========================================================================
    if output_dir is None:
        output_dir = repo_root / "results" / "flash_diagnostic" / edf_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: GFP and source power over time
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    
    # GFP
    ax = axes[0]
    ax.plot(gfp_times, gfp, 'b-', alpha=0.7, linewidth=0.5)
    ax.axhline(gfp_mean + z_threshold * gfp_std, color='r', linestyle='--', label=f'z={z_threshold} threshold')
    ax.scatter(gfp_spike_times, gfp[gfp_spike_samples], c='r', s=10, zorder=5, label='GFP spikes')
    ax.set_ylabel('GFP')
    ax.set_title('Global Field Power (Sensor Level)')
    ax.legend(loc='upper right')
    
    # Source power (original)
    ax = axes[1]
    ax.plot(timestamps, source_power_original, 'g-', alpha=0.7, linewidth=1)
    threshold_original = np.mean(source_power_original) + z_threshold * np.std(source_power_original)
    ax.axhline(threshold_original, color='r', linestyle='--', label=f'z={z_threshold} threshold')
    flash_times_orig = [f.timestamp for f in flashes_original]
    flash_powers_orig = [f.power for f in flashes_original]
    ax.scatter(flash_times_orig, flash_powers_orig, c='r', s=30, zorder=5, label='Flashes')
    ax.set_ylabel('Source Power')
    ax.set_title('Source Power (Original Method - No Weighting)')
    ax.legend(loc='upper right')
    
    # Source power (Hann-weighted)
    ax = axes[2]
    ax.plot(timestamps_hann, source_power_hann, 'm-', alpha=0.7, linewidth=1)
    threshold_hann = np.mean(source_power_hann) + z_threshold * np.std(source_power_hann)
    ax.axhline(threshold_hann, color='r', linestyle='--', label=f'z={z_threshold} threshold')
    flash_times_hann = [f.timestamp for f in flashes_hann]
    flash_powers_hann = [f.power for f in flashes_hann]
    ax.scatter(flash_times_hann, flash_powers_hann, c='r', s=30, zorder=5, label='Flashes')
    ax.set_ylabel('Source Power')
    ax.set_xlabel('Time (s)')
    ax.set_title('Source Power (Hann Window Weighted)')
    ax.legend(loc='upper right')
    
    # Add annotation markers if available
    if annotations:
        for ax in axes:
            for ann in annotations[:50]:  # Limit to avoid clutter
                mid = (ann.start_time + ann.end_time) / 2
                ax.axvline(mid, color='orange', alpha=0.3, linewidth=0.5)
    
    plt.tight_layout()
    plot_path = output_dir / 'diagnostic_timeseries.png'
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"\nSaved timeseries plot to: {plot_path}")
    
    # Plot 2: Histogram of z-scores
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    z_original = (source_power_original - np.mean(source_power_original)) / np.std(source_power_original)
    z_hann = (source_power_hann - np.mean(source_power_hann)) / np.std(source_power_hann)
    
    axes[0].hist(z_original, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[0].axvline(z_threshold, color='r', linestyle='--', label=f'z={z_threshold}')
    axes[0].set_xlabel('Z-score')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Source Power Z-scores (Original)')
    axes[0].legend()
    
    axes[1].hist(z_hann, bins=50, alpha=0.7, color='magenta', edgecolor='black')
    axes[1].axvline(z_threshold, color='r', linestyle='--', label=f'z={z_threshold}')
    axes[1].set_xlabel('Z-score')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Source Power Z-scores (Hann Weighted)')
    axes[1].legend()
    
    plt.tight_layout()
    hist_path = output_dir / 'diagnostic_histogram.png'
    plt.savefig(hist_path, dpi=150)
    plt.close()
    print(f"Saved histogram plot to: {hist_path}")
    
    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    results = {
        'edf_path': str(edf_path),
        'n_windows': n_windows,
        'window_seconds': window_seconds,
        'overlap_fraction': overlap_fraction,
        'z_threshold': z_threshold,
        'gfp': {
            'n_spikes': len(gfp_spike_times),
            'spike_times': gfp_spike_times.tolist()[:20],
        },
        'original_method': {
            'n_flashes': len(flashes_original),
            'flash_times': [f.timestamp for f in flashes_original],
            'n_annotation_matches': n_matches_original,
        },
        'hann_weighted': {
            'n_flashes': len(flashes_hann),
            'flash_times': [f.timestamp for f in flashes_hann],
            'n_annotation_matches': n_matches_hann,
        },
        'correlations': {
            'gfp_vs_source_original': corr_original,
            'gfp_vs_source_hann': corr_hann,
        },
        'n_annotations': len(annotations),
        'output_dir': str(output_dir),
    }
    
    # Print interpretation
    print(f"\n📊 Results:")
    print(f"   GFP spikes at sensor level: {len(gfp_spike_times)}")
    print(f"   Source flashes (original):  {len(flashes_original)}")
    print(f"   Source flashes (Hann):      {len(flashes_hann)}")
    
    if len(flashes_hann) < len(flashes_original):
        reduction = (1 - len(flashes_hann) / max(1, len(flashes_original))) * 100
        print(f"\n✅ Hann weighting reduced flashes by {reduction:.1f}%")
        print("   This suggests window stitching contributed to the flash artifacts.")
    elif len(flashes_hann) >= len(flashes_original):
        print(f"\n⚠️  Hann weighting did not reduce flashes significantly.")
        print("   The flashes may be from the inverse solver itself or genuine signals.")
    
    if corr_original < 0.5:
        print(f"\n⚠️  Low correlation ({corr_original:.2f}) between GFP and source power")
        print("   Source flashes may not correspond to real EEG amplitude changes.")
        print("   This suggests potential inverse solver artifacts.")
    else:
        print(f"\n✅ Reasonable correlation ({corr_original:.2f}) between GFP and source power")
        print("   Source activity generally tracks with EEG amplitude.")
    
    if annotation_csv and annotations:
        if n_matches_original < len(flashes_original) * 0.3:
            print(f"\n⚠️  Only {n_matches_original}/{len(flashes_original)} flashes match neurologist annotations")
            print("   Most flashes do NOT correspond to clinical events.")
        else:
            print(f"\n✅ {n_matches_original}/{len(flashes_original)} flashes match neurologist annotations")
    
    # Save results JSON
    results_path = output_dir / 'diagnostic_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to: {results_path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Diagnose EEG inverse solver flash artifacts")
    parser.add_argument("edf_path", type=str, help="Path to EDF file")
    parser.add_argument("--annotation_csv", type=str, help="Path to neurologist annotation CSV")
    parser.add_argument("--simu_name", default="mes_debug")
    parser.add_argument("--subject", default="fsaverage")
    parser.add_argument("--source_space", default="ico3")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--inter_layer", type=int, default=4096)
    parser.add_argument("--window_seconds", type=float, default=None)
    parser.add_argument("--overlap_fraction", type=float, default=0.5)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--z_threshold", type=float, default=3.0)
    parser.add_argument("--annotation_tolerance", type=float, default=0.5)
    parser.add_argument("--output_dir", type=str, default=None)
    
    args = parser.parse_args()
    
    run_diagnostic(
        edf_path=Path(args.edf_path),
        annotation_csv=Path(args.annotation_csv) if args.annotation_csv else None,
        simu_name=args.simu_name,
        subject=args.subject,
        source_space=args.source_space,
        model_path=Path(args.model_path) if args.model_path else None,
        inter_layer=args.inter_layer,
        window_seconds=args.window_seconds,
        overlap_fraction=args.overlap_fraction,
        max_windows=args.max_windows,
        z_threshold=args.z_threshold,
        annotation_tolerance=args.annotation_tolerance,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )


if __name__ == "__main__":
    main()
