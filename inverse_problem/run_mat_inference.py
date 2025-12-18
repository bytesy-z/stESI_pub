#!/usr/bin/env python3
"""Run inference on simulation MAT files with ground truth evaluation."""

from __future__ import annotations

import argparse
import json
import sys
import webbrowser
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import mne
import numpy as np
import torch
from scipy.io import loadmat, savemat

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from load_data import HeadModel
from load_data.FolderStructure import FolderStructure
from models.cnn_1d import CNN1Dpl
from utils import utl
from utils import utl_metrics as met
from plot_1dcnn_inference_heatmap import (
    BrainGeometry,
    _ensure_within_repo,
    _find_default_model_checkpoint,
    _infer_model,
    _load_brain_geometry,
    _save_interactive_heatmap,
)


@dataclass
class MetricsResult:
    """Container for evaluation metrics."""
    nmse: float = 0.0
    auc: float = 0.0
    localization_error_mm: float = 0.0
    time_error_ms: float = 0.0
    seed_indices: List[int] = field(default_factory=list)
    estimated_seed_indices: List[int] = field(default_factory=list)
    peak_times_gt: List[int] = field(default_factory=list)
    peak_times_pred: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict:
        return {
            "nmse": self.nmse,
            "auc": self.auc,
            "localization_error_mm": self.localization_error_mm,
            "time_error_ms": self.time_error_ms,
            "seed_indices": self.seed_indices,
            "estimated_seed_indices": self.estimated_seed_indices,
            "peak_times_gt": self.peak_times_gt,
            "peak_times_pred": self.peak_times_pred,
        }


@dataclass
class SegmentSummary:
    index: int
    start_sample: int
    start_time_seconds: float
    eeg_max_abs: float
    mat_path: Path


def _apply_exponential_smoothing(
    activity_timeline: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """
    Apply exponential moving average (EMA) smoothing to source activity timeline.
    
    Uses bidirectional smoothing (forward + backward pass) to eliminate phase lag.
    Recommended alpha=0.5-0.7 for light visualization smoothing.
    WARNING: Heavy smoothing (alpha<0.3) degrades localization accuracy by ~80%.
    
    Args:
        activity_timeline: (n_sources, n_frames) array
        alpha: Smoothing parameter (0-1). Higher = less smoothing.
        
    Returns:
        smoothed: (n_sources, n_frames) array
    """
    if alpha < 0 or alpha > 1:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    
    smoothed = activity_timeline.copy()
    n_sources, n_frames = smoothed.shape
    
    # Forward pass
    for t in range(1, n_frames):
        smoothed[:, t] = alpha * smoothed[:, t] + (1 - alpha) * smoothed[:, t - 1]
    
    # Backward pass (eliminate phase lag)
    for t in range(n_frames - 2, -1, -1):
        smoothed[:, t] = alpha * smoothed[:, t] + (1 - alpha) * smoothed[:, t + 1]
    
    return smoothed


def _load_model(
    checkpoint_path: Path,
    n_electrodes: int,
    n_sources: int,
    inter_layer: int,
    kernel_size: int,
) -> CNN1Dpl:
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


def _load_mat_eeg(mat_path: Path) -> np.ndarray:
    """Load EEG data from MAT file."""
    data = loadmat(str(mat_path))
    
    # Handle different MAT file formats
    if "eeg_data" in data:
        eeg_data = data["eeg_data"]
        if eeg_data.dtype.names and "EEG" in eeg_data.dtype.names:
            eeg = eeg_data[0, 0]["EEG"]
        else:
            eeg = eeg_data
    elif "EEG" in data:
        eeg = data["EEG"]
    else:
        raise ValueError(f"Cannot find EEG data in MAT file. Keys: {list(data.keys())}")
    
    return np.asarray(eeg, dtype=np.float64)


def _load_mat_source(mat_path: Path, n_sources: int = None, metadata_path: Path = None) -> Optional[np.ndarray]:
    """Load source data from MAT file if available.
    
    Args:
        mat_path: Path to the source MAT file
        n_sources: Total number of sources in the full source space
        metadata_path: Path to metadata JSON file containing active source indices
    
    Returns:
        Full source data array (n_sources, n_times) or None
    """
    if not mat_path.exists():
        return None
    
    data = loadmat(str(mat_path))
    
    # Handle different MAT file formats for source data
    if "Jact" in data:
        jact_data = data["Jact"]
        if jact_data.dtype.names and "Jact" in jact_data.dtype.names:
            source_values = np.asarray(jact_data[0, 0]["Jact"], dtype=np.float64)
        else:
            source_values = np.asarray(jact_data, dtype=np.float64)
    elif "src_data" in data:
        source_values = np.asarray(data["src_data"], dtype=np.float64)
    else:
        return None
    
    # If we have metadata with active source indices, expand to full source space
    if metadata_path is not None and metadata_path.exists() and n_sources is not None:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Get all active source indices from all patches
        act_src_indices = []
        n_patches = metadata.get("n_patch", 1)
        for p in range(n_patches):
            patch_key = f"patch_{p + 1}"
            if patch_key in metadata.get("act_src", {}):
                act_src_indices.extend(metadata["act_src"][patch_key])
        
        act_src_indices = sorted(set(act_src_indices))
        
        # Check if source values match the number of active indices
        n_times = source_values.shape[1] if len(source_values.shape) > 1 else source_values.shape[0]
        n_active = len(act_src_indices)
        
        if source_values.shape[0] == n_active:
            # Expand to full source space
            full_source = np.zeros((n_sources, n_times), dtype=np.float64)
            full_source[act_src_indices, :] = source_values
            return full_source
        elif source_values.shape[0] == n_sources:
            # Already full source space
            return source_values
        else:
            print(f"Warning: Source shape {source_values.shape} doesn't match expected active sources ({n_active}) or full space ({n_sources})")
            # Return as-is if we can't reconcile
            return source_values
    
    return source_values


def _find_source_file_for_eeg(eeg_path: Path, simu_root: Path) -> Optional[Path]:
    """Find the corresponding source file for an EEG file in simulation folder."""
    # Extract the sample ID from filename (e.g., "1_eeg.mat" -> "1")
    name = eeg_path.stem
    if "_eeg" in name:
        sample_id = name.replace("_eeg", "")
    else:
        sample_id = name
    
    # Search for source file in standard simulation paths
    possible_paths = [
        simu_root / "sources" / "Jact" / f"{sample_id}_src_act.mat",
        simu_root / "sources" / f"{sample_id}_src.mat",
        eeg_path.parent.parent / "sources" / "Jact" / f"{sample_id}_src_act.mat",
        eeg_path.parent.parent.parent / "sources" / "Jact" / f"{sample_id}_src_act.mat",
    ]
    
    for src_path in possible_paths:
        if src_path.exists():
            return src_path
    
    return None


def _find_metadata_file_for_eeg(eeg_path: Path, simu_root: Path) -> Optional[Path]:
    """Find metadata JSON file for a simulation sample."""
    name = eeg_path.stem
    if "_eeg" in name:
        sample_id = name.replace("_eeg", "")
    else:
        sample_id = name
    
    possible_paths = [
        simu_root / "md" / f"{sample_id}_md_json_flie.json",  # Note: typo in original data
        simu_root / "md" / f"{sample_id}_md.json",
        simu_root / "metadata" / f"{sample_id}_metadata.json",
        eeg_path.parent.parent / "md" / f"{sample_id}_md_json_flie.json",
        eeg_path.parent.parent.parent / "md" / f"{sample_id}_md_json_flie.json",
    ]
    
    for md_path in possible_paths:
        if md_path.exists():
            return md_path
    
    return None


def _build_neighbors_fallback(positions: np.ndarray, k: int = 6) -> np.ndarray:
    """Build k-nearest-neighbor graph from source positions."""
    n_sources = positions.shape[0]
    k = min(k, max(n_sources - 1, 1))
    if k <= 0:
        return np.full((n_sources, 1), -1, dtype=int)
    
    diff = positions[:, None, :] - positions[None, :, :]
    dist_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dist_sq, np.inf)
    idx = np.argpartition(dist_sq, range(k), axis=1)[:, :k]
    return idx + 1  # 1-based indexing for compatibility


def _compute_metrics(
    pred: torch.Tensor,
    gt: torch.Tensor,
    positions: np.ndarray,
    neighbors: np.ndarray,
    fs: float,
    seeds: Optional[List[int]] = None,
    patches: Optional[List[List[int]]] = None,
) -> MetricsResult:
    """Compute evaluation metrics between prediction and ground truth."""
    result = MetricsResult()
    
    n_times = gt.shape[1]
    t_vec = np.arange(0, n_times / fs, 1 / fs)
    spos = torch.from_numpy(positions)
    
    # Normalize for metric computation
    gt_norm = gt.clone()
    gt_max = gt_norm.abs().max()
    if gt_max > 0:
        gt_norm = gt_norm / gt_max
    
    pred_norm = pred.clone()
    pred_max = pred_norm.abs().max()
    if pred_max > 0:
        pred_norm = pred_norm / pred_max
    
    # If no seeds provided, find them from GT (sources with highest activity)
    if seeds is None:
        # Find sources with significant activity
        max_activity = gt.abs().max(dim=1).values
        threshold = 0.1 * max_activity.max()
        active_mask = max_activity > threshold
        seeds = torch.where(active_mask)[0].tolist()
        
        if not seeds:
            # Fallback: use source with maximum activity
            seeds = [torch.argmax(max_activity).item()]
    
    # Default patches to seed neighborhoods if not provided
    if patches is None:
        patches = []
        for s in seeds:
            patch = utl.get_patch(order=3, idx=s, neighbors=neighbors)
            patches.append(patch.tolist() if hasattr(patch, 'tolist') else list(patch))
    
    result.seed_indices = seeds
    all_active_sources = [s for p in patches for s in p]
    
    le_sum = 0.0
    te_sum = 0.0
    nmse_sum = 0.0
    auc_sum = 0.0
    
    for kk, s in enumerate(seeds):
        # Get other sources to exclude from eval zone
        if kk < len(patches):
            other_sources = np.setdiff1d(all_active_sources, patches[kk])
        else:
            other_sources = np.array([])
        
        # Find time of max activity in GT
        t_eval_gt = torch.argmax(gt[s, :].abs()).item()
        result.peak_times_gt.append(t_eval_gt)
        
        # Find estimated seed in neighboring area
        eval_zone = utl.get_patch(order=5, idx=s, neighbors=neighbors)
        eval_zone = np.setdiff1d(eval_zone, other_sources, assume_unique=True)
        if eval_zone.size == 0:
            eval_zone = np.array([s])
        
        # Tighter zone for better localization
        tighter_zone = utl.get_patch(order=2, idx=s, neighbors=neighbors)
        tighter_zone = np.setdiff1d(tighter_zone, other_sources, assume_unique=True)
        if tighter_zone.size > 0:
            eval_zone = tighter_zone
        
        if s not in eval_zone:
            eval_zone = np.append(eval_zone, s)
        
        eval_zone = np.unique(eval_zone).astype(int)
        eval_zone_idx = torch.as_tensor(eval_zone, device=pred.device, dtype=torch.long)
        
        # Find estimated seed
        s_hat = eval_zone_idx[torch.argmax(pred[eval_zone_idx, t_eval_gt].abs())].item()
        result.estimated_seed_indices.append(s_hat)
        
        # Find time of max activity in prediction
        t_eval_pred = torch.argmax(pred[s_hat, :].abs()).item()
        result.peak_times_pred.append(t_eval_pred)
        
        # Compute metrics
        le_sum += torch.sqrt(((spos[s, :] - spos[s_hat, :]) ** 2).sum()).item()
        te_sum += np.abs(t_vec[t_eval_gt] - t_vec[t_eval_pred]) if t_eval_gt < len(t_vec) and t_eval_pred < len(t_vec) else 0
        
        # AUC
        auc_val = met.auc_t(gt, pred, t_eval_gt, thresh=True, act_thresh=0.0)
        auc_sum += auc_val
        
        # nMSE at peak time
        nmse_tmp = ((gt_norm[:, t_eval_gt] - pred_norm[:, t_eval_gt]) ** 2).mean().item()
        nmse_sum += nmse_tmp
    
    n_seeds = len(seeds)
    result.localization_error_mm = (le_sum / n_seeds) * 1e3  # Convert m to mm
    result.time_error_ms = (te_sum / n_seeds) * 1e3  # Convert s to ms
    result.nmse = nmse_sum / n_seeds
    result.auc = auc_sum / n_seeds
    
    return result


def _segment_signal(
    data: np.ndarray,
    window_samples: int,
    step_samples: int,
    pad: bool,
    max_windows: Optional[int],
) -> Iterable[Tuple[int, np.ndarray]]:
    """Segment signal into windows."""
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


def _prepare_animation_timeline(
    all_predictions: List[dict],
) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare animation timeline with one frame per window."""
    if not all_predictions:
        raise ValueError("all_predictions list is empty")
    
    n_windows = len(all_predictions)
    n_sources = all_predictions[0]['predictions'].shape[0]
    
    activity_timeline = np.zeros((n_sources, n_windows), dtype=np.float32)
    timestamps = np.zeros(n_windows, dtype=np.float32)
    
    for i, window in enumerate(all_predictions):
        win_pred = window['predictions']
        total_activity = np.abs(win_pred).sum(axis=0)
        peak_idx = np.argmax(total_activity)
        activity_timeline[:, i] = win_pred[:, peak_idx]
        timestamps[i] = (window['start_time'] + window['end_time']) / 2.0
    
    return activity_timeline, timestamps


def _prepare_output_dir(base_dir: Optional[str], repo_root: Path, mat_path: Path) -> Path:
    """Prepare output directory for results."""
    if base_dir:
        out_dir = Path(base_dir).expanduser()
    else:
        out_dir = repo_root / "results" / "mat_inference" / mat_path.stem
    out_dir = _ensure_within_repo(out_dir, repo_root)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "segments").mkdir(exist_ok=True)
    return out_dir


def _save_segment_to_mat(segment: np.ndarray, path: Path) -> None:
    """Save segment to MAT file."""
    savemat(str(path), {"eeg_data": {"EEG": segment.astype(np.float32)}})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 1dCNN inference on simulation MAT files.")
    parser.add_argument("mat_path", type=str, help="Path to the MAT file to process.")
    parser.add_argument(
        "--simu_name",
        default="mes_debug",
        help="Simulation name matching the training configuration.",
    )
    parser.add_argument(
        "--subject",
        default="fsaverage",
        help="Subject folder under simulation/ containing the head model assets.",
    )
    parser.add_argument(
        "--orientation",
        default="constrained",
        choices=["constrained", "unconstrained"],
        help="Source orientation of the head model.",
    )
    parser.add_argument(
        "--electrode_montage",
        default="standard_1020",
        help="Electrode montage name used during training.",
    )
    parser.add_argument(
        "--source_space",
        default="ico3",
        help="Source space name (e.g. ico3).",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Explicit path to 1dCNN checkpoint.",
    )
    parser.add_argument(
        "--source_file",
        type=str,
        default=None,
        help="Path to ground truth source MAT file (for computing metrics).",
    )
    parser.add_argument(
        "--inter_layer",
        type=int,
        default=4096,
        help="Intermediate channel size used during training.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=5,
        help="1dCNN kernel size used during training.",
    )
    parser.add_argument(
        "--train_loss",
        default="cosine",
        help="Loss used during training.",
    )
    parser.add_argument(
        "--window_samples",
        type=int,
        default=None,
        help="Window size in samples. Defaults to n_times from config.",
    )
    parser.add_argument(
        "--overlap_fraction",
        type=float,
        default=0.5,
        help="Fractional overlap between consecutive windows.",
    )
    parser.add_argument(
        "--max_windows",
        type=int,
        default=None,
        help="Optional cap on number of windows to process.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for outputs.",
    )
    parser.add_argument(
        "--open_plot",
        action="store_true",
        help="Open the resulting plot in browser.",
    )
    parser.add_argument(
        "--no_pad_last",
        action="store_true",
        help="Drop trailing window instead of padding.",
    )
    parser.add_argument(
        "--use_global_norm",
        action="store_true",
        help="If set, use global 99th percentile normalization instead of per-window max. Reduces artificial amplitude variations.",
    )
    parser.add_argument(
        "--smoothing_alpha",
        type=float,
        default=None,
        help="Optional temporal smoothing parameter (0-1). Use 0.5-0.7 for light visualization smoothing. WARNING: heavy smoothing (<0.3) degrades localization accuracy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    repo_root = REPO_ROOT
    mat_path = Path(args.mat_path).expanduser().resolve()
    if not mat_path.exists():
        raise FileNotFoundError(f"MAT file not found: {mat_path}")
    
    # Load simulation config
    simu_root = repo_root / "simulation" / args.subject
    simu_root = _ensure_within_repo(simu_root, repo_root)
    
    config_path = (
        simu_root
        / args.orientation
        / args.electrode_montage
        / args.source_space
        / "simu"
        / args.simu_name
        / f"{args.simu_name}{args.source_space}_config.json"
    )
    config_path = _ensure_within_repo(config_path, repo_root)
    if not config_path.exists():
        raise FileNotFoundError(f"Could not locate simulation config: {config_path}")
    
    with config_path.open() as f:
        general_config = json.load(f)
    
    general_config["simu_name"] = args.simu_name
    general_config["electrode_space"]["electrode_montage"] = args.electrode_montage
    general_config.setdefault("eeg_snr", "infdb")
    
    # Load head model
    folders = FolderStructure(str(simu_root), general_config)
    electrode_space = HeadModel.ElectrodeSpace(folders, general_config)
    source_space = HeadModel.SourceSpace(folders, general_config)
    head_model = HeadModel.HeadModel(electrode_space, source_space, folders, args.subject)
    
    fs = float(general_config["rec_info"]["fs"])
    n_times_config = int(general_config["rec_info"]["n_times"])
    window_samples = args.window_samples if args.window_samples else n_times_config
    
    overlap_fraction = min(max(args.overlap_fraction, 0.0), 0.95)
    step_samples = max(1, int(round(window_samples * (1.0 - overlap_fraction))))
    
    output_dir = _prepare_output_dir(args.output_dir, repo_root, mat_path)
    segments_dir = output_dir / "segments"
    
    # Load model
    if args.model_path:
        model_path = _ensure_within_repo(Path(args.model_path), repo_root)
    else:
        token = f"{args.simu_name}{args.source_space}_"
        checkpoint = _find_default_model_checkpoint(token, args.inter_layer)
        if checkpoint is None:
            raise FileNotFoundError("Unable to locate 1dCNN checkpoint. Specify --model_path explicitly.")
        model_path = _ensure_within_repo(checkpoint, repo_root)
    
    model = _load_model(
        model_path,
        head_model.electrode_space.n_electrodes,
        head_model.source_space.n_sources,
        args.inter_layer,
        args.kernel_size,
    )
    
    print(f"Loaded model from: {model_path}")
    print(f"Processing MAT file: {mat_path}")
    
    # Load EEG data
    eeg_data = _load_mat_eeg(mat_path)
    print(f"EEG data shape: {eeg_data.shape}")
    
    # Load ground truth source if available
    gt_source = None
    source_file = None
    metadata_file = None
    simu_folder = simu_root / args.orientation / args.electrode_montage / args.source_space / "simu" / args.simu_name
    
    if args.source_file:
        source_file = Path(args.source_file)
    else:
        # Try to find corresponding source file
        source_file = _find_source_file_for_eeg(mat_path, simu_folder)
    
    # Find metadata file for proper source expansion
    metadata_file = _find_metadata_file_for_eeg(mat_path, simu_folder)
    
    if source_file and source_file.exists():
        n_sources = head_model.source_space.n_sources
        gt_source = _load_mat_source(source_file, n_sources=n_sources, metadata_path=metadata_file)
        if gt_source is not None:
            print(f"Loaded ground truth source from: {source_file}")
            print(f"Source data shape: {gt_source.shape}")
            if metadata_file:
                print(f"Using metadata from: {metadata_file}")
    
    # Build neighbors for metrics
    neighbors = _build_neighbors_fallback(source_space.positions)
    
    leadfield = torch.from_numpy(head_model.fwd["sol"]["data"]).float()
    
    # Compute global normalization factor (99th percentile)
    if args.use_global_norm:
        global_max_abs = float(np.percentile(np.abs(eeg_data), 99))
        if global_max_abs <= 0:
            raise ValueError("Global 99th percentile of EEG data is zero. Cannot normalize.")
        print(f"Using global normalization: 99th percentile = {global_max_abs:.6e}")
    else:
        global_max_abs = None
        print("Using per-window normalization (default)")
    
    # Process segments
    all_window_predictions: List[Dict] = []
    all_window_metrics: List[Dict] = []
    best_window = None
    best_score = -float('inf')
    
    for seg_idx, (start_sample, segment) in enumerate(
        _segment_signal(
            eeg_data,
            window_samples,
            step_samples,
            pad=not args.no_pad_last,
            max_windows=args.max_windows,
        )
    ):
        # Use global or per-window normalization
        if global_max_abs is not None:
            max_abs = global_max_abs
        else:
            max_abs = float(np.abs(segment).max())
            if max_abs <= 0:
                print(f"Skipping window {seg_idx} - zero variance")
                continue
        
        # Save segment
        mat_seg_path = segments_dir / f"segment_{seg_idx:04d}.mat"
        _save_segment_to_mat(segment, mat_seg_path)
        
        # Run inference
        eeg_tensor = torch.from_numpy((segment / max_abs).astype(np.float32))
        pred = _infer_model(
            model,
            eeg_tensor,
            args.train_loss,
            max_src_val=1.0,
            max_eeg_val=max_abs,
            leadfield=leadfield,
        )
        
        # Compute window score
        energy = pred.abs().sum(dim=0)
        peak_idx = int(torch.argmax(energy).item())
        score = float(energy[peak_idx].item())
        peak_activity = pred[:, peak_idx].abs().cpu().numpy()
        
        # Store predictions
        window_data = {
            'window_idx': seg_idx,
            'start_time': start_sample / fs,
            'end_time': (start_sample + window_samples) / fs,
            'predictions': pred.detach().cpu().numpy(),
            'max_abs': max_abs,
            'peak_idx': peak_idx,
            'score': score,
        }
        all_window_predictions.append(window_data)
        
        # Compute metrics if ground truth available
        window_metrics = None
        if gt_source is not None:
            end_sample = min(start_sample + window_samples, gt_source.shape[1])
            gt_segment = gt_source[:, start_sample:end_sample]
            if gt_segment.shape[1] < window_samples:
                gt_segment = np.pad(gt_segment, ((0, 0), (0, window_samples - gt_segment.shape[1])), mode="edge")
            
            gt_tensor = torch.from_numpy(gt_segment.astype(np.float32))
            
            # Get seeds and patches from metadata if available
            seeds = None
            patches = None
            if metadata_file and metadata_file.exists():
                with open(metadata_file, 'r') as f:
                    md = json.load(f)
                seeds = md.get("seeds", None)
                if seeds is not None and not isinstance(seeds, list):
                    seeds = [seeds]
                if "act_src" in md:
                    patches = []
                    for p in range(md.get("n_patch", 1)):
                        patch_key = f"patch_{p + 1}"
                        if patch_key in md["act_src"]:
                            patches.append(md["act_src"][patch_key])
            
            metrics = _compute_metrics(
                pred.detach().cpu(),
                gt_tensor,
                source_space.positions,
                neighbors,
                fs,
                seeds=seeds,
                patches=patches,
            )
            window_metrics = {
                'window_idx': seg_idx,
                **metrics.to_dict()
            }
            all_window_metrics.append(window_metrics)
        
        # Track best window
        if score > best_score:
            best_score = score
            best_window = {
                'window_idx': seg_idx,
                'start_sample': start_sample,
                'start_time': start_sample / fs,
                'peak_idx': peak_idx,
                'peak_activity': peak_activity,
                'score': score,
                'prediction': pred.detach().cpu(),
                'metrics': window_metrics,
            }
    
    if best_window is None:
        raise RuntimeError("No valid windows processed")
    
    print(f"\nProcessed {len(all_window_predictions)} windows")
    print(f"Best window: {best_window['window_idx']} with score {best_window['score']:.4f}")
    
    # Generate visualization
    geom: BrainGeometry = _load_brain_geometry(head_model)
    cmap_name = "inferno"
    
    interactive_path = output_dir / f"{mat_path.stem}_window{best_window['window_idx']:04d}_interactive.html"
    _save_interactive_heatmap(
        geom,
        best_window['peak_activity'],
        title=f"1dCNN Inference — {mat_path.stem} window {best_window['window_idx']}",
        cmap_name=cmap_name,
        output_path=interactive_path,
    )
    print(f"Saved interactive plot to {interactive_path}")
    
    # Prepare animation data
    if len(all_window_predictions) > 0:
        print(f"Generating animation data from {len(all_window_predictions)} windows...")
        
        activity_timeline, timestamps = _prepare_animation_timeline(all_window_predictions)
        
        # Apply temporal smoothing if requested
        if args.smoothing_alpha is not None:
            print(f"Applying EMA temporal smoothing with alpha={args.smoothing_alpha:.2f}...")
            activity_smoothed = _apply_exponential_smoothing(activity_timeline, args.smoothing_alpha)
            
            # Save both raw and smoothed versions
            save_raw_too = True
        else:
            activity_smoothed = None
            save_raw_too = False
        
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
        
        # Prepare animation data dictionary (use smoothed if available)
        activity_to_save = activity_smoothed if activity_smoothed is not None else activity_timeline
        
        animation_data = {
            'activity': activity_to_save.astype(np.float32),
            'timestamps': timestamps.astype(np.float32),
            'source_positions': np.ascontiguousarray(geom.positions_mm.astype(np.float32)),
            'triangles': triangles,
            'fps': np.array(actual_fps, dtype=np.int32),
        }
        
        animation_path = output_dir / 'animation_data.npz'
        np.savez_compressed(str(animation_path), **animation_data)
        
        file_size_mb = animation_path.stat().st_size / (1024 * 1024)
        print(f"Saved animation data to {animation_path}")
        print(f"  - {len(timestamps)} frames at ~{actual_fps} FPS ({timestamps[-1]:.2f}s duration)")
        print(f"  - File size: {file_size_mb:.2f} MB")
        if activity_smoothed is not None:
            print(f"  - Applied EMA smoothing (alpha={args.smoothing_alpha:.2f})")
        
        # Optionally save raw version if smoothing was applied
        if save_raw_too:
            animation_data_raw = {
                'activity': activity_timeline.astype(np.float32),
                'timestamps': timestamps.astype(np.float32),
                'source_positions': np.ascontiguousarray(geom.positions_mm.astype(np.float32)),
                'triangles': triangles,
                'fps': np.array(actual_fps, dtype=np.int32),
            }
            animation_path_raw = output_dir / 'animation_data_raw.npz'
            np.savez_compressed(str(animation_path_raw), **animation_data_raw)
            file_size_raw_mb = animation_path_raw.stat().st_size / (1024 * 1024)
            print(f"Saved raw (unsmoothed) animation data to {animation_path_raw}")
            print(f"  - File size: {file_size_raw_mb:.2f} MB")
    
    # Compute aggregate metrics if ground truth available
    aggregate_metrics = None
    if all_window_metrics:
        aggregate_metrics = {
            'mean_nmse': np.mean([m['nmse'] for m in all_window_metrics]),
            'mean_auc': np.mean([m['auc'] for m in all_window_metrics]),
            'mean_localization_error_mm': np.mean([m['localization_error_mm'] for m in all_window_metrics]),
            'mean_time_error_ms': np.mean([m['time_error_ms'] for m in all_window_metrics]),
            'best_window_nmse': best_window['metrics']['nmse'] if best_window['metrics'] else None,
            'best_window_auc': best_window['metrics']['auc'] if best_window['metrics'] else None,
            'best_window_localization_error_mm': best_window['metrics']['localization_error_mm'] if best_window['metrics'] else None,
            'n_windows': len(all_window_metrics),
        }
        print(f"\nAggregate Metrics:")
        print(f"  Mean nMSE: {aggregate_metrics['mean_nmse']:.4f}")
        print(f"  Mean AUC: {aggregate_metrics['mean_auc']:.4f}")
        print(f"  Mean Localization Error: {aggregate_metrics['mean_localization_error_mm']:.2f} mm")
        print(f"  Best Window Localization Error: {aggregate_metrics['best_window_localization_error_mm']:.2f} mm")
    
    # Save summary
    summary = {
        "mat_file": str(mat_path),
        "window_samples": window_samples,
        "overlap_fraction": overlap_fraction,
        "n_windows_processed": len(all_window_predictions),
        "best_window": {
            "window_index": best_window['window_idx'],
            "start_time_seconds": best_window['start_time'],
            "peak_time_index": best_window['peak_idx'],
            "score": best_window['score'],
        },
        "interactive_plot": str(interactive_path.relative_to(output_dir)),
        "has_ground_truth": gt_source is not None,
        "source_file": str(source_file) if source_file else None,
        "metrics": aggregate_metrics,
        "window_metrics": all_window_metrics if all_window_metrics else None,
    }
    
    summary_path = output_dir / "inference_summary.json"
    with summary_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")
    
    if args.open_plot:
        webbrowser.open(interactive_path.as_uri())


if __name__ == "__main__":
    main()
