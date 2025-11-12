"""
2023-08-25 script to evaluate results
modification august 2023 - to use with deepsif datasets (neural mass model based and sereega based)
"""

import argparse
import datetime
import os
import re
import sys
from os.path import expanduser
from pathlib import Path

# Import librairies
import mne
import numpy as np
import torch
from matplotlib import pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
import yaml
import json
import pandas as pd
from pytorch_lightning import seed_everything
from scipy.io import loadmat
from torch.utils.data import DataLoader, random_split

from loaders import ModSpikeEEGBuild, EsiDatasetds_new
from load_data import HeadModel
from load_data.FolderStructure import FolderStructure
from utils import utl
from utils import utl_metrics as met
from utils import utl_inv as inv

############# METHODS ############################
linear_methods = ["MNE", "sLORETA"]  # , "eLORETA"]
nn_methods = ["cnn_1d", "lstm", "deep_sif"]
methods = linear_methods + nn_methods

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
seed_everything(0)
device = torch.device("cpu")
print(f"Device: {device}")

home = expanduser("~")

save_suffix = "test"

################################################################################################################
parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
#argument to load the data
parser.add_argument("simu_name", type=str, help="name of the simulation")
parser.add_argument("-root_simu", type=str, required=True, help="Simulation folder (parent folder in the folder tree containing the simulations)")
parser.add_argument("-results_path", type=str, required=True, help="Path to where to save results")
parser.add_argument(
    "-eval_simu_type", type=str, help="type of simulation used (NMM or SEREEGA)"
)
parser.add_argument("-orientation", type=str, default="constrained", help="constrained or unconstrained, orientation of the sources")
parser.add_argument("-electrode_montage", type=str, default="standard_1020", help="name of the electrode montage to use")
parser.add_argument("-source_space", type=str, default="ico3", help="name of the source space")

parser.add_argument("-spikes_folder", type=str, default="nmm_spikes_nov23", help="folder with spikes for NMM based simulations")
parser.add_argument(
    "-n_times", type=int, default=500, help="number of time samples in the signal"
)
parser.add_argument(
    "-to_load",
    default=100,
    type=int,
    help="number of samples to load in the train+val dataset",
)
parser.add_argument(
    "-per_valid",
    default=0.2,
    type=float,
    help="fraction of the dataset to use for validation",
)
parser.add_argument(
    "-eeg_snr",
    default=5,
    type=int,
    help="SNR of the EEG data (additive white gaussian noise)",
)
parser.add_argument(
    "-subject_name", type=str, default="fsaverage", help="name of the subject used"
)


parser.add_argument(
    "-net_from_file", action="store_true", help="load network parameters from yaml file"
)
parser.add_argument(
    "-params_nn",
    type=str,
    default="./params_nns.yaml",
    help="yaml file with the parameters of the networks to load",
)

# parameters to load the trained model
parser.add_argument(
    "-train_bs", type=int, default=8, help="batch size used for training"
)
parser.add_argument(
    "-n_epochs", type=int, default=100, help="number of epochs used for training11"
)
parser.add_argument(
    "-train_loss", type=str, default="cosine", help="loss used to train the networks"
)
parser.add_argument(
    "-scaler", type=str, default="linear", help="type of scaling to use"
)
parser.add_argument(
    "-train_simu_type", type=str, default="sereega", help="simulation type used for training"
)
parser.add_argument(
    "-train_simu_name", type=str, default="eval", help="name of the simulation used for training" 
)
parser.add_argument(
    "-n_train_samples", type=int, default=-1, help="number of training samples, if <0 : number of samples in the params file will be used"
)
parser.add_argument(
    "-train_sfolder", type=str, default="eval", help="name of the folder in which network are saved"
)
parser.add_argument(
    "-inter_layer", type=int, default=2048, help="number of channels of the 1dcnn"
)
parser.add_argument(
    "-kernel_size", type=int, default=5, help="kernel size of the 1dcnn"
)
# 
parser.add_argument(
    "-mets", "--methods", nargs="+", help="methods to use", default=methods
)
parser.add_argument(
    "-sfolder",
    type=str,
    default=f"valid_{datetime.datetime.now().year}-{datetime.datetime.now().month}-{datetime.datetime.now().day}",
    help="Name of the folder to create to save results.",
)
parser.add_argument(
    "-save_suff", type=str, default=save_suffix, help="suffix to save metric values"
)

args = parser.parse_args()
#----------------------------------------------------------------------#
root_simu = args.root_simu

# Always store evaluation artifacts inside the repository results directory.
repo_root = Path(__file__).resolve().parent.parent
repo_results_dir = (repo_root / "results").resolve()

requested_results = Path(args.results_path).expanduser()
if not requested_results.is_absolute():
    requested_results = (repo_results_dir / requested_results).resolve()
else:
    requested_results = requested_results.resolve()

try:
    requested_results.relative_to(repo_results_dir)
    results_base_path = requested_results
except ValueError:
    print(
        "Requested results_path is outside the repository. Redirecting to project results directory:"
        f" {repo_results_dir}"
    )
    results_base_path = repo_results_dir

results_base_path.mkdir(parents=True, exist_ok=True)

dataset = f"{args.simu_name}{args.source_space}_"
eval_results_path_path = results_base_path / dataset / "eval" / args.sfolder
eval_results_path_path.mkdir(parents=True, exist_ok=True)

results_path = str(results_base_path)
eval_results_path = str(eval_results_path_path)
print(f"Saving evaluation outputs to {eval_results_path}")

##----------------LOAD EVAL DATA---------------------##
simu_path = f"{root_simu}/{args.orientation}/{args.electrode_montage}/{args.source_space}/simu/{args.simu_name}"
model_path = f"{root_simu}/{args.orientation}/{args.electrode_montage}/{args.source_space}/model"
config_file = f"{simu_path}/{args.simu_name}{args.source_space}_config.json"

with open(config_file, "r") as f:
    general_config_dict = json.load(f)
general_config_dict["eeg_snr"] = args.eeg_snr
general_config_dict["simu_name"] = args.simu_name

folders = FolderStructure(root_simu, general_config_dict)
source_space = HeadModel.SourceSpace(folders, general_config_dict)
electrode_space = HeadModel.ElectrodeSpace(folders, general_config_dict)
head_model = HeadModel.HeadModel(electrode_space, source_space, folders, "fsaverage")

if args.source_space in ["fsav_994", "ico3"] : 
    fwd = loadmat(f"{model_path}/LF_{args.source_space}.mat")["G"]
else : 
    fwd = head_model.fwd['sol']['data']

# Only load neighbors and region mapping if needed by selected methods
methods = args.methods
need_neighbors = any(m in methods for m in ["deep_sif", "lstm", "MNE", "sLORETA"])  # add other methods if needed
need_region_mapping = any(m in methods for m in ["deep_sif", "lstm", "MNE", "sLORETA"])  # add other methods if needed

neighbors = None
region_mapping = None
fwd_vertices = None
fwd_regions = None
n_vertices = None
n_regs = None

if need_neighbors:
    if os.path.isfile(f"{folders.model_folder}/fs_cortex_neighbors_994.mat"):
        neighbors = (
            loadmat(f"{folders.model_folder}/fs_cortex_neighbors_994.mat")["nbs"] - 1
        )
    else:
        try:
            neighbors_mat = loadmat(f"{folders.model_folder}/fs_cortex_20k_region_mapping.mat")
            neighbors = neighbors_mat["nbs"][0]
            m = -1
            for n in neighbors:
                if n.shape[1] > m:
                    m = n.shape[1]
            neighbors_ref = np.ones((neighbors.shape[0], m), dtype=int) * (-1)
            for r in range(neighbors_ref.shape[0]):
                nbs = neighbors[r][0]
                neighbors_ref[r, : len(nbs)] = nbs
            from scipy.io import savemat
            savemat(
                f"{folders.model_folder}/fs_cortex_neighbors_994.mat", {"nbs": neighbors_ref}
            )
            neighbors = neighbors_ref
        except Exception:
            neighbors = None

if need_region_mapping:
    try:
        region_mapping = loadmat(f"{folders.model_folder}/fs_cortex_20k_region_mapping.mat")["rm"][0]
    except Exception:
        region_mapping = None

if need_region_mapping or need_neighbors:
    try:
        fwd_vertices = mne.read_forward_solution(
            f"{folders.model_folder}/fwd_verticesfsav_994-fwd.fif"
        )
        fwd_vertices = mne.convert_forward_solution(
            fwd_vertices, surf_ori=True, force_fixed=True, use_cps=True, verbose=0
        )
        fwd_regions = mne.read_forward_solution(f"{folders.model_folder}/fwd_fsav_994-fwd.fif")
        fwd_regions = mne.convert_forward_solution(
            fwd_regions, surf_ori=True, force_fixed=True, use_cps=True, verbose=0
        )
        fwd_regions["sol"]["data"] = fwd
        n_vertices = fwd_vertices["nsource"]
        n_regs = len(np.unique(region_mapping)) if region_mapping is not None else None
    except Exception:
        fwd_vertices = None
        fwd_regions = None
        n_vertices = None
        n_regs = None

if neighbors is None:
    # Build a simple k-nearest-neighbor graph from source positions as a fallback.
    pos = source_space.positions
    n_sources_local = pos.shape[0]
    k = min(6, max(n_sources_local - 1, 1))
    if k <= 0:
        neighbors = np.full((n_sources_local, 1), -1, dtype=int)
    else:
        # Compute pairwise distances in a memory-friendly way using broadcasting.
        diff = pos[:, None, :] - pos[None, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        idx = np.argpartition(dist_sq, range(k), axis=1)[:, :k]
        neighbors = idx + 1  # use 1-based indexing to remain compatible with get_patch filtering

if region_mapping is None:
    region_mapping = np.arange(source_space.n_sources)
    n_regs = len(region_mapping)

if n_vertices is None:
    n_vertices = source_space.n_sources

fs = general_config_dict["rec_info"]["fs"]
n_times = general_config_dict["rec_info"]["n_times"]
t_vec = np.arange(0, n_times / fs, 1 / fs)
spos = torch.from_numpy(source_space.positions)  # in meter
mne_info = head_model.electrode_space.info
####################################################################
## load dataset
if args.eval_simu_type.upper() == "NMM":
    spikes_data_path = f"{root_simu}/{args.orientation}/{args.electrode_montage}/{args.source_space}/simu/{args.spikes_folder}"
    dataset_meta_path = f"{simu_path}/{args.simu_name}.mat"

    ds_dataset = ModSpikeEEGBuild(
        spike_data_path=spikes_data_path,
        metadata_file=dataset_meta_path,
        fwd=fwd,
        n_times=args.n_times,
        args_params={"dataset_len": args.to_load},
        spos=source_space.positions,
        norm=args.scaler,
    )

elif args.eval_simu_type.upper() == "SEREEGA":
    ds_dataset = EsiDatasetds_new(
        root_simu,
        config_file,
        args.simu_name,
        args.source_space,
        general_config_dict["electrode_space"]["electrode_montage"],
        args.to_load,
        args.eeg_snr,
        noise_type={"white": 1.0, "pink": 0.0},
    )

else:
    sys.exit("unknown simulation type (argument simu_type)")

n_electrodes = fwd.shape[0]
n_sources = fwd.shape[1]
# split dataset
_, val_ds = random_split(ds_dataset, [1 - args.per_valid, args.per_valid])
val_dataloader = DataLoader(dataset=val_ds, batch_size=1, shuffle=False)
n_val_samples = len(val_dataloader)
print(f">>>>>>>>>>>> Evaluation on {n_val_samples} samples <<<<<<<<<<<<<<<<<")
##------------------------------------------------------------------------##
##------------------------------------------------------------------------##
## trained neural networks parameters
if args.net_from_file : 
    with open(args.params_nn, "r") as f:
        params_file = yaml.safe_load(f)

    cnn1d_params    = params_file["cnn1d"]
    lstm_params     = params_file["lstm"]
    deep_sif_params = params_file["deep_sif"]
#if args.n_train_samples > 0 :
#    cnn1d_params['n_train_samples'] = args.n_train_samples
#    lstm_params['n_train_samples'] = args.n_train_samples
#    deep_sif_params['n_train_samples'] = args.n_train_samples
#if args.dataset : 
#    cnn1d_params['dataset'] = f"{args.simu_name}{args.source_space}_"
#    lstm_params['dataset'] = f"{args.simu_name}{args.source_space}_"
#    deep_sif_params['dataset'] = f"{args.simu_name}{args.source_space}_"

else : 
    train_dataset = f"{args.train_simu_name}{args.source_space}_"
    train_params = {
        "train_simu_type" : args.train_simu_type,
        "inter_layer" : args.inter_layer,
        "kernel_size" : args.kernel_size,
        "n_epochs" : args.n_epochs, 
        "batch_size" : args.train_bs, 
        "dataset" : train_dataset, 
        "exp" : args.train_sfolder,
        "n_train_samples" : args.n_train_samples,
        "loss" :args.train_loss,
        "norm" : args.scaler, 
        "n_electrodes" : n_electrodes, 
        "n_sources" : n_sources, 
        "hidden_size" : 85, 
        "temporal_input_size" : 500, 
    }
    cnn1d_params = train_params
    lstm_params = train_params
    deep_sif_params = train_params


##############################################################################################################################################
############### load networks

methods = args.methods

if "cnn_1d" in methods:
    if args.net_from_file : 
        train_dataset = cnn1d_params['dataset']
    train_results_path = f"{results_path}/{train_dataset}"

    from models.cnn_1d import CNN1Dpl as cnn1d_net

    if (
        cnn1d_params["n_electrodes"] != head_model.electrode_space.n_electrodes
        or cnn1d_params["n_sources"] != head_model.source_space.n_sources
    ):
        sys.exit(
            (
                f"number of electrodes or sources in head model does not match with number of electrodes or sources in 1dcnn model"
                f"electrodes head model : {head_model.electrode_space.n_electrodes} - electrodes 1dcnn : {cnn1d_params['n_electrodes']}\n"
                f"sources head model : {head_model.source_space.n_sources} - sources 1dcnn : {cnn1d_params['n_sources']}"
            )
        )

    # Attempt to locate the trained model saved by main_train.py.
    # Default naming in main_train.py uses the original model argument (typically "1dcnn").
    expected_fragments = [
        f"simu_{args.train_simu_type}".lower(),
        f"srcspace_{head_model.source_space.src_sampling}".lower(),
        f"interlayer_{cnn1d_params['inter_layer']}".lower()
        if isinstance(cnn1d_params.get("inter_layer"), (int, float))
        else None,
        f"trainset_{cnn1d_params['n_train_samples']}"
        if isinstance(cnn1d_params.get("n_train_samples"), (int, float))
        and cnn1d_params["n_train_samples"] >= 0
        else None,
        f"epochs_{cnn1d_params['n_epochs']}"
        if isinstance(cnn1d_params.get("n_epochs"), (int, float))
        else None,
        f"loss_{cnn1d_params['loss']}".lower() if cnn1d_params.get("loss") else None,
        f"norm_{cnn1d_params['norm']}".lower() if cnn1d_params.get("norm") else None,
        str(cnn1d_params.get("exp", "")).lower(),
    ]
    expected_fragments = [frag for frag in expected_fragments if frag]

    target_filenames = {
        "1dcnn_model.pt",
        "1DCNN_model.pt",
    }

    candidate_paths = []

    search_roots = {os.path.abspath(train_results_path)}

    eval_dir = os.path.dirname(os.path.abspath(__file__))
    repo_results_root = os.path.abspath(
        os.path.join(eval_dir, "..", "results", train_dataset)
    )
    search_roots.add(repo_results_root)

    exp_folder = cnn1d_params.get("exp")
    if exp_folder:
        for base_root in list(search_roots):
            search_roots.add(os.path.join(base_root, exp_folder))

    for base_root in sorted(search_roots):
        if not os.path.isdir(base_root):
            continue
        for root, _, files in os.walk(base_root):
            for fname in files:
                if fname in target_filenames or fname.lower() in target_filenames:
                    candidate_paths.append(os.path.join(root, fname))

    def _score_path(path: str) -> int:
        plower = path.lower()
        score = 0
        for frag in expected_fragments:
            if frag and frag in plower:
                score += 1
        return score

    if candidate_paths:
        candidate_paths.sort(key=lambda p: (_score_path(p), -len(p)))
        cnn_model_path = candidate_paths[-1]
    else:
        # Fallback to previously expected deterministic path for clarity in error.
        subfolder = (
            f"simu_{args.train_simu_type}_"
            f"srcspace_{head_model.source_space.src_sampling}"
            f"_model_1dcnn"
            f"_interlayer_{cnn1d_params['inter_layer']}"
            f"_trainset_{cnn1d_params['n_train_samples']}"
            f"_epochs_{cnn1d_params['n_epochs']}"
            f"_loss_{cnn1d_params['loss']}"
            f"_norm_{cnn1d_params['norm']}"
        )
        fallback_path = (
            f"{train_results_path}/{cnn1d_params.get('exp', '')}/{subfolder}/trained_models/1dcnn_model.pt"
        )
        searched_locations = "\n".join(
            sorted({path for path in search_roots if os.path.isdir(path)})
        ) or train_results_path
        sys.exit(
            (
                "Could not locate trained 1dCNN model.\n"
                f"Searched under:\n{searched_locations}\n"
                f"Expected filename(s): {', '.join(sorted(target_filenames))}\n"
                f"Example expected folder: {fallback_path}"
            )
        )

    print(f"CNN model found at: {cnn_model_path}")

    checkpoint_parent = os.path.dirname(os.path.dirname(cnn_model_path))
    folder_signature = os.path.basename(checkpoint_parent).lower()

    interlayer_match = re.search(r"interlayer_(\d+)", folder_signature)
    if interlayer_match:
        inter_layer_ckpt = int(interlayer_match.group(1))
        if cnn1d_params.get("inter_layer") != inter_layer_ckpt:
            print(
                f"Updating inter_layer from {cnn1d_params.get('inter_layer')} to {inter_layer_ckpt} based on checkpoint"
            )
            cnn1d_params["inter_layer"] = inter_layer_ckpt

    kernel_match = re.search(r"kernel(?:size)?_(\d+)", folder_signature)
    if kernel_match:
        kernel_size_ckpt = int(kernel_match.group(1))
        if cnn1d_params.get("kernel_size") != kernel_size_ckpt:
            print(
                f"Updating kernel_size from {cnn1d_params.get('kernel_size')} to {kernel_size_ckpt} based on checkpoint"
            )
            cnn1d_params["kernel_size"] = kernel_size_ckpt

    net_parameters = {
        "channels": [
            cnn1d_params["n_electrodes"],
            cnn1d_params["inter_layer"],
            cnn1d_params["n_sources"],
        ],
        "kernel_size": cnn1d_params["kernel_size"],
        "bias": False,
        "optimizer": None,
        "lr": 1e-3,
        "criterion": None,  # nn.MSELoss(reduction='sum'), #CosineSimilarityLoss(),
        # "dropout_rate" : 0.2
    }
    cnn = cnn1d_net(**net_parameters)
    cnn.load_state_dict(torch.load(cnn_model_path))
    cnn.eval()
    cnn_model_name = os.path.relpath(cnn_model_path, train_results_path)


if "lstm" in methods:
    if args.net_from_file : 
        train_dataset = lstm_params['dataset']
    train_results_path = f"{results_path}/{train_dataset}"
    if (
        lstm_params["n_electrodes"] != head_model.electrode_space.n_electrodes
        or lstm_params["n_sources"] != head_model.source_space.n_sources
    ):
        sys.exit(
            (
                f"number of electrodes or sources in head model does not match with number of electrodes or sources in lstm model"
                f"electrodes head model : {head_model.electrode_space.n_electrodes } - electrodes lstm : {lstm_params['n_electrodes']}"
                f"sources head model : {head_model.source_space.n_sources } - sources lstm : {lstm_params['n_sources']}"
            )
        )

    lstm_model_name = (
        f"simu_{args.train_simu_type}_"
        f"srcspace_{head_model.source_space.src_sampling}"
        f"_model_lstm"
        f"_trainset_{lstm_params['n_train_samples']}"
        f"_epochs_{lstm_params['n_epochs']}"
        f"_loss_{lstm_params['loss']}"
        f"_norm_{lstm_params['norm']}.pt"
    )
    lstm_model_path = f"{train_results_path}/trained_models/{lstm_params['exp']}/{lstm_model_name}"
    if os.path.exists(lstm_model_path):
        print("LSTM model is available for use")
    else:
        sys.exit(
            "LSTM model is not accessible.\nTry other parameters or train your model first."
        )

    # from models.lstm import HeckerLSTM as lstm_net
    from models.lstm import HeckerLSTMpl as lstm_net

    net_parameters = {
        "n_electrodes": lstm_params["n_electrodes"],
        "hidden_size": lstm_params["hidden_size"],
        "n_sources": lstm_params["n_sources"],
        "bias": False,
        "optimizer": None,
        "lr": 1e-3,
        "criterion": None,  # nn.MSELoss(reduction='sum'), #CosineSimilarityLoss(),
        "mc_dropout_rate": 0,
    }

    lstm = lstm_net(**net_parameters)
    lstm.load_state_dict(torch.load(lstm_model_path))
    lstm.eval()


if "deep_sif" in methods:
    if args.net_from_file : 
        train_dataset = deep_sif_params['dataset']
    train_results_path = f"{results_path}/{train_dataset}"
    if (
        deep_sif_params["n_electrodes"] != head_model.electrode_space.n_electrodes
        or deep_sif_params["n_sources"] != head_model.source_space.n_sources
    ):
        sys.exit(
            (
                f"number of electrodes or sources in head model does not match with number of electrodes or sources in deep sif model"
                f"electrodes head model : {head_model.electrode_space.n_electrodes} - electrodes deep sif : {deep_sif_params['n_electrodes']}"
                f"sources head model : {head_model.source_space.n_sources} - sources deep sif : {deep_sif_params['n_sources']}"
            )
        )

    deep_sif_model_name = (
        f"simu_{args.train_simu_type}_"
        f"srcspace_{head_model.source_space.src_sampling}"
        f"_model_deepsif"
        f"_trainset_{deep_sif_params['n_train_samples']}"
        f"_epochs_{deep_sif_params['n_epochs']}"
        f"_loss_{deep_sif_params['loss']}"
        f"_norm_{deep_sif_params['norm']}.pt"
    )
    deep_sif_model_path = f"{train_results_path}/trained_models/{deep_sif_params['exp']}/{deep_sif_model_name}"
    if os.path.exists(deep_sif_model_path):
        print("DEEP SIF model is available for use")
    else:
        sys.exit(
            f"DEEP SIF model is not accessible.\nTry other parameters or train your model first.\n{deep_sif_model_path}"
        )

    net_parameters = {
        "num_sensor": deep_sif_params["n_electrodes"],
        "num_source": deep_sif_params["n_sources"],
        "temporal_input_size": deep_sif_params["temporal_input_size"],
        "optimizer": None,
        "lr": 1e-3,
        "criterion": None,
    }

    from models.deepsif import DeepSIFpl as deep_sif_net

    deep_sif = deep_sif_net(**net_parameters)
    deep_sif.load_state_dict(
        torch.load(deep_sif_model_path, map_location=torch.device("cpu"))
    )
    deep_sif.eval()

##################################################################################################
# to save metric values
methods.append("gt")
nmse_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
loc_error_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
psnr_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
time_error_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}
auc_dict = {method: np.zeros((n_val_samples, 1)) for method in methods}

##############################################################
########## test for noise covariance estimation #############
"""
# 1. take 10 random samples in the dataset
# 2. perform SVD and noise decomposition on each and concatenate noise signals
# 3. use the concatenated results to estimate noise cov
from scipy.linalg import svd
rand_samp = np.random.randint( 0, len(val_ds), 10 )
sig_noise = np.zeros((n_electrodes, n_times))
for i in rand_samp : 
    sig = val_ds[i][0].numpy()
    u,s,v = svd( sig.transpose() )
    noise = sig.transpose() - np.dot( u[:,:2] * s[:2], v[:2,:] )
    sig_noise = np.concatenate( (sig_noise, noise.transpose()), axis=1 )
raw_noise = mne.io.RawArray(data = sig_noise, info=mne_info)
noise_cov = mne.compute_raw_covariance(raw_noise)
"""

######################## DO THE EVALUATION

noise_only_eeg_data = []
#################################
if args.eval_simu_type.lower() == "sereega":
    md_keys = [k for k, _ in val_ds.dataset.md_dict.items()]
c = 0
nf=0
overlapping_regions = 0
for k in val_ds.indices:
    M, j = val_ds.dataset[k]
    M, j = M.float(), j.float()

    M_unscaled = M * val_ds.dataset.max_eeg[k]
    j_unscaled = j * val_ds.dataset.max_src[k]

    j_unscaled_vertices = np.zeros((n_vertices, n_times))
    for r in range(n_regs):
        j_unscaled_vertices[np.where(region_mapping == r)[0], :] = j_unscaled[r, :]

    # data covariance:
    # activity_thresh = 0.1
    # noise_cov, data_cov, nap = inv.mne_compute_covs(
    #    (M_unscaled).numpy(), mne_info, activity_thresh
    # )

    ### TEST BETTER NOISE COV
    raw_noise = mne.io.RawArray(
        data=np.random.randn(head_model.electrode_space.n_electrodes, 600),
        info=mne_info,verbose=False
    )
    noise_cov = mne.compute_raw_covariance(raw_noise, verbose=False)
    data_cov = 1
    if data_cov is not None:
        ## ici il y a un distinction à faire selon les jeux de données
        if args.eval_simu_type.lower() == "sereega":
            seeds = val_ds.dataset.md_dict[md_keys[k]]["seeds"]
            if type(seeds) is int:
                seeds = [seeds]
        else:
            seeds = list(val_ds.dataset.dataset_meta["selected_region"][k][:, 0])
            if type(seeds) is int:
                seeds = [seeds]

        seeds = [int(s) for s in seeds]
        if len(seeds) > 0 and max(seeds) >= head_model.source_space.n_sources:
            seeds = [s - 1 for s in seeds]

        eeg = mne.io.RawArray(
            data=M, info=head_model.electrode_space.info, first_samp=0.0, verbose=False
        )
        eeg = mne.set_eeg_reference(eeg, "average", projection=True, verbose=False)[0]

        # stc_gt = mne.SourceEstimate(
        #    data=j_unscaled_vertices, # 256//2 = instant du pic à visualiser @TODO : change le codage en dur
        #    vertices= [ fwd_vertices['src'][0]['vertno'], fwd_vertices['src'][1]['vertno'] ],
        #    tmin=0.,
        #    tstep=1/fs,
        #    subject="fsaverage"
        # )

        # compute the diverse inverse solutions
        for method in methods:
            if method == "gt" : 
                j_hat = j_unscaled
            # compute inverse solution
            elif method in linear_methods:
                lambda2 = 1.0 / (args.eeg_snr**2)
                inv_op = mne.minimum_norm.make_inverse_operator(
                    info=eeg.info,
                    forward=fwd_regions,
                    noise_cov=noise_cov,
                    loose=0,
                    depth=0,
                    verbose=False
                )
                stc_hat = mne.minimum_norm.apply_inverse_raw(
                    raw=eeg, inverse_operator=inv_op, lambda2=lambda2, method=method, verbose=False
                )

                j_hat = torch.from_numpy(stc_hat.data)

            elif method == "cnn_1d":
                with torch.no_grad():
                    j_hat = cnn.model(M.unsqueeze(0)).squeeze()
                if cnn1d_params["loss"] == "cosine":
                    j_hat = utl.gfp_scaling(
                        M_unscaled,
                        j_hat,
                        torch.from_numpy(head_model.fwd["sol"]["data"]),
                    )
                else :#if cnn1d_params["post_scale"] == "amp":
                    #if args.scaler == "eeg_max":
                    j_hat = j_hat * val_ds.dataset.max_src[k]
                #else:
                #    pass

            elif method == "lstm":
                with torch.no_grad():
                    j_hat = lstm(M.unsqueeze(0)).squeeze()
                if lstm_params["loss"] == "cosine":
                    j_hat = utl.gfp_scaling(
                        M_unscaled,
                        j_hat,
                        torch.from_numpy(head_model.fwd["sol"]["data"]),
                    )  # * esi_datamodule.train_scaler.maxs[k]
                else : #if lstm_params["post_scale"] == "amp":
                    #if args.scaler == "eeg_max":
                    j_hat = j_hat * val_ds.dataset.max_src[k]

            elif method == "deep_sif":
                with torch.no_grad():
                    j_hat = deep_sif(M.unsqueeze(0)).squeeze()
                if deep_sif_params["loss"] == "cosine":
                    j_hat = utl.gfp_scaling(
                        M_unscaled,
                        j_hat,
                        torch.from_numpy(head_model.fwd["sol"]["data"]),
                    )  # * esi_datamodule.train_scaler.maxs[k]
                else : #if deep_sif_params["post_scale"] == "amp":
                    #if args.scaler == "eeg_max":
                    j_hat = j_hat * val_ds.dataset.max_src[k]

            else:
                sys.exit(f"unrecognized method {method}")

            le = 0
            te = 0
            nmse = 0
            auc_val = 0
            seeds_hat = []
            ## check for overlap ------ @TODO : fix ok for 2 sources, not for more
            
            ## ici il y a un distinction à faire selon les jeux de données
            if args.eval_simu_type.lower() == "sereega":
                seeds = val_ds.dataset.md_dict[md_keys[k]]["seeds"]
                if type(seeds) is int:
                    seeds = [seeds]
            else:
                seeds = list(val_ds.dataset.dataset_meta["selected_region"][k][:, 0])
                seeds = [s.astype(int) for s in seeds]
                if type(seeds) is int:
                    seeds = [seeds]
                
            patches = [ [] for _ in range(len(seeds)) ]
            if args.eval_simu_type.lower() == "nmm" :
                raw_lb = val_ds.dataset.dataset_meta["selected_region"][k].astype(
                    int
                )
                for kk in range(len(seeds)) : 
                    curr_lb = utl.get_patch(order=3, idx=seeds[kk], neighbors=neighbors)
                    #curr_lb = raw_lb[kk, np.logical_not(ispadding(raw_lb[kk]))]
                    patches[kk] = curr_lb
            else : 
                for kk in range(len(seeds)) : 
                    patch_vals = np.array(
                        val_ds.dataset.md_dict[md_keys[k]]["act_src"][f"patch_{kk+1}"]
                    ).astype(int)
                    if patch_vals.size > 0 and patch_vals.max() >= head_model.source_space.n_sources:
                        patch_vals = patch_vals - 1
                    patches[kk] = patch_vals.tolist()
            if len(patches) >= 2:
                inter = list(
                    set(patches[0]).intersection(patches[1])
                )
                if len(inter) > 0:  # for overlapping regions : only keep seed with max activity
                    overlapping_regions += 1
                    to_keep = torch.argmax(
                        torch.Tensor(
                            [
                                j[seeds[0], :].abs().max(),
                                j[seeds[1], :].abs().max(),
                            ]
                        )
                    )
                    seeds = [seeds[to_keep]]
            ## ----------------------
            act_src = [ s for l in patches for s in l ]
            # compute metrics -----------------------------------------------------------------
            gt_norm = j_unscaled.clone()
            gt_max = gt_norm.abs().max()
            if gt_max > 0:
                gt_norm = gt_norm / gt_max

            pred_norm = j_hat.clone()
            pred_max = pred_norm.abs().max()
            if pred_max > 0:
                pred_norm = pred_norm / pred_max

            for kk in range(len(seeds)) :
                s = seeds[kk]
                other_sources = np.setdiff1d(
                    act_src, patches[kk]
                )
                t_eval_gt = torch.argmax(j[s, :].abs())

                # find estimated seed, in a neighboring area
                eval_zone = utl.get_patch(order=5, idx=s, neighbors=neighbors)
                # remove sources from other patches of the eval zone (case of close sources regions)
                eval_zone = np.setdiff1d(eval_zone, other_sources, assume_unique=True)
                if eval_zone.size == 0:
                    eval_zone = np.array([s])

                tighter_zone = utl.get_patch(order=2, idx=s, neighbors=neighbors)
                tighter_zone = np.setdiff1d(tighter_zone, other_sources, assume_unique=True)
                if tighter_zone.size > 0:
                    eval_zone = tighter_zone

                if s not in eval_zone:
                    eval_zone = np.append(eval_zone, s)

                eval_zone = np.unique(eval_zone).astype(int, copy=False)

                eval_zone_idx = torch.as_tensor(eval_zone, device=j_hat.device, dtype=torch.long)
                s_hat = eval_zone_idx[torch.argmax(j_hat[eval_zone_idx, t_eval_gt].abs())].item()

                t_eval_pred = torch.argmax(j_hat[s_hat, :].abs())

                le += torch.sqrt(((spos[s, :] - spos[s_hat, :]) ** 2).sum())
                te += np.abs(t_vec[t_eval_gt] - t_vec[t_eval_pred])
                auc_val += met.auc_t(
                    j_unscaled, j_hat, t_eval_gt, thresh=True, act_thresh=0.0
                )  # probablement peut mieux faire

                #nmse += met.nmse_t_fn(j_unscaled, j_hat, t_eval_gt)
                nmse_tmp = ((gt_norm[:, t_eval_gt] - pred_norm[:, t_eval_gt]) ** 2).mean().item()
                nmse += nmse_tmp
                
                seeds_hat.append(s_hat)

            le = le / len(seeds)
            te = te / len(seeds)
            nmse = nmse / len(seeds)
            auc_val = auc_val / len(seeds)
            tmaxs_pred = torch.argmax(j_hat[seeds_hat, :].abs(), dim=1)
            # time error (error on the instant of the max. activity):
            time_error_dict[method][c] = float(te)
            # print(f"time error: {time_error*1e3} [ms]")

            # localisation error
            loc_error_dict[method][c] = float(le)
            # print(f"localisation error: {loc_error*1e3} [mm]")

            # instant nMSE:
            nmse_dict[method][c] = float(nmse)
            # print(f"nmse at instant of max activity: {nmse_t:.4f}")

            # PSNR
            gt_np = gt_norm.detach().cpu().numpy()
            pred_np = pred_norm.detach().cpu().numpy()
            mse = np.mean((gt_np - pred_np) ** 2)
            if mse <= 0:
                psnr_val = float("inf")
            else:
                data_range = gt_np.max() - gt_np.min()
                if data_range <= 0:
                    data_range = 1.0
                psnr_val = float(10.0 * np.log10((data_range ** 2) / mse))
            psnr_dict[method][c] = psnr_val
            # print(f"psnr for total source distrib: {psnr_val:.4f} [dB]")

            # AUC
            # act_src = esi_datamodule.val_ds.act_src[k]
            auc_dict[method][c] = auc_val
            # print(f"auc: {auc_val:.4f}")

            # change plots to visu. multiple sources
            idx_max_gt = seeds[0]
            idx_max_pred = seeds_hat[0]

    else:
        noise_only_eeg_data.append(c)
    c += 1

    if c%100 == 0 : 
        print(f"---------------{c} validation samples done -------------")
############################################################################
if len(noise_only_eeg_data) > 0:
    for method in methods:
        nmse_dict[method] = np.delete(nmse_dict[method], noise_only_eeg_data)
        loc_error_dict[method] = np.delete(loc_error_dict[method], noise_only_eeg_data)
        auc_dict[method] = np.delete(auc_dict[method], noise_only_eeg_data)
        time_error_dict[method] = np.delete(
            time_error_dict[method], noise_only_eeg_data
        )
        psnr_dict[method] = np.delete(psnr_dict[method], noise_only_eeg_data)
#####################################################################
#############################################################################


for method in methods:
    print(f" >>>>>>>>>>>>>>> Results method {method} <<<<<<<<<<<<<<<<<<<<<<<<<<")
    print(f"mean time error: {time_error_dict[method].mean()*1e3} [ms]")
    print(f"mean localisation error: {loc_error_dict[method].mean()*1e3} [mm]")
    print(f"mean nmse at instant of max activity: {nmse_dict[method].mean():.4f}")
    print(f"psnr for total source distrib: {psnr_dict[method].mean():.4f} [dB]")
    print(f"auc: {auc_dict[method].mean():.4f}")


##################################################################################################"
#os.makedirs(f"{eval_results_path}/{dataset}/eval/{args.sfolder}", exist_ok=True)
#  Save into a csv file

for method in methods:
    if method == "cnn_1d":
        method_info = cnn_model_name
    elif method == "lstm":
        method_info = lstm_model_name
    elif method == "deep_sif":
        method_info = deep_sif_model_name
    else:
        method_info = "none"
    my_values = [
        {
            "simu_name": args.simu_name,
            "src_space": head_model.source_space.src_sampling,
            "method": method,
            "method_info": method_info,
            "valset": str(n_val_samples),
            "noise db": f"{args.eeg_snr}",
            "mean nmse": f"{nmse_dict[method].mean()}",
            "std nmse": f"{nmse_dict[method].std()}",
            "mean loc error": f"{loc_error_dict[method].mean()}",
            "std loc error": f"{loc_error_dict[method].std()}",
            "mean auc": f"{auc_dict[method].mean()}",
            "std auc": f"{auc_dict[method].std()}",
            "mean time error": f"{time_error_dict[method].mean()}",
            "std time error": f"{time_error_dict[method].std()}",
            "mean psnr": f"{psnr_dict[method].mean()}",
            "std psnr": f"{psnr_dict[method].std()}",
        }
    ]

    fields = [
        list(my_values[0].keys())[k] for k in range(len(list(my_values[0].keys())))
    ]

    import csv

    suffix_save_metrics = (
        f"train_simu_{args.train_simu_type}_"
        f"eval_simu_{args.eval_simu_type}_"
        f"method_{method}"
        f"_srcspace_{head_model.source_space.src_sampling}"
        f"_dataset{args.simu_name}"
        f"_n_train_{args.n_train_samples}"
        f"{args.save_suff}"
    )

    with open(
        f"{eval_results_path}/evaluation_metrics_{suffix_save_metrics}.csv",
        "w",
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writeheader()
        writer.writerows(my_values)

print(
    f">>>>>>> results saved in :{eval_results_path}/evaluation_metrics_{suffix_save_metrics}.csv"
)


################### save all values to then plot distribution ###############################
for method in methods:
    if method == "cnn_1d":
        method_info = cnn_model_name
    elif method == "lstm":
        method_info = lstm_model_name
    elif method == "deep_sif":
        method_info = deep_sif_model_name
    else:
        method_info = "none"
    my_values = {
        "nmse": np.squeeze(nmse_dict[method]),
        "loc error": np.squeeze(loc_error_dict[method]),
        "auc": np.squeeze(auc_dict[method]),
        "time error": np.squeeze(time_error_dict[method]),
        "psnr": np.squeeze(psnr_dict[method]),
    }

    df = pd.DataFrame(data=my_values)

    # fields = [
    #    list(my_values[0].keys())[k] for k in range(len(list(my_values[0].keys())))
    # ]

    suffix_save_metrics = (
        f"train_simu_{args.train_simu_type}_"
        f"eval_simu_{args.eval_simu_type}_"
        f"method_{method}"
        f"_srcspace_{head_model.source_space.src_sampling}"
        f"_dataset{args.simu_name}"
        f"_n_train_{args.n_train_samples}"
        f"{args.save_suff}"
    )

    df.to_csv(
        f"{eval_results_path}/evaluation_{suffix_save_metrics}.csv"
    )

    # with open(
    #    f"{home}/Documents/Results/{dataset}/eval/{args.sfolder}/evaluation_{suffix_save_metrics}.csv",
    #    "w",
    # ) as csvfile:
    # writer = csv.DictWriter(csvfile, fieldnames=fields)
    # writer.writeheader()
    # writer.writerows(my_values)

print(
    f">>>>>>> results saved in :{eval_results_path}/evaluation_{suffix_save_metrics}.csv"
)


###################################### plot distribution and save figs #######################################
# TODO
