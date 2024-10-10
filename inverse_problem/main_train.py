"""
17 nov. 2023
## !! leadfield path to change in the file if necessary

"""

import os
import argparse
import time
import sys

import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch.utils.data import DataLoader, random_split
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger, tensorboard

from pytorch_lightning import seed_everything

from loaders import ModSpikeEEGBuild, EsiDatasetds_new
from scipy.io import loadmat
from utils.utl import CosineSimilarityLoss, logMSE
from load_data.FolderStructure import FolderStructure
from load_data import HeadModel
import json

# Training on GPU if available
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print(f"Device: {device}")


# seed
seed_everything(0)

home = os.path.expanduser("~")
# ------------------------ ARGPARSE -----------------------------------------------#

parser = argparse.ArgumentParser(fromfile_prefix_chars="@")

#argument to load the data
parser.add_argument("simu_name", type=str, help="name of the simulation")
parser.add_argument("-root_simu", type=str, required=True, help="path to the folder containing data")
parser.add_argument("-results_path", type=str, required=True, help="where to save results")

parser.add_argument("-orientation", type=str, default="constrained", help="constrained or unconstrained, orientation of the sources")
parser.add_argument("-electrode_montage", type=str, default="standard_1020", help="name of the electrode montage to use")
parser.add_argument("-source_space", type=str, default="ico3", help="name of the source space")
parser.add_argument(
    "-simu_type", type=str, help="type of simulation used (NMM or SEREEGA)"
)
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
    "-model",
    default="1dcnn",
    type=str,
    help="name of the model to use (neural network",
)
parser.add_argument(
    "-inter_layer",
    type=int,
    default=4096,
    help="number of channels of the hidden layer of the 1D CNN",
)
parser.add_argument(
    "-kernel_size", type=int, default=5, help="kernel size of the 1D CNN"
)

parser.add_argument(
    "-n_epochs", "--ep", default=100, type=int, help="number of epochs for training"
)
parser.add_argument(
    "-no_early_stop", action="store_false", help="do not use early stopping"
)
parser.add_argument("-batch_size", "--bs", default=8, type=int, help="batch size")
parser.add_argument(
    "-scaler",
    default="linear",
    type=str,
    help="type of normalisation to use (max or linear)",
)
parser.add_argument(
    "-loss", default="cosine", type=str, help="type of loss function to use"
)
parser.add_argument(
    "-sfolder",
    default="trainings",
    type=str,
    help="name of the subfolder in which to save results",
)


args = parser.parse_args()

# ------ where to save results -------- #
results_path = f"{args.results_path}/{args.simu_name}{args.source_space}_/{args.sfolder}"
os.makedirs(f"{results_path}/{args.sfolder}", exist_ok=True)
## -------------------------------------- LOAD DATA ----------------------------------------------- ##
root_simu = args.root_simu

simu_path = f"{root_simu}/{args.orientation}/{args.electrode_montage}/{args.source_space}/simu/{args.simu_name}"
model_path = f"{root_simu}/{args.orientation}/{args.electrode_montage}/{args.source_space}/model"


config_file = f"{simu_path}/{args.simu_name}{args.source_space}_config.json"

with open(config_file, "r") as f:
    general_config_dict = json.load(f)
general_config_dict["eeg_snr"] = args.eeg_snr
general_config_dict["simu_name"] = args.simu_name

folders = FolderStructure(root_simu, general_config_dict)
source_space_obj = HeadModel.SourceSpace(folders, general_config_dict)

# load the proper leadfield for the regional source space :
if args.source_space == "fsav_994":
    fwd = loadmat(f"{model_path}/LF_fsav_994.mat")["G"]
# else ? ## TODO
############################### LOAD DATA ################################
if args.simu_type.upper() == "NMM":
    spikes_data_path = f"{root_simu}/{args.orientation}/{args.electrode_montage}/{args.source_space}/simu/{args.spikes_folder}"
    dataset_meta_path = f"{simu_path}/{args.simu_name}.mat"

    ds_dataset = ModSpikeEEGBuild(
        spike_data_path=spikes_data_path,
        metadata_file=dataset_meta_path,
        fwd=fwd,
        n_times=args.n_times,
        args_params={"dataset_len": args.to_load},
        spos=source_space_obj.positions,
        norm=args.scaler,
    )

elif args.simu_type.upper() == "SEREEGA":
    simu_data_path = f"{home}/Documents/Data/simulation"
    config_file = f"{simu_data_path}/{args.simu_name}{args.source_space}_config.json"

    ds_dataset = EsiDatasetds_new(
        root_simu,
        config_file,
        args.simu_name,
        args.source_space,
        "standard_1020",
        args.to_load,
        args.eeg_snr,
        noise_type={"white": 1.0, "pink": 0.0},
        norm=args.scaler,
    )

else:
    sys.exit("unknown simulation type (argument simu_type)")


train_ds, val_ds = random_split(
    ds_dataset,
    [int(args.to_load * (1 - args.per_valid)), int(args.to_load * args.per_valid)],
)
train_dataloader = DataLoader(dataset=train_ds, batch_size=args.bs, shuffle=True)
val_dataloader = DataLoader(dataset=val_ds, batch_size=args.bs, shuffle=False)

n_electrodes = fwd.shape[0]
n_sources = fwd.shape[1]

## ------------------------------------------- NETWORK TO LOAD --------------------------##
# loss function
if args.loss == "cosine":
    crit = CosineSimilarityLoss()
elif args.loss.upper() == "mse":
    crit = F.mse_loss()
elif args.loss.upper() == "logmse":
    crit = logMSE()
else:
    sys.exit("unknown loss function")

# ---------- CNN 1D ---------#
if args.model.upper() == "1DCNN":
    from models.cnn_1d import CNN1Dpl as net

    lr = 1e-3
    net_parameters = {
        "channels": [
            n_electrodes,
            args.inter_layer,
            n_sources,
        ],
        "kernel_size": args.kernel_size,
        "bias": False,
        "optimizer": torch.optim.Adam,
        "lr": lr,
        "criterion": crit,
    }
    model = net(**net_parameters)

##------------- LSTM ---------------------##
elif args.model.upper() == "LSTM":
    from models.lstm import HeckerLSTMpl as net

    lr = 1e-3
    net_parameters = {
        "n_electrodes": n_electrodes,
        "hidden_size": 85,
        "n_sources": n_sources,
        "bias": False,
        "optimizer": torch.optim.Adam,
        "lr": lr,
        "criterion": crit,  # nn.MSELoss(reduction='sum'), #CosineSimilarityLoss(),
        "mc_dropout_rate": 0,
    }
    model = net(**net_parameters)

##------------------ DEEPSIF ----------------##
elif args.model.upper() == "DEEPSIF":
    from models.deepsif import DeepSIFpl as net

    lr = 1e-3
    net_parameters = {
        "num_sensor": n_electrodes,
        "num_source": n_sources,
        "temporal_input_size": 500,
        "optimizer": torch.optim.Adam,
        "lr": lr,
        "criterion": crit,
    }
    model = net(**net_parameters)

else:
    sys.exit("unknown model")

##------------------- TRAINING ----------------------------##
n_train_samples = len(train_ds)

if args.model.upper() == "1DCNN":
    subfolder = (
        f"simu_{args.simu_type}_"
        f"srcspace_{args.source_space}"
        f"_model_{args.model}"
        f"_interlayer_{args.inter_layer}"
        f"_trainset_{n_train_samples}"
        f"_epochs_{args.ep}"
        f"_loss_{args.loss}"
        f"_norm_{args.scaler}"
    ) 
else:
    subfolder = (
        f"simu_{args.simu_type}_"
        f"srcspace_{args.source_space}"
        f"_model_{args.model}"
        f"_trainset_{n_train_samples}"
        f"_epochs_{args.ep}"
        f"_loss_{args.loss}"
        f"_norm_{args.scaler}"
    )

results_path = f"{results_path}/{subfolder}"
os.makedirs(f"{results_path}/trained_models", exist_ok=True)
best_model_path = f"{results_path}/trained_models/{args.model}_model.pt"

print(f"### best model path: {best_model_path} ###")

# --------------------- TRAINING ------------------------#
checkpoint_callback = ModelCheckpoint(
    dirpath=f"{results_path}/pl_checkpoints",
    filename="{epoch}-{train_loss:.2f}",
    monitor="train_loss",
)

logger = TensorBoardLogger(
    save_dir=f"{results_path}/logs/")#, name=f"{args.sfolder}")#, 
#    version=f"{args.model.lower()}_{args.simu_type.lower()}_trainsize_{n_train_samples}_loss_{args.loss}_norm_{args.scaler}")

# gradient clipping
if args.model.upper() == "LSTM":
    gc_val = 1  # gradient are clipped to a value of 1 for the LSTM
else:
    gc_val = 0

# early stopping strategy
if args.no_early_stop:
    cbs = [checkpoint_callback]
else:
    early_stop_cb = EarlyStopping(monitor="validation_loss", min_delta=0.0, patience=20)
    cbs = [checkpoint_callback, early_stop_cb]

# trainer
trainer = pl.Trainer(
    accelerator=device.type,
    max_epochs=args.ep,
    logger=logger,
    callbacks=cbs,
    log_every_n_steps=1, ## pas ce que je veux non?
    gradient_clip_val=gc_val,
)

print(
    f"<<<<<<<<<<<<<<<<<<<<<<<<<<< training model {args.model.upper()} >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
)
start_t = time.time()
trainer.fit(model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)
end_t = time.time()
#trainer.save_checkpoint(f"{results_path}/pl_checkpoints/{args.sfolder}.ckpt")


### save best model.
best_model = net.load_from_checkpoint(
    checkpoint_path=trainer.checkpoint_callback.best_model_path,
    map_location=torch.device("cpu"),
    **net_parameters,
)
torch.save(best_model.state_dict(), best_model_path)
print(f"Training time : {(end_t-start_t)}")

with open('./training_times.txt','a') as f :
    f.write(
        f"model : {subfolder} - training time : {(end_t-start_t):0.3f}s\n"
    ) 
    f.write(
        "-------------------------------------------------------------\n"
    )