""" 
Script to create a head model and save the information for future use 
- in python (mne-python)
- in Matlab (SEREEGA)
"""

import argparse
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import mne
import numpy as np
from scipy.io import savemat, loadmat

import utl_head_model as utl_hm

home = os.path.expanduser('~')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-sn",
        "--subject_name",
        type=str,
        required=True,
        #default="fsaverage",
        help="Name of the subject to use",
    )

    parser.add_argument(
        "-em",
        "--electrode_montage",
        type=str,
        required=True,
        #default="standard_1020",
        help="name of the electrode montage to use",
    )
    parser.add_argument(
        "-ss",
        "--source_sampling",
        type=str,
        #default="ico3",
        required=True,
        help="Subsampling to use to sub-sample the brain mesh",
    )
    
    parser.add_argument(
        "-o","--constrained", action="store_true", help="Constrained orientation"
    )
    parser.add_argument("-volume", action="store_true", help="Volume source space")

    parser.add_argument(
        "-cond","--conductivity",
        type=float,
        nargs=3,
        default=(0.3, 0.006, 0.3),
        help="Conductivity values for the BEM model (layers = brain, skull, skin)",
    )

    parser.add_argument(
        "-dps",
        "--deepsif_anatpath",
        type=str,
        default=f"{os.path.expanduser('~')}/Documents/Data",
        help="Name of the folder in which the deepsif anatomy data are saved (if subject_name==deepsif)",
    )

    parser.add_argument(
        "-data_path",
        type=str,
        default=f"{os.path.expanduser('~')}/Documents/Data",
        help="Name of the folder in which to save simulation data",
    )

    parser.add_argument("-s","--save", action="store_true", help="Save data")
    parser.add_argument("-p","--plot", action="store_true", help="Plot figures")

    args = parser.parse_args()


if args.volume:
    args.constrained = False
    suf = "vol_" + str(args.source_sampling)
else:
    suf = str(args.source_sampling)


if args.subject_name == "fsaverage":
    fs_dir = mne.datasets.fetch_fsaverage(verbose=0)
    subjects_dir = os.path.dirname(fs_dir)
    trans = os.path.join(fs_dir, "bem", "fsaverage-trans.fif")
elif args.subject_name == "sample":
    subjects_dir = mne.datasets.sample.data_path() / "subjects"
    trans = f"{mne.datasets.sample.data_path()}/MEG/sample/sample_audvis_raw-trans.fif"
elif args.subject_name == "deepsif" : 
    deepsif_datapath = f"{home}/{args.deepsif_anatpath}"
    region_mapping = loadmat(f"{deepsif_datapath}/fs_cortex_20k_region_mapping.mat")['rm'][0]
else:
    sys.exit("uknown subject")

# create folders
if args.save:
    # Folder name cnstructerd with "simulation/constrained/montage_kind/sampling/model"
    if args.constrained:
        folder_save_fname = os.path.join(
            args.data_path, "simulation", args.subject_name, "constrained", args.electrode_montage, suf, "model"
        )
        
    else:
        folder_save_fname = os.path.join(
            args.data_path, "simulation", args.subject_name, "constrained", args.electrode_montage, suf, "model"
        )
    os.makedirs( folder_save_fname ,exist_ok=True )

##---------CREATE HEAD MODEL-------##
if args.subject_name in ["fsaverage", "sample"] : 
    _, bem, src, info, fwd = utl_hm.create_head_model(
        subject=args.subject_name,
        subjects_dir=subjects_dir,
        trans=trans,
        montage_kind=args.electrode_montage,
        sampling=args.source_sampling,
        constrained=args.constrained,
        volume=args.volume,
        conductivity=args.conductivity,
        fs=1000,
    )
    channels_info, chanlocs_info, sources_info, leadfield = utl_hm.hm_mne_to_sereega(
        info=info, fwd=fwd
    )
    # additional data : triangles and vertices, used later for neighbors computation (extended sources)
    tris = {"tris_lh": fwd["src"][0]["use_tris"], "tris_rh": fwd["src"][1]["use_tris"]}
    tris_lh = {"tris_lh": fwd["src"][0]["use_tris"]}
    tris_rh = {"tris_rh": fwd["src"][1]["use_tris"]}
    verts = {"verts_lh": fwd["src"][0]["vertno"], "verts_rh": fwd["src"][1]["vertno"]}

else : 
    res_dict = utl_hm.create_deepsif_based_headmodel(
        region_mapping = region_mapping, 
        conductivity=args.conductivity, 
        fs = 500, 
        montage_kind=args.electrode_montage
    )
    channels_info, chanlocs_info, sources_info, _ = utl_hm.hm_mne_to_sereega(
        info=res_dict['info'], fwd = res_dict['fwd_regions'] 
    )
    leadfield = {'G' : res_dict['summed_leadfield']}


################# SAVE #####################################
    ### save data
if args.save:
    ch_save_fname = os.path.join(folder_save_fname, "ch_" + suf + ".mat")
    savemat(ch_save_fname, channels_info)
    chlocs_save_fname = os.path.join(folder_save_fname, "chlocs_" + suf + ".mat")
    savemat(chlocs_save_fname, chanlocs_info)
    lf_save_fname = os.path.join(folder_save_fname, "LF_" + suf + ".mat")
    savemat(lf_save_fname, leadfield)
    src_save_fname = os.path.join(folder_save_fname, "sources_" + suf + ".mat")
    savemat(src_save_fname, sources_info)
    if args.subject_name in ["fsaverage", "sample"] : 
        savemat(os.path.join(folder_save_fname, "tris_" + suf + ".mat"), tris)
        savemat(os.path.join(folder_save_fname, "verts_" + suf + ".mat"), verts)
        savemat(os.path.join(folder_save_fname, "tris_lh_" + suf + ".mat"), tris_lh)
        savemat(os.path.join(folder_save_fname, "tris_rh_" + suf + ".mat"), tris_rh)
        # Saving the generated forward solution
        fwd_save_fname = os.path.join(folder_save_fname, "fwd_" + suf + "-fwd.fif")
        mne.write_forward_solution(fwd_save_fname, fwd, overwrite=True, verbose=False)
    else : 
        # Saving the generated forward solution
        fwd_save_fname = os.path.join(folder_save_fname, "fwd_vertices" + suf + "-fwd.fif")
        mne.write_forward_solution(fwd_save_fname, res_dict['fwd_vertices'], overwrite=True, verbose=False)
        fwd_save_fname = os.path.join(folder_save_fname, "fwd_regions" + suf + "-fwd.fif")
        mne.write_forward_solution(fwd_save_fname, res_dict['fwd_regions'], overwrite=True, verbose=False)
    
    print(f"Saved in : {folder_save_fname}")


##############################################################################
### visualisation ###

chPosx = chanlocs_info["X"]
chPosy = chanlocs_info["Y"]
chPosz = chanlocs_info["Z"]

sPosx = sources_info["positions"][:, 0]
sPosy = sources_info["positions"][:, 1]
sPosz = sources_info["positions"][:, 2]

if args.plot:
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(chPosx, chPosy, chPosz, color="black")
    ax.scatter(sPosx, sPosy, sPosz, marker="+", color="red")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y[m]")
    ax.set_zlabel("z[m]")
    plt.title("Sources and electrods position of the head model")
    plt.show(block=False)

    # check alignement of source and electrodes.
    mne.viz.plot_alignment(
        info=info,
        trans=trans,
        subject=args.subject_name,
        subjects_dir=subjects_dir,
        fwd=fwd,
        src=fwd["src"],
        show_axes=True,
    )
# some info
print(
    f"'\n\nLeadfield dimension : {channels_info['nb_channels']}x{sources_info['nb_sources']}"
)
"""
if args.plot:
    code_folder = os.path.dirname(  os.getcwd() )
    save_folder = os.path.dirname( code_folder )
    plot_bem_kwargs = dict(
        subject=subject,
        brain_surfaces='white', orientation='coronal',
        slices=[50, 100, 150, 200])
    fig = mne.viz.plot_bem(**plot_bem_kwargs)
    plt.savefig(f"{save_folder}/results/bem_surfaces_fsaverage_{sampling}")

    mne.viz.plot_bem(src=src, **plot_bem_kwargs)
    plt.savefig(f"{save_folder}/results/bem_surface_sources_fsaverage_{sampling}")

    plotter = mne.viz.create_3d_figure(size=(600, 400), bgcolor="white")
    mne.viz.plot_alignment(subject=subject,
                                 surfaces='white', coord_frame='mri',
                                 src=src, fig=plotter)
    mne.viz.set_3d_view(plotter, azimuth=173.78)
    fig = plt.figure()
    plt.imshow(plotter._plotter.image)
    plt.savefig(f"{save_folder}/results/brain_surface_sources_fsaverage_{sampling}")
"""
