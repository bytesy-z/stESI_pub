import sys

import numpy as np
from utils.utl import load_mat

import mne
from mne.datasets import sample
from mne.io import read_raw_fif
def make_sample_montage():
    data_path = sample.data_path() 
    fname_raw = data_path / "MEG" / "sample" / "sample_audvis_raw.fif"
    raw = read_raw_fif(fname_raw) 
    raw.pick_types(meg=False, eeg=True, stim=False, exclude=()).load_data()
    raw.pick_types(eeg=True)

    dig_montage = raw.info.get_montage()
    return dig_montage

class ElectrodeSpace:
    """  
    Get, build and store information about the electrode space

    - n_electrodes      : number of electrodes
    - positions         : positions of the electrodes
    - montage_kind      : name of the electrode montage used
    - electrode_names   : name of the electrodes
    - electrode_montage : electrode montage of mne-python (DigMontage),
                         useful to manipulate eeg data
    - info              : info object from mne-python
    - fs                : sampling frequency
    

    @TODO : add visualisation function to plot electrodes in 2D or 3D
    """

    def __init__(self, folders, general_config_dict):
        """ 
        - folders: FolderStructure object containing all the name of the folders
        - general_config_dict: dictionnary with information about simulation configuration
        """

        # load the ch_source_sampling.mat file which contains basic information and data of the electrode space
        electrode_info = load_mat(
            f"{folders.model_folder}/ch_{general_config_dict['source_space']['src_sampling']}.mat")

        self.n_electrodes = electrode_info['nb_channels']
        self.positions = electrode_info['positions']
        self.montage_kind = general_config_dict['electrode_space']['electrode_montage']
        self.electrode_names = [k for k in electrode_info['names']]

        # recreate the electrode montage from mne python
        if self.montage_kind in mne.channels.get_builtin_montages( ): 
            self.electrode_montage = mne.channels.make_standard_montage(
                self.montage_kind)
        #elif self.montage_kind == "spm": 
        #    self.electrode_montage = ld.make_spm_montage() 
        elif self.montage_kind == "sample":
            self.electrode_montage = make_sample_montage()
        else: 
            sys.exit("Error: unknown electrode montage")

        if self.montage_kind == "standard_1020": 
            exclude_mdn             = ['T3', 'T4', 'T5', 'T6']
            ids_duplicate = []
            for e in exclude_mdn:
                ids_duplicate.append( np.where( [ch==e for ch in self.electrode_montage.ch_names] )[0][0] )
            ch_names = list( np.delete(self.electrode_montage.ch_names, ids_duplicate) )
            
            self.info = mne.create_info(
                ch_names, 
                general_config_dict['rec_info']['fs'], 
                ch_types='eeg', verbose=False)            
        else : 
            self.info = mne.create_info(
                self.electrode_montage.ch_names, general_config_dict['rec_info']['fs'], ch_types='eeg', verbose=None)
        
        self.info.set_montage(self.electrode_montage)

        self.fs = general_config_dict['rec_info']['fs']

    def _attributes(self):
        """
        return the liste of attribute/variables of the object
        """
        return list(self.__dict__.keys())


class SourceSpace:
    """  
    - src_sampling  : name of the source subsampling used to subsample the source space ('oct3', 'ico3'...)
    - n_sources     : number of sources
    - constrained   : True if constrained orientation, False if unconstrained
    - positions     : source positions
    - orientations  : source orientations (values are filled during HeadModel initialization)

    @TODO: add visualisation of source positions
    """
    def __init__(self, folders, general_config_dict, surface=True, volume=False):
        self.src_sampling   = general_config_dict['source_space']['src_sampling']
        #self.n_sources      = general_config_dict['source_space']['n_sources']
        self.constrained    = general_config_dict['source_space']['constrained_orientation']

        source_info = load_mat(
            f"{folders.model_folder}/sources_{self.src_sampling}.mat")

        self.positions = source_info['positions']
        self.n_sources = self.positions.shape[0]
        self.orientations = []  # to complete

        # useless for now
        self.surface = surface
        self.volume = volume

    def _attributes(self):
        """
        return the liste of attribute/variables of the object
        """
        return list(self.__dict__.keys())

class HeadModel:
    """  
    Gather electrode space and source space + forward solution
    - electrode_space   : ElectrodeSpace object
    - source_space      : SourceSpace object
    - subject_name      : default is 'fsaverage', name of the subject used.
    - fwd               : mne python Forward ojbect created during head model generation
    - leadfield         : leadfield matrix

    @TODO : add visualizaion of electrode and sources
    """
    def __init__(self, electrode_space, source_space, folders, subject_name='fsaverage'):
        self.electrode_space    = electrode_space
        self.source_space       = source_space

        self.subject_name       = subject_name
        # get the forward object from mne python
        fwd = mne.read_forward_solution(
            f"{folders.model_folder}/fwd_{source_space.src_sampling}-fwd.fif",
            verbose=False)
        # constrain source orientation if necessary
        self.fwd = mne.convert_forward_solution(
            fwd,
            surf_ori=source_space.constrained,
            force_fixed=source_space.constrained,
            use_cps=True, verbose=0)

        self.leadfield = self.fwd['sol']['data']
        # add orientation to source space
        self.source_space.orientations = self.fwd['source_nn']

    def _attributes(self):
        """
        return the liste of attribute/variables of the object
        """
        return list(self.__dict__.keys())