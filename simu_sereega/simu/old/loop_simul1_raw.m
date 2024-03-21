% 18 janvier 2022
% Simulation un peu plus propre en utilisant + de fonctions pour les
% différentes étapes
% Simulation de données de type "raw" dans mne-python i.e une seule série
% temporelle.
clear; close all;
%% PARAMETERS
root_name = '/home/reynaudsarah/Documents/Code/Data/';

source_constrained = true; %false;
spacing            = 'oct3'; 
suf                = strcat( '_', spacing); 
elec_montage       = 'easycap-M10'; 
simu_name = 'loop_raw'; simu_name = strcat('_', simu_name); 
do_save = true; 

%%% General parameters %%%
fs = 1000; % Sampling rate (Hz)
T = 2000; % time serie duration (ms), i.e 2 seconds here
nb_samples = fs*T/1000; % Le /1000 est du au fait que n est en ms

%%% Active sources parameters %%%
nb_act_sources    = 2; 
random_sources   = true; %sources selected randomly or not?
% random_sources = false : 
act_src_idx = []; % give the source index you want to use. @TODO : mieux faire ça

%%% Source activation signal parameters %%%
type = 'erp';

%%% Noise sources parameters %%%
% ratio of the number of sources to activate as noise sources 
ratio = 1.0/4.0;
% nb_noise_src = ratio*SRC.nb_sources;
% noise_src_idx = lf_get_source_spaced(LFsereega, nb_noise_src, 2); 
% noise_src_pos = sPos(noise_src_idx);
SNRsrc = 10; SNRmeas = 10;
noise_coeff = struct('eyes', 0.05 , 'anat', 0.1 , 'colored', 1, 'elect', 0.1 );
SNRs = [SNRsrc, SNRmeas];

viz = true;
%%
% folder path for constrained sources or unconstrained sources
% if constrained : p dipoles, if unconstrained : 3p dipoles (3 dipoles at
% each source position). 
if source_constrained 
    folder_path = strcat( root_name, '/simulation/constrained/', elec_montage, '/' ,suf, '/model' ); 
    saving_folder = strcat( root_name, '/simulation/constrained/', elec_montage, '/' ,suf, '/simu' ); 
else
    folder_path = strcat( root_name, '/simulation/unconstrained/', elec_montage, '/' ,suf, '/model' );
    saving_folder = strcat( root_name, '/simulation/unconstrained/', elec_montage, '/' ,suf, '/simu'); 
end
if do_save
    saving_folder = strcat(saving_folder, '/', simu_name ); 
    % TODO : faire en sorte de pas essayer de créer un dossier quand il
    % existe en fait déjà...
    mkdir(saving_folder); 
    mkdir( strcat( saving_folder, '/sources') ); 
    mkdir( strcat( saving_folder, '/eeg') ); 
    mkdir( strcat( saving_folder, '/timeline') ); 
    
end

% get the data using the unpack_fwdModel function
[CH, chanlocs, SRC, LFmne, LFsereega] = utl_unpack_fwdModel( folder_path, suf, source_constrained );
% Visualisation
plot_headmodel(LFsereega); 
title('Sources and electrods positions')

% Timeline :
epochs = struct( 'n',1, 'srate', fs, 'length', T, 'marker','event1', 'prestim', 0 ); 
%% Sources positions : 
sPos = LFsereega.pos; % possible source positions

% Values of amplitude, center and width will be taken in these range of
% values to simulate different signals
range_ampl = [10 40]*10^-9; 
range_center = [300 300]; 
range_width = [50 100]; 

nb_ite = 5; 
for k = 1:nb_ite
    [ampl, center, width] =erp_random_parameters(range_ampl, range_center, range_width); 
    

    if random_sources 
        act_src_idx = lf_get_source_random(LFsereega, nb_act_sources);
    else
        [act_src_idx, ~] = get_leftMC_src(LFsereega, false);
    end
    
    act_src_pos = sPos(act_src_idx, :);
    disp(strcat('Active source(s) index : ', num2str(act_src_idx)));

    act_sig_params = struct('type', 'erp', 'ampl', ampl, 'center', center,...
        'width', width);
    src_params = struct('random_sources', true, 'nb_act_src', nb_act_sources,...
        'act_src_idx', act_src_idx, 'act_sig', act_sig_params, 'noise_src_ratio', ratio);


    [J, M, SNRres, src_info] = sim_simul1(LFmne, LFsereega, src_params, ...
        epochs, noise_coeff, SNRs, false);
end
