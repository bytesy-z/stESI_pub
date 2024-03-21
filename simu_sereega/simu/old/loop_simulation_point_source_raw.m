% 2 OCTOBRE 2022
% From test_save_raw.m
% Test a new saving process: 
% - save raw signals (1 trial per file)
% - each simulation comes with a configuration file which does the mathcing
% of the different file names and source index.
clear; close all; 
%% PARAMETERS
do_save     = true; 

root_folder        = '/home/reynaudsarah/Documents/Data';

% Basic settings
simu_name               = '_test_point_source_';
src_sampling            = 'oct3';  %source space in mne-python. Can be a string or an int (int if volume=true)
montage_kind            = 'easycap-M10'; % electrod montage
constrained_orientation = true; % orientation of sources.

% advanced settings
volume             = false ; % set to true for a volume source space, false for surface source space
sphere             = false; 

if volume 
    suf = strcat( 'vol_', num2str(src_sampling), '.0');
    constrained_orientation = false; %if volume source space the sources are necessarily unconstrained
elseif sphere
    suf = strcat( 'sphere_', num2str(src_sampling), '.0' );
    constrained_orientation = false;
else
    suf = num2str(src_sampling) ;
    clear volume
    clear sphere
end

%%
%%% General parameters %%%
fs       = 512; % Sampling rate (Hz)
duration = 500; % trial duraction (ms)
n_times  = fs*(duration/1000); % number of time samples in one trial
n_trials = 1; % number of trials, 1 = raw data.

n_ex_per_source = 1; %50 % number of examples per source to simulate

timeline = struct( 'n', n_trials, ...
    'srate', fs, ...
    'length', duration, ...
    'marker','event1', ...
    'prestim', 0 ); 

%%% Active sources parameters %%%
n_act_sources    = 1; 

%%% Some plot in the simulation function?
viz      = true;
%% Parameters for ERP
ampl   = 1e-9; %Am
center = duration/2; %ms 
width  = 50; %ms
type   = 'erp';

erp_params = struct( 'ampl', ampl, 'width', width, 'center', center ); 
erp_dev    = struct('ampl', 0.5, 'width' , 0.1, 'center', 0.5  ); 
% TODO: enregistrer ces paramètres!
%%
% folder path for constrained sources or unconstrained sources
if constrained_orientation 
    folder_path     = strcat( ...
        root_folder, '/simulation/constrained/', ...
        montage_kind, '/' ,src_sampling, '/model' ); 
    saving_folder   = strcat(...
        root_folder, '/simulation/constrained/',...
        montage_kind, '/' ,src_sampling, '/simu' ); 
else
    folder_path = strcat(...
        root_folder, '/simulation/unconstrained/', ...
        montage_kind, '/' , src_sampling, '/model' );
    saving_folder   = strcat(...
        root_folder, '/simulation/unconstrained/',...
        montage_kind, '/' ,src_sampling, '/simu'); 
end


if do_save
    saving_folder = strcat(saving_folder, '/', simu_name ); 
    % TODO : faire en sorte de pas essayer de créer un dossier quand il
    % existe en fait déjà...
    mkdir(saving_folder); 
    mkdir( strcat( saving_folder, '/sources') ); 
    mkdir( strcat( saving_folder, '/sources/Jact') );
    mkdir( strcat( saving_folder, '/sources/Jnoise') );
    mkdir( strcat( saving_folder, '/eeg') ); 
    mkdir( strcat( saving_folder, '/eeg/infdb') ); 
end
%% load model
% get the data using the unpack_fwdModel function
[CH, chanlocs, SRC, LFmne, LFsereega] = utl_unpack_fwdModel( folder_path, suf, constrained_orientation );
% Visualisation
plot_headmodel(LFsereega); 
title('Sources and electrods positions')

sPos            = SRC.positions; 
n_sources       = SRC.nb_sources; 
n_electrodes    = CH.nb_channels;
%% Simulation
rng(0); % seed
n_samples = n_sources * n_ex_per_source; 

idx_simulations = randi([1,n_sources], 1, n_samples); 

src_dict        = struct();
eeg_infdb_dict  = struct(); 


incr = 1;

t_vec = linspace(0, duration/1000 - 1/fs, n_times);
figure()
hold on
for i = idx_simulations
    
    c = get_component_Grech(LFsereega, i, timeline, erp_params, erp_dev, false);
    % Generate noise free scalp data
    [X,source_data] = generate_scalpdata(c, LFsereega, timeline,...
        'showprogress',1); % problème ici si je veux mettre showprogress = 0 
    %-> erreur dans la fonction generate scalpadata "Unrecognized function
    %or variable 'progress_start'."
    
%     % add noise to sensors
%     N = rand(size(X));
%     % For each SNR add white noise 
%     alpha_meas = sqrt(SNR_meas)./(1+sqrt(SNR_meas));
%     X_25db = utl_add_snr( X, N, SNR_meas(1) ); %*amplFactor;
%     X_10db = utl_add_snr( X, N, SNR_meas(2) ); %*amplFactor;
%     X_5db = utl_add_snr( X, N, SNR_meas(3) ); %*amplFactor;
    
    
    % Sauvegarde??
    if do_save 
        if constrained_orientation 
            nb_dipoles = SRC.nb_sources; 
        else
            nb_dipoles = SRC.nb_sources*3;
        end
        
               
        act_src_file    = strcat( num2str(incr),'_src_act','.mat' ) ;
        noise_src_file  = strcat( num2str(incr),'_src_noise','.mat' );
        
        
        src_dict.( strcat('id_',num2str(incr)) ) = struct(...
            'idx_act_src', i-1, ...
            'act_src_file_name', act_src_file, ...
            'noise_src_file_name', noise_src_file); 
        
        %noise_src_dict = struct(...
        %    'id', incr, ...
        %    'idx_noise_src', [], ...
        %    'act_src_file_name', noise_src_file); 
        
        eeg_infdb_file  = strcat( num2str(incr), '_eeg.mat');
        eeg_infdb_dict.( strcat('id_',num2str(incr)) ) = struct(...
            'eeg_file', eeg_infdb_file, ...
            'idx_act_src', i-1 ) ;
        
        Jact   = struct('Jact', source_data); 
        Jnoise = struct('Jnoise', []) ;      
        
        act_src_file    = strcat( saving_folder, '/sources/Jact/', act_src_file ) ;
        noise_src_file  = strcat( saving_folder, '/sources/Jnoise/' , noise_src_file );
        save(act_src_file, 'Jact');
        save(noise_src_file, 'Jnoise');
        
        EEG             = struct( 'EEG', X ); 
        eeg_data        = EEG; 
        eeg_infdb_file  = strcat( saving_folder, '/eeg/infdb/',eeg_infdb_file );
        save(eeg_infdb_file, 'eeg_data'); 
        
        if rem(incr,10)== 0
            plot(t_vec, source_data, "LineWidth", 1.5 ) ;
            hold on
        end
    end
    incr = incr + 1;
    
end
hold off
xlabel('Time [s]')
ylabel('Source amplitude')
title('Example of waveform for different samples')

%%
if do_save
    
    js_src_dict     = jsonencode(src_dict);
    src_dict_file   = strcat(saving_folder, '/', simu_name, '_', src_sampling, '_src_files.json'); 
    js_file         = fopen(src_dict_file, 'w'); 
    fprintf(js_file, '%s', js_src_dict);
    fclose(js_file);

    js_eeg_dict     = jsonencode(eeg_infdb_dict);
    eeg_dict_file   = strcat(saving_folder, '/', simu_name, '_', src_sampling, '_eeg_infdb_files.json'); 
    js_file         = fopen(eeg_dict_file, 'w'); 
    fprintf(js_file, '%s', js_eeg_dict);
    fclose(js_file);

        
    electrode_space = struct( ...
        'n_electrodes', n_electrodes, ...
        'electrode_montage', montage_kind); 
    source_space = struct(...
        'n_sources', n_sources,...
        'constrained_orientation', constrained_orientation,...
        'src_sampling', src_sampling); 
    rec_info = struct(...
        'fs', fs, ...
        'n_trials', n_trials,...
        'n_times', n_times, ...
        'trial_ms_duration', duration); 
        
    general_dict = struct(...
        'electrode_space', electrode_space, ...
        'source_space', source_space, ...
        'rec_info', rec_info); 

    
    js_general_dict     = jsonencode(general_dict);
    general_config_file = strcat(root_folder, '/simulation/', simu_name, src_sampling, '_config.json');
    js_file             = fopen(general_config_file, 'w'); 
    fprintf(js_file, '%s', js_general_dict);
    fclose(js_file);
end
%%
% EEG = utl_create_eeglabdataset(X, timeline, LFsereega);
% pop_topoplot(EEG, 1, [400, 450, 500], '', [1 8]);
% pop_eegplot(EEG, 1, 1, 1)
%%

disp( "_____________All done____________" );
