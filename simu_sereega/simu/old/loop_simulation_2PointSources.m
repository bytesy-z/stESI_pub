% 8 février 2022
% Simulation avec 2 sources ponctuelle + bruit blanc
% In a loop for different source positions. 
% What changes btwn trials : amplitude of the source, noise. 
% From 1 source distribution -> 4 different SNR measurements (+ a noiseless
% measurement).

%_________________________________________________________________________
% TODO 
%_________________________________________________________________________

clear; close all;
%% PARAMETERS
root_name = '/home/reynaudsarah/Documents/Code/Data/';
do_save = true; 

source_constrained = true; % orientation of sources.
volume             = false ; % set to true for a volume source space, false for surface source space
sphere             = false; 

spacing            = 'ico3';  %spacing between sources. Can be a string or an int (int if volume=true)
elec_montage       = 'easycap-M10'; % electrod montage

if volume 
    suf = strcat( 'vol_', num2str(spacing), '.0');
    source_constrained = false; %if volume source space the sources are necessarily unconstrained
elseif sphere
    suf = strcat( 'sphere_', num2str(spacing), '.0' );
    source_constrained = false;
else
    suf = num2str(spacing) ;
end

simu_name = '_2_psource_spaced_'; 

train_name = strcat( simu_name, '/train' );
test_name = strcat( simu_name, '/train' );

%%% General parameters %%%
% Sampling rate (Hz)
fs = 1000; 
% time serie duration (ms)
T = 1000; % i.e 2 seconds
nb_samples = fs*T/1000; % Le /1000 est du au fait que n est en ms
nb_trials = 100;
% -->
timeline = struct( 'n',nb_trials, 'srate', fs, 'length', T, 'marker','event1', 'prestim', 0 ); 

%%% Active sources parameters %%%
nb_act_sources    = 2; 

%%% Source activation signal parameters %%%
% ampl = 1*10^-9; %nA
% center = 500; %ms 
% width = 20; %ms
% type = 'erp';

SNR_meas = [25, 15, 10, 5]; %dB
% Some plot in the simulation function?
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
    mkdir( strcat( saving_folder, '/sources/Jact') );
    mkdir( strcat( saving_folder, '/sources/Jnoise') );
    mkdir( strcat( saving_folder, '/eeg') ); 
    mkdir( strcat( saving_folder, '/eeg/infdb') ); 
    for snr = SNR_meas 
        mkdir( strcat( saving_folder, '/eeg/', num2str(snr), 'db') );
    end
    mkdir( strcat( saving_folder, '/timeline') ); 
end
%%
% get the data using the unpack_fwdModel function
[CH, chanlocs, SRC, LFmne, LFsereega] = utl_unpack_fwdModel( folder_path, suf, source_constrained );
% Visualisation
plot_headmodel(LFsereega); 
title('Sources and electrods positions')
%%
nb_pairs = 10; 
sPos = SRC.positions; 

dev = [0.5, 0.1, 0];
erp_params = struct('ampl', 1e-9, 'center', 500, 'width', 20); 
for i = 1:nb_pairs
    %src_idx = randi(SRC.nb_sources, [nb_act_sources,1]); 
    
    src_idx = lf_get_source_spaced(LFsereega, 2, 50);
    
    % Component definition -> in the function 
    c1 = get_component_Grech(LFsereega, src_idx(1), timeline, erp_params, dev, false);
    c2 = get_component_Grech(LFsereega, src_idx(2), timeline, erp_params, dev, false);
    
    c = [c1, c2];
    % Generate noise free scalp data
    [X,source_data] = generate_scalpdata(c, LFsereega, timeline);

    N = rand(size(X));
    % For each SNR add white noise 
    alpha_meas = sqrt(SNR_meas)./(1+sqrt(SNR_meas));
    %X_25db = utl_mix_data( X, N, alpha_meas(1))*amplFactor;
    X_25db = utl_add_snr( X, N, SNR_meas(1));
    %X_15db = utl_mix_data( X, N, alpha_meas(2))*amplFactor; 
    X_15db = utl_add_snr( X, N, SNR_meas(2));
    %X_10db = utl_mix_data( X, N, alpha_meas(3))*amplFactor; 
    X_10db = utl_add_snr( X, N, SNR_meas(3));
    %X_5db  = utl_mix_data( X, N, alpha_meas(4))*amplFactor;
    X_5db = utl_add_snr( X, N, SNR_meas(4));
    
    % Sauvegarde??
    if do_save 
        if source_constrained 
            nb_dipoles = SRC.nb_sources; 
        else
            nb_dipoles = SRC.nb_sources*3;
        end
    
        Jact = struct('idx', src_idx-1,... %-1 to match indexation in python
            'position', sPos(src_idx,:), ...
            'signal', source_data, ...
            'nb', nb_act_sources, ...
            'nb_dipoles', nb_dipoles ) ; 
        Jnoise = struct('idx', [],... %-1 to match indexation in python
            'position', [], ...
            'signal', [], ...
            'nb', 0, ...
            'nb_dipoles', nb_dipoles ) ; 
        EEG = struct( 'EEG', X, 'fs', fs); 
        EEG_25db = struct( 'EEG', X_25db, 'fs', fs); 
        EEG_15db = struct( 'EEG', X_15db, 'fs', fs); 
        EEG_10db = struct( 'EEG', X_10db, 'fs', fs); 
        EEG_5db  = struct(  'EEG', X_5db, 'fs', fs); 
        
        event_data = timeline; 
        
        tmp = strcat('_src_',num2str(src_idx(1)), '_', num2str(src_idx(2)) );
        
        save(strcat(saving_folder, '/sources/Jact/Jact',tmp,'.mat'),'Jact');
        save(strcat(saving_folder, '/sources/Jnoise/Jnoise', tmp,'.mat'),'Jnoise');
        eeg_data = EEG; 
        save(strcat(saving_folder, '/eeg/infdb/EEG', tmp,'.mat'),'eeg_data'); 
        eeg_data = EEG_25db; 
        save(strcat(saving_folder, '/eeg/25db/EEG_25db', tmp,'.mat'),'eeg_data');
        eeg_data = EEG_15db; 
        save(strcat(saving_folder, '/eeg/15db/EEG_15db', tmp,'.mat'),'eeg_data');
        eeg_data = EEG_10db; 
        save(strcat(saving_folder,  '/eeg/10db/EEG_10db', tmp,'.mat'),'eeg_data'); 
        eeg_data = EEG_5db; 
        save(strcat(saving_folder,  '/eeg/5db/EEG_5db', tmp,'.mat'),'eeg_data'); 
        
        disp(i);
    end
    
end

if do_save
    save(strcat(saving_folder, '/timeline/Epochs','.mat'), 'event_data');
end

% 
% EEG = utl_create_eeglabdataset(X, timeline, LFsereega);
% pop_topoplot(EEG, 1, [400, 450, 500, 550, 600], '', [1 8]);
% pop_eegplot(EEG, 1, 1, 1)
%     
%% Check des amplitudes...
test = source_data(1, 500, : ); 
test = test(1,:); 

figure(); 
stem(test);
xlabel('Trial nb°') ; ylabel('Max amplitude of the simulated peak'); 
title('Values of peaks amplitude for each trial, for channel 1'); 


figure()
subplot(2,2,4)
subplot(2,2,1)
plot(X_25db(1,:,1))
title('25db')
subplot(2,2,2)
plot(X_15db(1,:,1))
title('15db')
subplot(2,2,3)
plot(X_10db(1,:,1))
title('10db')
subplot(2,2,4)
plot(X_5db(1,:,1))
title('5db')
%%

    