% 26 janvier 2022
% Simulation inspirée de la simulation utilisée dans l'article 
% Review on solving the inverse problem in EEG source analysis, Grech et
% al.
% In a loop for different source positions. 
% What changes btwn trials : amplitude of the source, noise. 
% From 1 source distribution -> 4 different SNR measurements (+ a noiseless
% measurement).

%_________________________________________________________________________
% TODO 
% - Amplitude des signaux? -> ça déconne dans la fonction de sereega, sans
%    même ajouter de bruit pour un amplitude de signal source de 10-9 il me
%    crée un truc pas du tout en 10-9 --> problème réglé : c'était le
%    paramètre peakAmplitudeDv qui posait pblm. valeure tq les valeur de
%    peak vont varier entre (peakAmplitude-peakAmplitudeDv) et (peakAmplitude
%    + peakAmplitudeDv). Donc si tu mets 50 pour une amplitude de 10^-9
%    forcément...

% - Que faire du truc du moment dipolaire??
%_________________________________________________________________________

clear; close all;
%% PARAMETERS
root_folder = '/home/reynaudsarah/Documents/Data/';
do_save     = true; 

source_constrained = true; % orientation of sources.
volume             = false ; % set to true for a volume source space, false for surface source space
sphere             = false; 

spacing            = 'oct3';  %spacing between sources. Can be a string or an int (int if volume=true)
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

%simu_name = '_test_grech_'; 
simu_name  = '_dl_100ms_'; 
%%% General parameters %%%
fs       = 512; % Sampling rate (Hz)
duration = 100; %512; % ms
n_times  = fs*(duration/1000); % /1000 as duration is in ms
n_trials = 100;
% -->
timeline = struct( 'n', n_trials, 'srate', fs, 'length', duration, 'marker','event1', 'prestim', 0 ); 

%%% Active sources parameters %%%
n_act_sources    = 1; 

SNR_meas = [25, 15, 10, 5]; %dB
% Some plot in the simulation function?
viz = true;
%%
% folder path for constrained sources or unconstrained sources
% if constrained : p dipoles, if unconstrained : 3p dipoles (3 dipoles at
% each source position). 
if source_constrained 
    folder_path = strcat( root_folder, '/simulation/constrained/', elec_montage, '/' ,spacing, '/model' ); 
    saving_folder = strcat( root_folder, '/simulation/constrained/', elec_montage, '/' ,spacing, '/simu' ); 
else
    folder_path = strcat( root_folder, '/simulation/unconstrained/', elec_montage, '/' ,spacing, '/model' );
    saving_folder = strcat( root_folder, '/simulation/unconstrained/', elec_montage, '/' ,spacing, '/simu'); 
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
sPos    = SRC.positions; 
src_idx = SRC.nb_sources; 
nb_src  = length(src_idx); 

% Parameters for ERP
ampl   = 1; %e-9; %nA
center = duration/2; %ms 
width  = 50; %ms
type   = 'erp';
erp_params = struct( 'ampl', ampl, 'center', center, 'width', width ); 
erp_dev    = struct('ampl', 0.5, 'width' , 0.1, 'center', 0.5  ); 

for i = 1:src_idx 
    c = get_component_Grech(LFsereega, i, timeline, erp_params, erp_dev, false);

    % Generate noise free scalp data
    [X,source_data] = generate_scalpdata(c, LFsereega, timeline);

    N = rand(size(X));
    % For each SNR add white noise 
    alpha_meas = sqrt(SNR_meas)./(1+sqrt(SNR_meas));
    X_25db = utl_add_snr( X, N, SNR_meas(1) ); %*amplFactor;
    X_15db = utl_add_snr( X, N, SNR_meas(2) ); %*amplFactor;
    X_10db = utl_add_snr( X, N, SNR_meas(3) ); %*amplFactor;
    X_5db = utl_add_snr( X, N, SNR_meas(4) ); %*amplFactor;

    % Sauvegarde??
    if do_save 
        if source_constrained 
            nb_dipoles = SRC.nb_sources; 
        else
            nb_dipoles = SRC.nb_sources*3;
        end
    
        Jact = struct('idx', i-1,... %-1 to match indexation in python
            'position', sPos(i,:), ...
            'signal', source_data, ...
            'nb', length(i), ...
            'nb_dipoles', nb_dipoles ) ; 
        Jnoise = struct('idx', [],... %-1 to match indexation in python
            'position', [], ...
            'signal', [], ...
            'nb', 0, ...
            'nb_dipoles', nb_dipoles ) ; 
        
        EEG      = struct( 'EEG', X, 'fs', fs); 
        EEG_25db = struct( 'EEG', X_25db, 'fs', fs); 
        EEG_15db = struct( 'EEG', X_15db, 'fs', fs); 
        EEG_10db = struct( 'EEG', X_10db, 'fs', fs); 
        EEG_5db  = struct( 'EEG', X_5db, 'fs', fs); 
        
        tmp = strcat( '_src_',num2str(i-1) );
        
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
    end
    
end

if do_save
    event_data = timeline;
    save(strcat(saving_folder, '/timeline/Epochs','.mat'), 'event_data');
end
% EEG = utl_create_eeglabdataset(X, timeline, LFsereega);
% pop_topoplot(EEG, 1, [400, 450, 500], '', [1 8]);
% pop_eegplot(EEG, 1, 1, 1)
%% Check des amplitudes...
[a, t] = max(source_data); 
vec = 1:100; 
figure(); 
stem( vec, squeeze(a), 'o', 'lineWidth', 2 );
xlabel('Trial nb°') ; ylabel('Max amplitude of the simulated peak'); 
title('Values of maximum amplitude for each trial (active source signal)'); 

t_vec = 0:1/fs:duration/1000-1/fs;
figure()
subplot(2,2,4)
subplot(2,2,1)
plot(t_vec, X_25db(1,:,1)); xlabel('Time'); ylabel('Amplitude')
title('25db')
subplot(2,2,2)
plot(t_vec, X_15db(1,:,1))
title('15db'); xlabel('Time'); ylabel('Amplitude')
subplot(2,2,3)
plot(t_vec, X_10db(1,:,1))
title('10db'); xlabel('Time'); ylabel('Amplitude')
subplot(2,2,4)
plot(t_vec, X_5db(1,:,1))
title('5db'); xlabel('Time'); ylabel('Amplitude')