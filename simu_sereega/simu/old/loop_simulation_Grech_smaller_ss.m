% 5 mai 2022.
% Single source in a smaller source space.
%_________________________________________________________________________

clear; close all;
%% PARAMETERS
root_name = '/home/reynaudsarah/Documents/Code/Data/';
do_save = true; 

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

simu_name = '_dl_small_ss_10_'; 

%%% General parameters %%%
% Sampling rate (Hz)
fs = 512; 
% time serie duration (ms)
T = 1000; 
nb_samples = fs*(T/1000); % Le /1000 est du au fait que est en ms
nb_trials = 1000;
% -->
timeline = struct( 'n', nb_trials, 'srate', fs, 'length', T, 'marker','event1', 'prestim', 0 ); 

%%% Active sources parameters %%%
nb_act_sources    = 1; 

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
%src_left  = get_leftMC_src(LFsereega, false); 
%src_right = get_rightMC_src(LFsereega, false); 
n_sources = length(LFmne.pos); 
src_idx_all = [1:n_sources];

sPos    = SRC.positions; 
idx_src_lh  = src_idx_all( sPos(:,1)<0 );
idx_src_rh  = src_idx_all( sPos(:,1)>0 ) ;

% figure()
% scatter3( sPos(idx_src_lh,1), sPos(idx_src_lh,2), sPos(idx_src_lh,3), '.', 'blue' )
% hold on
% scatter3( sPos(idx_src_rh,1), sPos(idx_src_rh,2), sPos(idx_src_rh,3), '.', 'green' )
% hold off
% title('Left hemispher?')

nb_src_per_hem = 5; 

rng(1); %comme seed pytorch.
src_lh = randperm( idx_src_lh(end), nb_src_per_hem );
src_rh = randperm( idx_src_rh(end)-idx_src_rh(1), nb_src_per_hem );

source_space = [src_lh, src_rh+idx_src_rh(1)] 
%%
erp_params = struct('ampl', 1e-9, 'width', 20, 'center', 500);
dev = [0.5, 0.1, 0.5];
for i = source_space 
    % Component definition -> in the function 
    c = get_component_Grech(LFsereega, i, timeline, erp_params, dev, false);

    % Generate noise free scalp data
    [X,source_data] = generate_scalpdata(c, LFsereega, timeline);

    N = rand(size(X));
    % For each SNR add white noise 
    alpha_meas = sqrt(SNR_meas)./(1+sqrt(SNR_meas));
    %X_25db = utl_mix_data( X, N, alpha_meas(1))*amplFactor;
    X_25db = utl_add_snr( X, N, SNR_meas(1) ); %*amplFactor;
    %X_15db = utl_mix_data( X, N, alpha_meas(2))*amplFactor; 
    X_15db = utl_add_snr( X, N, SNR_meas(2) ); %*amplFactor;
    %X_10db = utl_mix_data( X, N, alpha_meas(3))*amplFactor; 
    X_10db = utl_add_snr( X, N, SNR_meas(3) ); %*amplFactor;
    %X_5db  = utl_mix_data( X, N, alpha_meas(4))*amplFactor;
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
        
        tmp = strcat('_src_',num2str(i-1) );
        
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
test = source_data(1, 256, : ); 
test = test(1,:); 

figure(); 
stem( test);
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

    