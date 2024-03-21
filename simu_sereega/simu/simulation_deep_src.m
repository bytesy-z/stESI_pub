% 17 f&vrier 2022
% Comme Grech mais avec une source "profonde" i.e placée dans le volume et
% non pas sur la surface corticale. 
%_________________________________________________________________________
% TODO 
%_________________________________________________________________________

clear; close all;
%% PARAMETERS
root_name = '/home/reynaudsarah/Documents/Code/Data/';
do_save = true; 

source_constrained = true; % orientation of sources.
volume             = true ; % set to true for a volume source space, false for surface source space
sphere             = false; 

spacing            = 10.0;  %spacing between sources. Can be a string or an int (int if volume=true)
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

simu_name = '_deep_source_';

do_save = true;
%%% General parameters %%%
% Sampling rate (Hz)
fs = 512; 
% time serie duration (ms)
T = 1000; % i.e 2 seconds
nb_samples = fs*T/1000; % Le /1000 est du au fait que n est en ms
nb_trials = 100;

%%% Active sources parameters %%%
nb_act_sources    = 1; 

%%% Source activation signal parameters %%%
ampl = 1e-9; %nA
center = 500; %ms 
width = 100; %ms
type = 'erp';

SNR_meas = [25, 15, 10, 5]; %dB
% Some plot in the simulation function?
viz = true;
%%
% folder path for constrained sources or unconstrained sources
% if constrained : p dipoles, if unconstrained : 3p dipoles (3 dipoles at
% each source position). 
if source_constrained 
    folder_path = strcat( root_name, 'simulation/constrained/', elec_montage, '/' ,suf, '/model' ); 
    saving_folder = strcat( root_name, 'simulation/constrained/', elec_montage, '/' ,suf, '/simu' ); 
else
    folder_path = strcat( root_name, 'simulation/unconstrained/', elec_montage, '/' ,suf, '/model' ); 
    saving_folder = strcat( root_name, 'simulation/unconstrained/', elec_montage, '/' ,suf, '/simu'); 
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
%%
% get the data using the unpack_fwdModel function
[CH, chanlocs, SRC, LFmne, LFsereega] = utl_unpack_fwdModel( folder_path, suf, source_constrained );
% Visualisation
plot_headmodel(LFsereega); 
title('Sources and electrods positions')
%%
sPos    = SRC.positions; 
src_idx = lf_get_source_random(LFsereega, nb_act_sources);

% source position without source position repetitions 
% offset for direction : 
% o_dir = 0 : souce simulated in x direction 
% o_dir = 1 : source_simulated in y direction 
% o_dit = 2 : source simulated in z direction
src_id = floor(src_idx/3); 
o_dir  = mod(src_idx, 3); 

src_pos = sPos(src_idx, :); 

timeline = struct( 'n',nb_trials, 'srate', fs, 'length', T, 'marker','event1', 'prestim', 0 ); 

% % Component definition -> function
% erp = struct( ...
%         'peakLatency', center,  ...    % in ms, starting at the start of the epoch
%         'peakWidth', width,   ...     % in ms
%         'peakAmplitude', ampl, ...
%         'peakAmplitudeDv', 50); 
% 
% erp = utl_check_class(erp,'type','erp');
% if viz
%     plot_source_location(src_idx, LFsereega);
%     title('Source visualization')
%     
%     plot_signal_fromclass(erp, timeline);
%     xlabel('Time [ms]'); ylabel('Amplitude [µV]');
%     title('Simulated ERP'); 
% end
% 
% c = struct();
% c.source = src_idx;       
% c.signal = {erp};
% c = utl_check_component(c, LFsereega);
erp_params = struct('ampl', 1e-9, 'width', 100, 'center', 500);
dev = [0, 0, 0];
c   = get_component_Grech(LFsereega, src_idx, timeline, erp_params, dev, false);


% Generate noise free scalp data
%% TMP
signal = generate_signal_fromcomponent(c, timeline); 
figure()
plot(signal)
title('Source signal...')
%%
[X,source_data] = generate_scalpdata(c, LFsereega, timeline);

% EEG = utl_create_eeglabdataset(X, timeline, LFsereega);
% pop_topoplot(EEG, 1, [400, 450, 500, 550, 600], '', [1 8]);
% pop_eegplot(EEG, 1, 1, 1)

N = rand(size(X));
% For each SNR add white noise 
alpha_meas = sqrt(SNR_meas)./(1+sqrt(SNR_meas));
X_25db = utl_mix_data( X, N, alpha_meas(1)); 
X_15db = utl_mix_data( X, N, alpha_meas(2)); 
X_10db = utl_mix_data( X, N, alpha_meas(3)); 
X_5db =  utl_mix_data( X, N, alpha_meas(4)); 


% Transform in EEGlab dataset to visualize topomaps...
% EEG_25db = utl_create_eeglabdataset(X_25db, timeline, LFsereega);
% pop_topoplot(EEG_25db, 1, [400, 450, 500, 550, 600], '', [1 8]);
% pop_eegplot(EEG_25db, 1, 1, 1)
% 
% EEG_15db = utl_create_eeglabdataset(X_15db, timeline, LFsereega);
% pop_topoplot(EEG_15db, 1, [400, 450, 500, 550, 600], '', [1 8]);
% pop_eegplot(EEG_15db, 1, 1, 1)
% 
% EEG_10db = utl_create_eeglabdataset(X_10db, timeline, LFsereega);
% pop_topoplot(EEG_10db, 1, [400, 450, 500, 550, 600], '', [1 8]);
% pop_eegplot(EEG_10db, 1, 1, 1)

%%
if do_save 
    if source_constrained 
        nb_dipoles = SRC.nb_sources; 
    else
        nb_dipoles = SRC.nb_sources*3;
    end
    %reel_src_idx = floor((src_idx-1)/3);
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
        
    tmp = strcat( '_src_',num2str( src_idx-1 ) );
    
    save(strcat(saving_folder, '/sources/Jact',tmp,'.mat'),'Jact');
    save(strcat(saving_folder, '/sources/Jnoise', tmp,'.mat'),'Jnoise');
    eeg_data = EEG; 
    save(strcat(saving_folder, '/eeg/EEG', tmp,'.mat'),'eeg_data'); 
    eeg_data = EEG_25db; 
    save(strcat(saving_folder, '/eeg/EEG_25db', tmp,'.mat'),'eeg_data');
    eeg_data = EEG_15db; 
    save(strcat(saving_folder, '/eeg/EEG_15db', tmp,'.mat'),'eeg_data');
    eeg_data = EEG_10db; 
    save(strcat(saving_folder,  '/eeg/EEG_10db', tmp,'.mat'),'eeg_data'); 
    eeg_data = EEG_5db; 
    save(strcat(saving_folder,  '/eeg/EEG_5db', tmp,'.mat'),'eeg_data'); 
    
    save(strcat(saving_folder, '/timeline/Epochs','.mat'), 'event_data');
end

% %%
% 
% figure()
% subplot(212)
% subplot(211)
% plot(source_data(:,:,1))
% title('Source activity')
% subplot(212)
% plot(X(1,:))
% title('EEG for channel 1')