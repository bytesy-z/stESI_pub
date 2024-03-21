% 8 février 2022
% Simulation avec 2 sources ponctuelles. 
% bruit blanc
% pas d'extension spatiale
%_________________________________________________________________________
% TODO 
% - mettre une distance donnée entre les deux sources? pour avoir des
% configs proches et des configs + espacées.
%_________________________________________________________________________

clear; close all;
%% PARAMETERS
root_name = '/home/reynaudsarah/Documents/Code/Data/';

source_constrained = true; %false;
spacing            = 'ico3'; 
suf                = strcat( '_', spacing); 
elec_montage       = 'easycap-M10'; 

%%% General parameters %%%
% Sampling rate (Hz)
fs = 1000; 
% time serie duration (ms)
T = 1000; % i.e 2 seconds
nb_samples = fs*T/1000; % Le /1000 est du au fait que n est en ms
nb_trials = 1;

%%% Active sources parameters %%%
nb_act_sources    = 2; 

%%% Source activation signal parameters %%%
ampl = 1*10^-9; %nA
center = 500; %ms 
width = 20; %ms
type = 'erp';

SNR_meas = [25, 15, 10, 5]; %dB
% Some plot in the simulation function?
viz = true;
%%
% folder path for constrained sources or unconstrained sources
% if constrained : p dipoles, if unconstrained : 3p dipoles (3 dipoles at
% each source position). 
if source_constrained 
    folder_path = strcat( root_name, '/simulation/constrained/', elec_montage, '/' ,suf, '/model' ); 
else
    folder_path = strcat( root_name, '/simulation/unconstrained/', elec_montage, '/' ,suf, '/model' ); 
end
%%
% get the data using the unpack_fwdModel function
[CH, chanlocs, SRC, LFmne, LFsereega] = utl_unpack_fwdModel( folder_path, suf, source_constrained );
% Visualisation
plot_headmodel(LFsereega); 
title('Sources and electrods positions')
%%
sPos = SRC.positions; 
src_idx = lf_get_source_random(LFsereega, nb_act_sources);
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

c1 = get_component_Grech(LFsereega, src_idx(1), timeline, false);
c2 = get_component_Grech(LFsereega, src_idx(2), timeline, false);

c = [c1, c2]; 

% Generate noise free scalp data
[X,source_data] = generate_scalpdata(c, LFsereega, timeline);

EEG = utl_create_eeglabdataset(X, timeline, LFsereega);

pop_topoplot(EEG, 1, [400, 450, 500, 550, 600], '', [1 8]);
pop_eegplot(EEG, 1, 1, 1)

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

figure()
subplot(212)
subplot(211)
plot(source_data(1,:)); hold on ; 
plot(source_data(2,:)); hold off
title('Source activity')
subplot(212)
plot(X(1,:))
title('EEG for channel 1')
%%
plot_source_location(src_idx, LFsereega);
title('Source visualization')