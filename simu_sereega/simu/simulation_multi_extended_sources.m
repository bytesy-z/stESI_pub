%
% 15.12.2022 : preparing for simulation of a dataset of 
% multiple extended sources

clear; close all
%% PARAMETERS
simu_name       = 'mes_debug';
n_examples      = 100; 


root_folder     = '/home/zik/UniStuff/FYP/stESI_pub';
do_save         = true; 

constrained_orientation = true; % orientation of sources.
volume                  = false ; % set to true for a volume source space, false for surface source space
spher                  = false; 

subject_name       = 'fsaverage';
montage_kind       = 'standard_1020' ; %'easycap-M10'; % electrode montage
src_sampling       = 'ico3';  %spacing between sources


if volume 
    suf = strcat( 'vol_', num2str(src_sampling), '.0');
    constrained_orientation = false; %if volume source space the sources are necessarily unconstrained
elseif spher
    suf = strcat( 'sphere_', num2str(src_sampling), '.0' );
    constrained_orientation = false;
else
    suf = num2str(src_sampling) ;
end
%% General parameters %%% 
order_csv_file = strcat("max_patch_order_",src_sampling, ".csv");


fs          = 512; % samplling frequency (Hz) 
duree       = 500; % time serie duration (ms)
n_times     = fs*duree/1000; % Number of time samples
n_trials    = 1; % number of trials

timeline    = struct( ...
    'n', n_trials,...
    'srate', fs,...
    'length', duree,...
    'marker','event1',...
    'prestim', 0 ); 

fprintf("Sampling frequency: %i \nTrial duration (ms): %i \nNumber of time samples: %i \nNumber of trials: %i\n", ...
    fs, duree, n_times, n_trials)

%% SPATIAL PATTERN PARAMETERS %%%
margin = 2; 
n_patch_min = 1; n_patch_max = 3;%3; 
order_min   = 1; order_max = 5;%3; 

%% TEMPORAL PATTERN PARAMETERS 

amplitude = 10; %1e-9; %nAm
center    = duree/2; %ms 
width     = 60; %ms
sig_type  = 'erp';

erp_dev_intra_patch = struct('ampl', 0, 'width' , 0, 'center', 0  ); 

% Variation between samples: 
% - variation in amplitude
% - variation in center position
% - variation in width
s_amplitude_dev   = 0.5; 
s_center_dev      = 0.5; 
s_width_dev       = 0.02;
base_amplitude  = amplitude; 
base_width      = width; 
base_center     = center; 

s_range_ampl = [ base_amplitude - s_amplitude_dev*base_amplitude; ...
    base_amplitude + s_amplitude_dev*base_amplitude ];
s_range_width = [ base_width - s_width_dev*base_width; ...
    base_width + s_width_dev*base_width ];
s_range_center = [ base_center - s_center_dev*base_center; ...
    base_center + s_center_dev*base_center ];

% Variation between patches of a same sample
p_amplitude_dev   = 0.7; 
p_center_dev      = 0.3; 
p_width_dev       = 0.1;

%% Other params
viz = true;
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% folder path for constrained sources or unconstrained sources
% if constrained : p dipoles, if unconstrained : 3p dipoles (3 dipoles at
% each source position). 
if constrained_orientation 
    folder_path = fullfile( root_folder, 'simulation', subject_name, 'constrained', montage_kind, suf, 'model' ); 
    saving_folder = fullfile( root_folder, 'simulation', subject_name, 'constrained', montage_kind, suf, 'simu' ); 
    suffix_save = fullfile( subject_name, 'constrained', montage_kind, suf, 'simu', simu_name );
else
    folder_path = fullfile( root_folder, 'simulation', subject_name, 'unconstrained', montage_kind, suf, 'model' ); 
    saving_folder = fullfile( root_folder, 'simulation', subject_name, 'unconstrained', montage_kind ,suf, 'simu');
    suffix_save = fullfile( subject_name, 'constrained', montage_kind, suf, 'simu', simu_name );
end

if do_save
    saving_folder = fullfile(saving_folder, simu_name ); 
    
     
    if exist(fullfile(saving_folder, '/sources/Jact'), 'dir') ~= 7
        mkdir( fullfile( saving_folder, '/sources/Jact') );
    end
    if exist(fullfile(saving_folder, '/sources/Jnoise'), 'dir') ~= 7
        mkdir( fullfile( saving_folder, '/sources/Jnoise') );
    end
    if exist(fullfile(saving_folder, '/eeg/infdb'), 'dir') ~= 7
        mkdir( fullfile( saving_folder, '/eeg/infdb') );
    end
    if exist(fullfile(saving_folder, '/md'), 'dir') ~= 7
        mkdir( fullfile( saving_folder, '/md') );
    end
    if exist(fullfile(saving_folder, '/timeline'), 'dir') ~= 7
        mkdir( fullfile( saving_folder, '/timeline') );
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data 
% get the data using the unpack_fwdModel function
[CH, chanlocs, SRC, LFmne, LFsereega] = utl_unpack_fwdModel( ...
    folder_path, suf, constrained_orientation );

% Visualisation
% plot_headmodel(LFsereega); 
% title('Sources and electrodes positions')

n_electrodes = CH.nb_channels;
n_sources  = SRC.nb_sources; 
sPos       = SRC.positions*1e-3; 

%% Compute neighbors from the mesh triangle data
tlh     = load(strcat(folder_path, '/tris_lh_', suf, '.mat'));
trh     = load(strcat(folder_path, '/tris_rh_', suf, '.mat'));
%tris_lh = tlh.tris_lh + 1; % +1 since matlab indexing start to 1.
%tris_rh = trh.tris_rh + 1;
verts = load(strcat(folder_path, '/verts_', suf, '.mat'));
verts = struct("lh", verts.verts_lh + 1, "rh", verts.verts_rh + 1); 
tris = struct("lh", tlh.tris_lh + 1, "rh", trh.tris_rh + 1); 

neighbors = utl_compute_neighbors(tris, verts);


%if ~isfile(order_csv_file) 
dummy = compute_order_max(n_patch_max, margin, neighbors, 15, true, false);
%end
while ~utl_check_max_order(order_csv_file, n_patch_max, order_max, margin)
    order_max = order_max -1; 
    disp("decreased given order max value to match number of patches and margin");
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SIMULATION
tic
for e =1:n_examples
    % to save data
    id      = e;
    seeds   = []; 
    orders  = []; 
    patches = struct();
    
    c_tot   = []; %total components 
    % For each sample
    % - randomly choose number of patch
    % - choose amplitude-width... base value
    % For each patch 
    %   - randomly choose order of the patch
    %   - randomly choose seed of the patch **among available seeds**
    %   - randomly choose signal parameters for the patch 
    %   - -> get_component
    %   - remove activated sources from available sources
    %   - save info of the patch.
    
    % spatial
    n_patch     = randi([n_patch_min, n_patch_max]); 
    
    % temporal
    base_amplitude  = s_range_ampl(1) + (s_range_ampl(2)-s_range_ampl(1))*rand(1);
    base_width      = s_range_width(1) + (s_range_width(2)-s_range_width(1))*rand(1);
    base_center     = s_range_center(1) + (s_range_center(2)-s_range_center(1))*rand(1); 

    p_range_ampl = [ base_amplitude - p_amplitude_dev*base_amplitude; ...
        base_amplitude + p_amplitude_dev*base_amplitude ];
    p_range_width = [ base_width - p_width_dev*base_width; ...
        base_width + p_width_dev*base_width ];
    p_range_center = [ base_center - p_center_dev*base_center; ...
        base_center + p_center_dev*base_center ];
    
    to_remove = [];
    available_sources = 1:n_sources; 
    
    
    for p =1:n_patch
        %spatial
        order   = randi([order_min, order_max]);
        available_sources(to_remove) = [];
        %seed    = randsample(available_sources, 1);
        seed = -1; 
        while ~any(available_sources==seed) 
            seed = randi([1,max(available_sources, [], 'all')],1);
        end
        % temporal
        erp_params          = struct(...
            'ampl',   p_range_ampl(1) + (p_range_ampl(2)-p_range_ampl(1))*rand(1), ...
            'width',  ceil( p_range_width(1) + (p_range_width(2)-p_range_width(1))*rand(1) ), ...
            'center', ceil( p_range_center(1) + (p_range_center(2)-p_range_center(1))*rand(1)) );
        
        % get the components of the patch
        [c,patch, patch_dim] = get_component_extended_src(order, seed, neighbors, sPos,...
            erp_params, erp_dev_intra_patch, LFsereega, timeline );
        
        margin_sources = utl_get_patch( order+margin, seed, neighbors );
        to_remove = [to_remove, margin_sources];
        available_sources = 1:n_sources;
        
        c_tot = [c_tot, c];
        
        patches.(strcat('patch_', num2str(p))) = patch-1; %don't forget -1 for python indices starting from 0 
        orders  = [orders, order];
        seeds   = [seeds, seed-1];
        
        
    end
    [X,source_data] = generate_scalpdata(c_tot, LFsereega, timeline);
    
    if do_save
        act_src_file    = strcat( num2str(id), '_src_act.mat' ) ;
        noise_src_file  = strcat( num2str(id), '_src_noise.mat' );
        md_json_file    = strcat( num2str(id), '_md_json_flie.json'); 
        eeg_infdb_file  = strcat( num2str(id),'_eeg.mat' );
        
        Jact   = struct('Jact', source_data); 
        Jnoise = struct('Jnoise', []); 
        eeg_data    = struct( 'EEG', X );
        
        md_dict = struct(...
            'id', id, ...
            'seeds', seeds, ...
            'orders', orders, ...
            'n_patch', n_patch, ...
            'act_src', patches);
        
        match_dict.( strcat('id_',num2str(id)) ) = struct(...
            'act_src_file_name', fullfile(suffix_save,  'sources', 'Jact', act_src_file), ...
            'noise_src_file_name', fullfile( suffix_save, 'sources', 'Jnoise', noise_src_file ), ...
            'eeg_file_name', fullfile( suffix_save, 'eeg', 'infdb', eeg_infdb_file ), ...
            'md_json_file_name', fullfile(suffix_save, 'md', md_json_file));
        
        act_src_file    = fullfile( saving_folder, 'sources', 'Jact', act_src_file) ;
        noise_src_file  = fullfile( saving_folder, 'sources', 'Jnoise', noise_src_file );
        
        save(act_src_file, 'Jact');
        save(noise_src_file, 'Jnoise');
        
        eeg_infdb_file = fullfile( saving_folder, 'eeg', 'infdb', eeg_infdb_file );
        save(eeg_infdb_file, 'eeg_data'); 

        js_md_dict = jsonencode(md_dict);
        md_json_file = fullfile(saving_folder, 'md', md_json_file); 
        js_file = fopen(md_json_file, 'w'); 
        fprintf(js_file, '%s', js_md_dict);
        fclose(js_file);


        
        
%         if e==1
%             match_json_file = strcat( '_match_json_file.json'); 
% 
%             js_match_dict = jsonencode(match_dict);
%             match_json_file = strcat(saving_folder, '/', simu_name, src_sampling, match_json_file); 
%             js_file = fopen(match_json_file, 'w'); 
%             fprintf(js_file, '%s', js_match_dict);
%             fclose(js_file);
%             
%             clear match_dict;
%         end%
%         
%         if rem(e,100)==0
%             match_json_file = strcat( '_match_json_file.json'); 
% 
%             js_match_dict = jsonencode(match_dict);
%             match_json_file = strcat(saving_folder, '/', simu_name, src_sampling, match_json_file); 
%             js_file = fopen(match_json_file, 'a'); 
%             fprintf(js_file, '%s', js_match_dict);
%             fclose(js_file);
%             
%             clear match_dict;
%         end% save each 1000 examples.
    end%do_save
    
    
    
        
end
simu_time = toc;

%%
ids = 1:n_examples;
if do_save
%    if rem(e,100) ~= 0
        match_json_file = strcat( '_match_json_file.json'); 

        js_match_dict = jsonencode(match_dict);
        match_json_file = strcat(saving_folder, '/', simu_name, src_sampling, match_json_file); 
        js_file = fopen(match_json_file, 'w'); 
        fprintf(js_file, '%s', js_match_dict);
        fclose(js_file);
%    end% json match
    
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
        'trial_ms_duree', duree); 

    general_dict = struct(...
        'electrode_space', electrode_space, ...
        'source_space', source_space, ...
        'rec_info', rec_info, ...
        'ids', ids); 

    js_general_dict = jsonencode(general_dict);
    general_config_file = strcat(saving_folder, '/', simu_name, src_sampling, '_config.json');
    js_file = fopen(general_config_file, 'w'); 
    fprintf(js_file, '%s', js_general_dict);
    fclose(js_file);
    
    %writecell(data_cell, strcat(saving_folder,"/info_simu_examples",simu_name, src_sampling,".csv") );

end

%%
disp("___________________DONE___________________");
disp("simulation took (s): ")
simu_time

hours = floor(simu_time/(60*60)) ;
mins = (simu_time/(60*60) - floor(simu_time/(60*60)) )*60;
disp(strcat("simu time in hours:",num2str(hours),"h",num2str(mins),"mn"));
