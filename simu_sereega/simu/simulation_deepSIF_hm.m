% 2023.07.17 : simulation of single/multiple extended sources based on the
% head model used in the deepSIF article and simulations i.e region based
% model from fsaverage5, with 994 regions from a 20k source space.
% Electrode montage hase 75 positions

clear; close all;
home = getenv('HOME'); 

anatomy_folder = strcat( home, "/Documents/deepsif/DeepSIF-main/anatomy/" );
%%
electrode_montage = load( ...
    strcat(anatomy_folder, "electrode_75.mat")).eloc75; 
n_electrodes = numel(electrode_montage);

ePos = zeros(n_electrodes, 3);

ePos(:,1) = [electrode_montage.X]; 
ePos(:,2) = [electrode_montage.Y]; 
ePos(:,3) = [electrode_montage.Z]; 

% figure()
% scatter3(ePos(:,1), ePos(:,2), ePos(:,3), 'filled');

%%
source_space = load(...
    strcat(anatomy_folder, "fs_cortex_20k_inflated.mat"));
sPos = source_space.centre;
n_sources = size(sPos,1);

region_mapping = load(...
    strcat(anatomy_folder, "fs_cortex_20k_region_mapping.mat")); 

leadield = load(...
    strcat(anatomy_folder, "leadfield_75_20k.mat"));
dummy_orientation = ones(n_sources, 3);

[G, oris, spos] = utl_reshape_leadfield(true, leadield.fwd, ...
        sPos, dummy_orientation, n_electrodes, n_sources);

chanlocs = load(...
    strcat(anatomy_folder, "electrode_75.mat")).eloc75;
    
LFsereega = struct( ...
        'leadfield', G, ...
        'orientation', oris, ...
        'pos', spos, ...
        'chanlocs', chanlocs);
%%
simu_name       = '_dl_mes_deepSIF_'; %fsaverage_';
n_examples      = 100; 


root_folder     = '/home/reynaudsarah/Documents/Data';
do_save         = true; 

constrained_orientation = true; % orientation of sources.
volume                  = false ; % set to true for a volume source space, false for surface source space
spher                  = false; 

montage_kind       = 'deepsif_75' ; %'easycap-M10'; % electrode montage
src_sampling       = 'deepsif_994';  %spacing between sources


if volume 
    suf = strcat( 'vol_', num2str(src_sampling), '.0');
    constrained_orientation = false; %if volume source space the sources are necessarily unconstrained
elseif spher
    suf = strcat( 'sphere_', num2str(src_sampling), '.0' );
    constrained_orientation = false;
else
    suf = num2str(src_sampling) ;
end
%%
% folder path for constrained sources or unconstrained sources
% if constrained : p dipoles, if unconstrained : 3p dipoles (3 dipoles at
% each source position). 
if constrained_orientation 
    folder_path = strcat( root_folder, '/simulation/constrained/', montage_kind, '/' ,suf, '/model' ); 
    saving_folder = strcat( root_folder, '/simulation/constrained/', montage_kind, '/' ,suf, '/simu' ); 
else
    folder_path = strcat( root_folder, '/simulation/unconstrained/', montage_kind, '/' ,suf, '/model' ); 
    saving_folder = strcat( root_folder, '/simulation/unconstrained/', montage_kind, '/' ,suf, '/simu'); 
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
    mkdir( strcat( saving_folder, '/md') ); 
    mkdir( strcat( saving_folder, '/timeline') ); 
end
%%
fs          = 500; % samplling frequency (Hz) 
duree       = 1000; % time serie duration (ms)
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
%%
margin = 2; 
n_patch_min = 1; n_patch_max = 3;%3; 
order_min   = 1; order_max = 3;%3; 

%% TEMPORAL PATTERN PARAMETERS 

amplitude = 1; %1e-9; %nAm
center    = duree/2; %ms 
width     = 50; %ms
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
verts = struct("lh", unique(source_space.tril)', "rh", unique(source_space.trir)'); 
tris = struct("lh", source_space.tril, "rh", source_space.trir); 

nbs = region_mapping.nbs; %utl_compute_neighbors(tris, verts);
neighbors = {};
for s = 1:numel(nbs)
    neighbors.( strcat('src_', num2str(s)) ) = int32( cell2mat(nbs(s)) );
end

%%
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
    % - randomlky choose number of patch
    % - choose amplitude-width... base value
    % for each patch 
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
        seed = -1; 
        while ~any(available_sources==seed) 
            seed = randi([1,max(available_sources, [], 'all')]);
        end
        %seed    = randsample(available_sources, 1); 
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
        
        act_src_file    = strcat( saving_folder, '/sources/Jact/', act_src_file) ;
        noise_src_file  = strcat( saving_folder, '/sources/Jnoise/', noise_src_file );
        
        save(act_src_file, 'Jact');
        save(noise_src_file, 'Jnoise');
        
        eeg_infdb_file = strcat( saving_folder, '/eeg/infdb/', eeg_infdb_file );
        save(eeg_infdb_file, 'eeg_data'); 

        js_md_dict = jsonencode(md_dict);
        md_json_file = strcat(saving_folder, '/md/', md_json_file); 
        js_file = fopen(md_json_file, 'w'); 
        fprintf(js_file, '%s', js_md_dict);
        fclose(js_file);


        match_dict.( strcat('id_',num2str(id)) ) = struct(...
            'act_src_file_name', act_src_file, ...
            'noise_src_file_name', noise_src_file, ...
            'eeg_file_name', eeg_infdb_file, ...
            'md_json_file_name', md_json_file);
        
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
    general_config_file = strcat(root_folder, '/simulation/', simu_name, src_sampling, '_config.json');
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

