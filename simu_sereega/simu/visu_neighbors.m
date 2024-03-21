% 28 mars - problème neighbors computation
%
%% 
clear; close all
%% PARAMETERS
simu_name       = '_test_mes_fsaverage_';
n_examples      = 100; 


root_folder     = '/home/reynaudsarah/Documents/Data';
do_save         = true; 

constrained_orientation = true; % orientation of sources.
volume                  = false ; % set to true for a volume source space, false for surface source space
spher                  = false; 

montage_kind       = 'sample'; %'standard_1020' ; %'easycap-M10'; % electrode montage
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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Load data 
% get the data using the unpack_fwdModel function
[CH, chanlocs, SRC, LFmne, LFsereega] = utl_unpack_fwdModel( folder_path, suf, constrained_orientation );

% Visualisation
% plot_headmodel(LFsereega); 
% title('Sources and electrodes positions')

n_electrodes = CH.nb_channels;
n_sources  = SRC.nb_sources; 
sPos       = SRC.positions*1e-3; 

%% Compute neighbors from the mesh triangle data
tlh     = load(strcat(folder_path, '/tris_lh_', suf, '.mat'));
trh     = load(strcat(folder_path, '/tris_rh_', suf, '.mat'));
tris = struct("lh", tlh.tris_lh + 1, "rh", trh.tris_rh + 1); 

verts = load(strcat(folder_path, '/verts_', suf, '.mat'));
verts = struct("lh", verts.verts_lh + 1, "rh", verts.verts_rh + 1); 

% Concaténation des 2 matrices de triangles
% tris        = zeros(2, length(tris_lh), 3, 'int64'); 
% tris(1,:,:) = tris_lh; 
% tris(2,:,:) = tris_rh; 
% 

% for hem=1:2
%     old_indices = sort( unique( tris(hem,:,:) ) );
%     new_indices = 1:1:length(old_indices) ;
% 
%     for i = 1:length(old_indices)
%         tris(hem, tris(hem,:,:)==old_indices(i) ) = int64(new_indices(i));
%     end
% end
% tris(2,:,:) = tris(2,:,:) + n_sources/2;
neighbors_v1 = utl_compute_neighbors(tris, verts);
%%
neighbors_2 = struct();
for i =1:( length(verts.lh) + length(verts.rh) )
    neighbors_v2.( strcat('src_', num2str(i)) ) = [];
end
idx_tris_old = int64( sort( unique(tris.lh) ) ); 
idx_vert_old = int64( sort( unique(verts.lh) ) );
[missing_verts, missing_verts_idx] = setdiff(idx_tris_old, idx_vert_old); 
    
idx_tris_new = int64( 1:1:length(idx_tris_old) ); 
idx_vert_new = int64( 1:1:length(idx_vert_old) ); 
%idx_vert_new(missing_verts_idx) = []; % remove missing vertices

vertices_lin = zeros(max(idx_vert_old), 1); 
vertices_lin(idx_vert_old) = idx_vert_new;

i = 1;
for v = idx_vert_old
    % find the neighbors of v 
    triangles_of_v = squeeze( tris.lh==v ); % triangles in which v is
    triangles_of_v = ...
              squeeze( tris.lh( sum(triangles_of_v, 2)>0,: ) );

    neighbors_of_v = unique(triangles_of_v)';
    neighbors_of_v( neighbors_of_v==v ) = []; %remove v
    neighbors_of_v = setdiff(int64(neighbors_of_v), int64(missing_verts)); %remove missing vertices

    neighbors_v2.( strcat('src_', num2str(i)) ) = ...
              [neighbors_v2.( strcat('src_', num2str(i)) ),  vertices_lin(neighbors_of_v)];
    i = i+1;
end
idx_tris_old = int64( sort( unique(tris.rh) ) ); 
idx_vert_old = int64( sort( unique(verts.rh) ) ); 

[missing_verts, missing_verts_idx] = setdiff(idx_tris_old, idx_vert_old); 

idx_tris_new = 1:1:length(idx_tris_old); 
idx_vert_new = 1:1:length(idx_vert_old); 
%idx_vert_new(missing_verts_idx) = []; % remove missing vertices

vertices_lin = zeros(max(idx_vert_old), 1); 
vertices_lin(idx_vert_old) = idx_vert_new;

for v = idx_vert_old
    % find the neighbors of v 
    triangles_of_v = squeeze( tris.rh==v ); % triangles in which v is
    triangles_of_v = ...
              squeeze( tris.rh( sum(triangles_of_v, 2)>0,: ) );

    neighbors_of_v = unique(triangles_of_v)';
    neighbors_of_v( neighbors_of_v==v ) = []; %remove v
    neighbors_of_v = setdiff(int64(neighbors_of_v), int64(missing_verts)); %remove missing vertices

    neighbors_v2.( strcat('src_', num2str(i)) ) = ...
              [neighbors_v2.( strcat('src_', num2str(i)) ), vertices_lin(neighbors_of_v)+length(verts.lh)];
    i = i+1;
end
%%
s = randi(n_sources);

for k = 1:10
    patch = utl_get_patch(k, s, neighbors_v1);
    figure()
    scatter3(sPos(:,1), sPos(:,2), sPos(:,3) ); 
    hold on; 
    scatter3( sPos(patch,1), sPos(patch, 2), sPos(patch,3), 'filled');
    hold on; 
    scatter3(sPos(s, 1), sPos(s, 2), sPos(s,3) , 'filled');
    hold off
    title(strcat("order", num2str(k) ) )
    %title(strcat("source", num2str(s) ) )
%     if rem(s, 10) == 0
%         pause
%     end
end
