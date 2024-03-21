% function neighbors = utl_compute_neighbors(tris)
% 2022.07.15
% function to compute the direct neighbors of each source in a source
% space, from triangle data of the mesh.
% 
% input: 
% - tris:  matrix of triangles tris(1,:,:) -> triangles of left hemisphere, 
%   tris(2,:,:) -> triangles of right hemisphere.
% ouput: 
% - neighbors: structure of neighbors. Each field = the nieghbors of a
%   given source (field names : src_i for i = 1:n_sources.
% 
% From: https://github.com/LukeTheHecker/esinet/blob/main/esinet/util/util.py
% Hecker, L., Rupprecht, R., Tebartz Van Elst, L., & Kornmeier, J. (2021). 
% ConvDip: A Convolutional Neural Network for Better EEG Source Imaging. 
% Frontiers in Neuroscience, 15, 569918. https://doi.org/10.3389/fnins.2021.569918


% function neighbors = utl_compute_neighbors(tris)
%     n_sources = numel( unique(tris) );
%     neighbors = struct(); 
%     for i =1:n_sources
%         neighbors.( strcat('src_', num2str(i)) ) = [];
%     end
%     for hem = 1:2
%           % pour chaque source
%           for i = 1:n_sources
%               % cherche les triangles dont la source fait partie
%               triangles_of_i = squeeze( tris(hem,:,:)==i );
%               % retourne une matrice de dimension de dimension n_sources/2 by 3
%               % avec des true/false, donc pour connaître les indices des
%               % triangles dont i fait partie, il faut savoir dans quelles
%               % colonnes de triangles of i il y a au moins un true
%               triangles_of_i = ...
%                   squeeze( tris( hem, sum(triangles_of_i, 2)>0,: ) );
% 
%               % une fois les triangles identifié, on flatten la matrice et
%               % récupère les indices uniques
%               neighbors_of_i = unique(triangles_of_i)'; 
%               % puis on enlève i... qui n'est pas sont propre voisin
%               neighbors_of_i( neighbors_of_i==i ) = [];
% 
%               % on concatène dans la structure pour avoir pour chaque source
%               % les voisins associés
%               neighbors.( strcat('src_', num2str(i)) ) = ...
%                   [neighbors.( strcat('src_', num2str(i)) ), neighbors_of_i];
%           end
%     end
% end%function

% v2
function neighbors = utl_compute_neighbors(tris, verts)
    neighbors = struct();
    for i =1:( length(verts.lh) + length(verts.rh) )
        neighbors.( strcat('src_', num2str(i)) ) = [];
    end
    
    % left hemisphere
    idx_tris_old = int64( sort( unique(tris.lh) ) ); 
    idx_vert_old = int64( sort( unique(verts.lh) ) ); 
    
    [missing_verts, missing_verts_idx] = setdiff(idx_tris_old, idx_vert_old); 
    
    idx_tris_new = 1:1:length(idx_tris_old); 
    idx_vert_new = 1:1:length(idx_vert_old); 
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
    
        neighbors.( strcat('src_', num2str(i)) ) = ...
                  [neighbors.( strcat('src_', num2str(i)) ), vertices_lin(neighbors_of_v)];
        i = i+1;
    end
    
    % right hemisphere %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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
    
        neighbors.( strcat('src_', num2str(i)) ) = ...
                  [neighbors.( strcat('src_', num2str(i)) ), vertices_lin(neighbors_of_v)+length(verts.lh)];
        i = i+1;
    end

end