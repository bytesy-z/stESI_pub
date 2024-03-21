% 6 janvier 2022
% Script in order to check the importation of data from mne
% Specially electrode and source positions
clear; close all;
%% PARAMETERS 
root_name = '/home/reynaudsarah/Documents/Data/';

source_constrained = true; % orientation of sources.
volume             = false ; % set to true for a volume source space, false for surface source space
sphere             = false; 

spacing            = 'ico3';  %spacing between sources. Can be a string or an int (int if volume=true)
elec_montage       = 'standard_1020'; % electrod montage
if volume 
    suf = strcat( 'vol_', num2str(spacing), '.0');
    source_constrained = false; %if volume source space the sources are necessarily unconstrained
elseif sphere
    suf = strcat( 'sphere_', num2str(spacing), '.0' );
    source_constrained = false;
else
    suf = num2str(spacing) ;
end

    
if source_constrained 
    folder_path = strcat( root_name, '/simulation/constrained/', elec_montage, '/' ,suf, '/model' ); 
else
    folder_path = strcat( root_name, '/simulation/unconstrained/', elec_montage, '/' ,suf, '/model' ); 
end

% loading information from mne-python for simulation
CH     = load(strcat(folder_path, '/ch_', suf, '.mat'));
CHLOCS = load(strcat(folder_path, '/chlocs_', suf, '.mat'));
SRC    = load(strcat(folder_path, '/sources_', suf, '.mat'));
LF     = load(strcat(folder_path, '/LF_', suf, '.mat'));
%% 
% Some important parameters
nb_sources  = SRC.nb_sources; 
nb_channels = CH.nb_channels;

% Reshape chanlocs argument (for leadfield definition, useless here)
chanlocs = utl_unpack_chanlocs(nb_channels, CHLOCS); 
%% Chack sources and electrods positions
chPosX = CHLOCS.X; 
chPosY = CHLOCS.Y; 
chPosZ = CHLOCS.Z; 

sPos = SRC.positions; 
sPosX = sPos(:, 1); 
sPosY = sPos(:, 2); 
sPosZ = sPos(:, 3); 

figure()
plot3( chPosX, chPosY, chPosZ, '.', 'color', 'k' ); 
hold on 
plot3( sPosX, sPosY, sPosZ, '+', 'color', 'r');
hold off
legend('Electrode positions', 'Source positions');
xlabel('x [mm]'); ylabel('y [mm]'); zlabel('z [mm]')
title('Electrodes and sources positions');

%% test kmean clustering
% cluster_size = 60; 
% k = ceil(nb_sources/cluster_size); 
% 
% [idx, C] = kmeans(sPos, k); 
% figure() 
% plot3( sPos(idx==1, 1), sPos(idx==1, 2), sPos(idx==1, 3), '+', 'color', 'r' )
% hold on 
% plot3( sPosX(idx>1), sPosY(idx>1), sPosZ(idx>1), '.', 'color', 'k');
% hold off
% title('Test clustering')
%% Test du truc d'Adrien
J = zeros(nb_sources, 1); 
idx = randi(nb_sources, 1); 
J(idx) = 1; 
Ds = zeros(size(J));

% Les deux formulations sont les mêmes. 
%sPos = sPos/1000;
for i = 1:nb_sources
    Ds(i) = sqrt( sum( (sPos(idx,:) - sPos(i,:)).^2 ) ); 
end

D = sqrt( sum( (repmat(sPos(idx,:), nb_sources, 1) - sPos).^2, 2) );

% Normalize between 0 and 1
Ds = Ds./max(Ds); 
% 
% 
% figure()
% plot(Ds, '+', 'MarkerSize' , 5); hold on 
% xline(idx); hold off
% ylim([0, max(Ds)]); 
% title('Distances')
% 
Ds_sort = sort(Ds); 
figure(); 
plot(Ds_sort); 
title('Distance, sorted')
% 
% 
sig = 0.6/6; mu = 0; 
% ampl =  exp( -0.5* ( (Ds_sort.^2)/sig^2 )  )*(1/(sqrt(2*pi)*sig));
% figure()
% plot(Ds_sort, ampl); 
% title('test')
% 
% t = linspace(0, 2, 1000); 
% gt = (1/sqrt(2*pi)) * exp( -0.5* (t.^2)/sig^2  ); 
% figure()
% plot(t, gt); 
% title('Gaussian?')

ampl =  exp( -0.5* ( (Ds.^2)/sig^2 )  )*(1/(sqrt(2*pi)*sig));
thresh = exp( -0.5* ( 9 ) )*(1/(sqrt(2*pi)*sig));
nb_act_src = sum(ampl>thresh) ;
idx_tot = 1:nb_sources; 
idx_act_src = idx_tot(ampl>thresh); 
%%
figure()
plot3( sPosX, sPosY, sPosZ, '.', 'color', 'r');
hold on 
plot3( sPosX(idx_act_src), sPosY(idx_act_src), sPosZ(idx_act_src), '*', 'color', 'k' ); 
hold off
legend('Source positions', 'Active sources');
xlabel('x [mm]'); ylabel('y [mm]'); zlabel('z [mm]')
title('Test extended source from distance from a seed source');

%%
% D_tot = zeros(nb_sources, nb_sources);
% for i = 1:nb_sources 
%     Di = sqrt( sum( (repmat(sPos(i,:), nb_sources, 1) - sPos).^2, 2) );
%     D_tot(i,:) = sort(Di);
% end
% mmin = mean(D_tot(:,2))
% mmax = mean(D_tot(:,end))

%%
lambda = eig(LF.G*(LF.G'));
lambda = flip(lambda);
S = svd(LF.G); 
figure()
plot(S, 'LineWidth', 3)
xlim([1, 61])
xlabel('Singular value n°')
ylabel('Value')
title('Singular values of leadfield matrix')
%%
cmap = winter; 
figure()
imshow(LF.G, [], 'Colormap', cmap)
xlabel('Sources')
ylabel('Electrodes')
title('Leadfield matrix')



