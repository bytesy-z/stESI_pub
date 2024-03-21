
% The purpose of this global variable is to help scripts loading stuff from
% the default locations even when the user has cd'd elsewhere. [from the
% setup_paths de simBCI
global SEREEGA_BASE_DIR;

SEREEGA_BASE_DIR = sprintf('%s/./', pwd());

% docs
addpath(sprintf('%s/docs/', SEREEGA_BASE_DIR));

% generate
addpath(sprintf('%s/generate', SEREEGA_BASE_DIR));
%leadfield
addpath(sprintf('%s/leadfield', SEREEGA_BASE_DIR));
addpath(sprintf('%s/leadfield/fieldtrip', SEREEGA_BASE_DIR));
addpath(sprintf('%s/leadfield/nyhead', SEREEGA_BASE_DIR));
addpath(sprintf('%s/leadfield/pha', SEREEGA_BASE_DIR));

%plot
addpath(sprintf('%s/plot', SEREEGA_BASE_DIR));

%pop
addpath(sprintf('%s/pop', SEREEGA_BASE_DIR));

%signal
addpath(sprintf('%s/signal', SEREEGA_BASE_DIR));
addpath(sprintf('%s/signal/arm', SEREEGA_BASE_DIR));
addpath(sprintf('%s/signal/data', SEREEGA_BASE_DIR));
addpath(sprintf('%s/signal/erp', SEREEGA_BASE_DIR));
addpath(sprintf('%s/signal/ersp', SEREEGA_BASE_DIR));
addpath(sprintf('%s/signal/noise', SEREEGA_BASE_DIR));

%utils
addpath(sprintf('%s/utils', SEREEGA_BASE_DIR));

%inverse algor
addpath(sprintf('%s/inverse_algorithms_simBCI', SEREEGA_BASE_DIR));

% Simulation
addpath(sprintf('%s/simu', SEREEGA_BASE_DIR));
addpath(sprintf('%s/simu/utl', SEREEGA_BASE_DIR));
addpath(sprintf('%s/simu/sim', SEREEGA_BASE_DIR));
addpath(sprintf('%s/simu/get_src', SEREEGA_BASE_DIR));

%mains
addpath(sprintf('%s/mains', SEREEGA_BASE_DIR));


%add the path of the eeglab folder (on va voir si ça fonctionne)
% @todo trouver une façon de ne pas le mettre à partir de mon home
% directory c'est pas très pratique 
addpath('/home/reynaudsarah/Documents/EEG/eeglab2021.1');

