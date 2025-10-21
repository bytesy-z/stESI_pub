
% The purpose of this global variable is to help scripts loading stuff from
% the default locations even when the user has cd'd elsewhere. [from the
% setup_paths de simBCI
global SEREEGA_BASE_DIR;

SEREEGA_BASE_DIR = '/home/zik/UniStuff/FYP/SEREEGA';

% Helper function to add path if folder exists
function addpath_if_exists(folder)
	if exist(folder, 'dir')
		addpath(folder);
	end
end

% docs
addpath_if_exists(sprintf('%s/docs/', SEREEGA_BASE_DIR));

% generate
addpath_if_exists(sprintf('%s/generate', SEREEGA_BASE_DIR));
%leadfield
addpath_if_exists(sprintf('%s/leadfield', SEREEGA_BASE_DIR));
addpath_if_exists(sprintf('%s/leadfield/fieldtrip', SEREEGA_BASE_DIR));
addpath_if_exists(sprintf('%s/leadfield/nyhead', SEREEGA_BASE_DIR));
addpath_if_exists(sprintf('%s/leadfield/pha', SEREEGA_BASE_DIR));

%plot
addpath_if_exists(sprintf('%s/plot', SEREEGA_BASE_DIR));

%pop
addpath_if_exists(sprintf('%s/pop', SEREEGA_BASE_DIR));

%signal
addpath_if_exists(sprintf('%s/signal', SEREEGA_BASE_DIR));
addpath_if_exists(sprintf('%s/signal/arm', SEREEGA_BASE_DIR));
addpath_if_exists(sprintf('%s/signal/data', SEREEGA_BASE_DIR));
addpath_if_exists(sprintf('%s/signal/erp', SEREEGA_BASE_DIR));
addpath_if_exists(sprintf('%s/signal/ersp', SEREEGA_BASE_DIR));
addpath_if_exists(sprintf('%s/signal/noise', SEREEGA_BASE_DIR));

%utils
addpath_if_exists(sprintf('%s/utils', SEREEGA_BASE_DIR));

%inverse algor
addpath_if_exists(sprintf('%s/inverse_algorithms_simBCI', SEREEGA_BASE_DIR));

% Simulation
addpath_if_exists(sprintf('%s/simu', SEREEGA_BASE_DIR));
addpath_if_exists(sprintf('%s/simu/utl', SEREEGA_BASE_DIR));
addpath_if_exists(sprintf('%s/simu/sim', SEREEGA_BASE_DIR));
addpath_if_exists(sprintf('%s/simu/get_src', SEREEGA_BASE_DIR));

%mains
addpath_if_exists(sprintf('%s/mains', SEREEGA_BASE_DIR));


%add the path of the eeglab folder (on va voir si ça fonctionne)
addpath_if_exists('/home/zik/UniStuff/FYP/eeglab2025.0.0');

