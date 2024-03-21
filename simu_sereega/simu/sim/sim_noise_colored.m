%function noise = sim_noise_colored(lf, timeline viz)
% Pris et adapté de simBCI

function noise = sim_noise_colored(lf, timeline, viz)

    [nb_channels, ~] = size( lf.leadfield ); 
    nb_samples = timeline.srate*timeline.length/1000;
    coeff = [1, 0.5, 0.3 ]; 
    exponent = 1.7; 
        
    fs = timeline.srate;
	physicalModel = lf;
    
    
    % 1. Estimate fake cross spectrum
    chPos = zeros(nb_channels, 3); 
    for c = 1:nb_channels
        chPos(c,1) = lf.chanlocs(c).X; 
        chPos(c,2) = lf.chanlocs(c).Y; 
        chPos(c,3) = lf.chanlocs(c).Z; 
    end
    
    crossSpectrum = zeros(nb_channels, nb_channels);
    for i=1:nb_channels
        for j=i:nb_channels
            d = norm(chPos(i,:) - chPos(j,:));
			crossSpectrum(i,j) = exp(-(d.^2));
			crossSpectrum(j,i) = crossSpectrum(i,j);
        end
    end
    
    if(viz)
        figure();
		imagesc(crossSpectrum);
		title('Fake cross spectrum');
		xlabel('Electrode'); ylabel('Electrode');
    end
    
    
    %2. Generate the three "colored" components of surface noise : 
    noise =         coeff(1) .* gen_colored_noise(crossSpectrum, fs, nb_samples, 1, 1);    % background noise, bci comp iv 4.2.1
	noise = noise + coeff(2) .* gen_colored_noise(crossSpectrum,fs, nb_samples, 150, 0);  % drift 1, bci comp iv 4.2.2
	noise = noise + coeff(3) .* gen_colored_noise(crossSpectrum, fs, nb_samples, 300, 0);  % drift 2, bci comp iv 4.2.2
    
    noise = noise'; 
    if viz
        figure()
        plot(noise(1,:)); 
        title('Colored noise'); 
    end


end