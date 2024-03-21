%
% generate 'nSamples' new samples having the same spectral characteristics as 'source'.
%
% 'Multiplier' > 1 can be used to squeeze the spectrum towards the low frequencies
% to generate slow 'baseline drifts'.
%
% The function can be used as part of a pipeline to replicate the BCI Competition IV
% artifical dataset (Tangermann & al. 2012, sections 4.2.1 and 4.2.2)
%
% Output f contains the summed eigenvalues per each frequency slice of the interpolated spectrum.
%
function [noiseColored, interpolatedSpectrum, f] = gen_colored_noise(crossSpectrum, samplingFreq, nSamples, multiplier, viz)

	if(nargin<4)
		multiplier = 1;
	end
	if(nargin<5)
		viz = 0;
    end
    
    

	dim = size(crossSpectrum,1);

    CPstd = ones(1, dim);
    CPmean = zeros(1,dim); 
    
	nSamplesModded = false;
	if rem(nSamples,2)
		nSamples=nSamples + 1;
		nSamplesModded = true;
	end

	% Due to FFT symmetry ...
	uniquePts = nSamples/2 + 1;

	interpolatedSpectrum = [];
	f=[];



	% Generates Gaussian white noise, maps it to time-frequency space, modulating it to
	% have a specified 1/f type slope and spectral covariance structure, identical for every frequency.
	% Generate Gaussian white noise and convert to the freq space
	noiseFFTColored = fft( randn(nSamples, dim) );
	% in this case the cross spectrum is same for each freq, and we add an 1/freq power law here.
	% n.b. In total, this probably resembles generating pink noise and enforcing a spatial cov structure for it.
	[E,D] = eig(crossSpectrum);
	%figure(1); imagesc(E); drawnow;
	targetFrequencies = linspace(0, multiplier*samplingFreq/2, uniquePts);
	% Filter the white noise to give it the desired covariance structure
	noiseFFTColored = noiseFFTColored*real(E*sqrt(D))';
	noiseFFTColored(1,:) = 0;	% zero DC

	weighting = 1./sqrt(targetFrequencies(2:end).^1.7);
	noiseFFTColored(2:uniquePts,:) = noiseFFTColored(2:uniquePts,:) .* repmat(weighting', [1 size(noiseFFTColored,2)]);

	noiseFFTColored(uniquePts+1:nSamples,:) = real(noiseFFTColored(nSamples/2:-1:2,:)) -1i*imag(noiseFFTColored(nSamples/2:-1:2,:));

	noiseColored = real(ifft(noiseFFTColored));


	if(0)
		% force sources spatial covariance structure. Note that this may adversely affect the earlier spectral structuring we did.
		[E1,D1] = eig(cov(noiseColored));
		noiseColoredWhite = noiseColored*real(E1*sqrt(pinv(D1)));
		[E2,D2] = eig(spectralModel.cov);
		noiseColored = (real(E2*sqrt(D2))*noiseColoredWhite')';
	end

	% set the channel means and variances from the original signal
	%
	% First scale to 0 mean, unit variance 	and then specify these params from the sample data
	noiseColored = normalize(noiseColored);
	noiseColored = noiseColored .* repmat( CPstd, [size(noiseColored,1) 1]);
	noiseColored = noiseColored + repmat(  CPmean, [size(noiseColored,1) 1]);

	if(nSamplesModded)
		noiseColored = noiseColored(1:nSamples-1,:);
    end
    
end
