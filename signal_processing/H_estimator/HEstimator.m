%% H Estimator by F Olivieri
function [h, H, COH] = HEstimator(InputSignal, OutputSignal, N_fft, N_overlap, EstimatorTypeStr, AlignmentBoolean)
% This function implements an H estimator
% InputSignal and Outpusignal are wav files of the same lenght
% OutputSignal can be a matrix of dimension
% NumbOfChannels*Duration

if nargin < 6, AlignmentBoolean = 1; end %perform the estimation with the alignment of the signals
if nargin < 5, EstimatorTypeStr = 'H1'; end  %The H1 estimator is the default one
if nargin < 4, N_overlap = round(N_fft/2); end % Default overlap is 50% of the length of the window

%% Check OutputSignal dimensions
[NumbOfChannels, DurationTime] = size(OutputSignal);

% The assumption is made that the number of channel is way smaller
% than the duration of the signals. If that is not the case,
% probably we have to transpose the matrix (as, for example,
% the multichannel matlab tlbx returns a matrix that is
% LENGTHTIME * NCHANNELS
if NumbOfChannels > DurationTime
    % then transpose...
    OutputSignal = transpose(OutputSignal);
    [NumbOfChannels, ~] = size(OutputSignal);
end

% Creating parameters for the H estimation
win = hanning(N_fft); % Window creation
Pxx = xspectrum(InputSignal, InputSignal, win, N_overlap);

lengthfreqvect = N_fft/2 + 1;
% Variable initialisation
h = zeros(NumbOfChannels, N_fft);
H = zeros(NumbOfChannels, lengthfreqvect);
COH = zeros(NumbOfChannels, lengthfreqvect);
% Actual estimation loop
for ch_idx = 1:NumbOfChannels
    
    curr_Out_signal = squeeze(OutputSignal(ch_idx, :));
    
    if AlignmentBoolean
        % Input/Output signal alignment
        [OutputSignalProcessed, delay] = AlignTwoSequences(InputSignal, curr_Out_signal);
    else% -> without compensation
        OutputSignalProcessed = curr_Out_signal;
    end
    
    Pyx = xspectrum(OutputSignalProcessed, InputSignal, win, N_overlap);
    Pyy = xspectrum(OutputSignalProcessed, OutputSignalProcessed, win, N_overlap);
    
    if strcmpi(EstimatorTypeStr, 'H1') %EstimatorType == 1
        temp_H = H1(Pyx, Pxx);
    elseif strcmpi(EstimatorTypeStr, 'H2')
        temp_H = Pyy./Pyx;       % H2 Estimator
    elseif strcmpi(EstimatorTypeStr, 'H3') % H3
        temp_H = Pyy - Pxx + sqrt((Pxx - Pyy).^2 + 4*abs(Pxy).^2)./(2*Pyx); %H3
    else
        disp('Not a valid choice of the estimator');
    end
    
    COH_temp  = coherencefunction(InputSignal, OutputSignalProcessed, win, N_overlap);
    
    temp_h = ifft(temp_H, N_fft, 'symmetric');
    
    if AlignmentBoolean
        temp_h = circshift(temp_h, delay); %circular shift to preserve phase information
    end
    
    temp_H = fft(temp_h, N_fft); % Compute the H again after the compensation
    % Returns the H and the COH till the fs/2 value
    temp_H = temp_H(1:lengthfreqvect);
    
    % Storing values
    h(ch_idx, :) = temp_h;
    H(ch_idx, :) = temp_H;
    COH(ch_idx, :) = COH_temp;
end% for ch_idx
end%HEstimator