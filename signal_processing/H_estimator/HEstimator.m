function [h, H, COH] = HEstimator(InputSignal, OutputSignal, N_fft, N_overlap, ...
    EstimatorTypeStr, AlignmentBoolean)
% HESTIMATOR Toolbox by Ferdinando Olivieri (f.olivieri@ieee.org)

%% SUMMARY
% This function implements an H-estimator [1] for the estimation of transfer functions
% (and impulse responses) of a Linear Time-Invariant System. The excitation
% signal must be a white noise.

% You can run an example code by calling
% >> [h, H, COH] = HEstimator();
% at the Matlab prompt

% [1] Shin, Kihong and Hammond, Joseph (2008)
% Fundamentals of signal processing for sound and vibration engineers,
% Chichester, UK, Wiley, 416pp.

%% INPUTS - OUTPUTS
% INPUTS:
% - INPUTSIGNAL is the N_SAMPLES X 1 vector of the white noise excitation signal
% - OUTPUTSIGNAL is the N_SAMPLES X N_CHANNELS matrix of the recorded signals.
% - N_FFT is the duration (in samples) of the estimated impulse responses
% - N_OVERLAP is the length of the overlap window between frames expressed in percentage of N_FFT (default: 50%)
% - ESTIMATORTYPESTR is a string indicating which H-Estimator to use ('H1' / 'H2' / 'H3' -- default 'H1')
% - ALIGNMENTBOOLEAN activates/deactivate estimation with aligned sequencies to improve signal/noise ratio in the estimated IRs (default: 1).
% OUTPUTS:
% - h: the N_FFT X N_CHANNELS matrix of the estimated impulse responses
% - H: the (N_FFT/2 + 1 X N_CHANNELS) matrix of the estimated transfer functions
% - COH: the (N_FFT/2 + 1 X N_CHANNELS) matrix of the coherence

%% LICENSE - TO-DO:ADD DISCLAIMER
% This program is free software: you can redistribute it and/or modify
% it under the terms of the GNU General Public License as published by
% the Free Software Foundation, either version 3 of the License, or
% any later version.
%
% This program is distributed in the hope that it will be useful,
% but WITHOUT ANY WARRANTY; without even the implied warranty of
% MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
% GNU General Public License for more details.
%
% If you apply any modification to the code, please do share the
% modified code. I would appreciate if you sent an email to
% f.olivieri@ieee.org if you used this code.
%
% You should have received a copy of the GNU General Public License
% along with this program.  If not, see <http://www.gnu.org/licenses/>.

%% EXAMPLE CODE
if nargin < 1
    disp('Running the example code...');
    % Create some fake data
    n_samples = 500; % length of excitation and recorded signals
    whitenoise = randn(n_samples, 1); % excitation signal
    
    outputtest = zeros(n_samples, 2); % fake 2-ch recording
    outputtest(:, 1) = circshift(whitenoise, 100); % basically a delayed version of the input
    outputtest(:, 2) = circshift(whitenoise, 200);
    
    % Estimation
    N_fft = 300; % duration of the estimated IRs
    [h_test, H_test, COH_test] = HEstimator(whitenoise, outputtest, N_fft);
    
    % Plot results
    figure;
    plot(h_test);
    title('IRs')
    disp('End of the example code...');
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% START PROCESSING
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if nargin < 6, AlignmentBoolean = 1; end %perform the estimation with the alignment of the signals
if nargin < 5, EstimatorTypeStr = 'H1'; end  %The H1 estimator is the default one
if nargin < 4, N_overlap = round(N_fft/2); end % Default overlap is 50% of the length of the window

%% Check INPUTS

% Check dimensions of the input matrices
[N_SAMPLES_out, N_CHANNELS] = size(OutputSignal);
[N_SAMPLES_in] = length(InputSignal);

if N_SAMPLES_out ~= N_SAMPLES_in
    disp('ERROR: The input signal and output signals must be of the same length. Now quitting.');
    return;
end

% Check duration of estimated IRs does not exceed duration of recorded signals
if N_fft > N_SAMPLES_out
    disp('ERROR: N_fft cannot be larger than N_SAMPLES. Now quitting.');
    return;
end

%% Variable initialization
lengthfreqvect = N_fft/2 + 1; % the number of frequency bins
h = zeros(N_fft, N_CHANNELS); % the matrix of estimated impulse responses
H = zeros(lengthfreqvect, N_CHANNELS); % the matrix of estimated Transfer functions
COH = zeros(lengthfreqvect, N_CHANNELS); % The matrix of the coherence

%% Processing

win = hanning(N_fft); % Window creation

% Auto-spectrum of the input signal
Pxx = xspectrum(InputSignal, InputSignal, win, N_overlap);

% Estimation loop
for ch_idx = 1:N_CHANNELS
    
    curr_Out_signal = OutputSignal(:, ch_idx);
    
    if AlignmentBoolean
        % Input/Output signal alignment
        [OutputSignalProcessed, delay] = AlignTwoSequences(InputSignal, curr_Out_signal);
    else% -> without compensation
        OutputSignalProcessed = curr_Out_signal;
    end
    
    % Cross-spectrum of the output/input
    Pyx = xspectrum(OutputSignalProcessed, InputSignal, win, N_overlap);
    % Cross-spectrum of the output/output
    Pyy = xspectrum(OutputSignalProcessed, OutputSignalProcessed, win, N_overlap);
    
    % Estimation of the transfer function
    if strcmpi(EstimatorTypeStr, 'H1') %H1 Estimator
        temp_H = Pyx./Pxx;       % H1 estimator; % H1
    elseif strcmpi(EstimatorTypeStr, 'H2') % H2 Estimator
        temp_H = Pyy./Pyx;       % H2 Estimator
    elseif strcmpi(EstimatorTypeStr, 'H3') % H3 Estimator
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
    h(:, ch_idx) = temp_h;
    H(:, ch_idx) = temp_H;
    COH(:, ch_idx) = COH_temp;
end% for ch_idx
end%HEstimator

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SUB-ROUTINES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function delay_sequence

end

function [DelayedOutputSignal, delay_in_samples] = ...
    AlignTwoSequences(InputSignal, OutputSignal)
% OutputSignal is the sequence that is shifted
% [1] CLASSIFICATION AND EVALUATION OF DISCRETE SUBSAMPLE TIME DELAY ESTIMATION ALGORITHMS
InputSignal = InputSignal(:);
OutputSignal = OutputSignal(:);
delay_in_samples = EstimateDelayInSamplesBtwTwoSequences(InputSignal, OutputSignal);
% Shift in the time domain
DelayedOutputSignal = circshift(OutputSignal, -delay_in_samples);
end

function [delay_in_samples,  CrossCorrelation] = EstimateDelayInSamplesBtwTwoSequences(InputSignal, OutputSignal)
% Estimates the delay in samples between OutputSignal and the reference sequence InputSignal
lengthInput = length(InputSignal);
lengthOutput = length(OutputSignal);
CrossCorrelation = xcorr(OutputSignal, InputSignal); %compute cross-correlation between vectors InputSignal and OutputSignal

[~, d] = max(CrossCorrelation); %find value and index of maximum value of cross-correlation amplitude
delay_in_samples = d - max(lengthInput, lengthOutput) + 1; %shift index d, as length(X1)=2*N-1; where N is the length of the signals
delay_in_samples = delay_in_samples - 1;

end%EstimateDelayInSamplesBtwTwoSequences