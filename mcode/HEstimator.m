function [h, H, COH] = HEstimator(InputSignal, OutputSignal, N_fft, N_overlap_perc, ...
    EstimatorTypeStr, AlignmentBoolean, PlotFlag, FigureNameTextStr, fs)
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
% - N_OVERLAP_PERC is the length of the overlap window between frames
%                  expressed in percentage of N_FFT (default: 50). 
%                  It can be equal to 25, 33, 50, 75.
% - ESTIMATORTYPESTR is a string indicating which H-Estimator to use ('H1' / 'H2' / 'H3' -- default 'H1')
% - ALIGNMENTBOOLEAN activate/deactivate estimation with aligned sequencies to improve signal/noise ratio in the estimated IRs (default: 1).
% - PLOTFLAG (either 1 or 0) to activate/deactivate plots for each estimation
% - FigureNameTextStr: a string for the figure name (mandatory if PLOTFLAG = 1.
% - FS sampling frequency (in Hz): mandatory if PLOTFLAG is set to 1.
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

%% Run EXAMPLE CODE
if nargin < 1
    run_example_code_H_estimator();
    % exit the function by assigning NaN to output values to avoid Matlab's
    % complaint if values are not assigned.
    h = NaN; H = NaN; COH = NaN;
    return;
end
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% START
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Check INPUTS
if nargin < 7, PlotFlag = 0; end  % no plots at each iteration
if nargin < 6, AlignmentBoolean = 1; end %perform the estimation with the alignment of the signals
if nargin < 5, EstimatorTypeStr = 'H1'; end  %The H1 estimator is the default one
if nargin < 4, N_overlap_perc = 50; end % Default overlap is 50% of the length of the window


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

% Check existance of variables if PlotFlag == 1
if PlotFlag == 1
    if ~exist('fs', 'var')
        disp('ERROR: if PLOTFLAG = 1 you must specify a sampling frequency in Hertz.'); return;
    end
    
    if ~exist('FigureNameTextStr', 'var')
        disp('ERROR: if PLOTFLAG = 1 you must specify a name for the figure.'); return;
    end
end

% Check N_overlap_percentage is initialized properly
switch N_overlap_perc
    case {25, 33, 50, 75}
        % Everything is OK. do nothing
    otherwise
        disp('ERROR: N_overlap_percentage must be either 25, 33, 50, 75.'); return;
end

%% Variable initialization
lengthfreqvect = N_fft/2 + 1; % the number of frequency bins
h = zeros(N_fft, N_CHANNELS); % the matrix of estimated impulse responses
H = zeros(lengthfreqvect, N_CHANNELS); % the matrix of estimated Transfer functions
COH = zeros(lengthfreqvect, N_CHANNELS); % The matrix of the coherence
N_overlap = round(N_fft*(N_overlap_perc/100)); % convert overlap percentage to samples
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
        temp_h = delay_sequence(temp_h, delay); %circular shift to the right-hand side preserve phase information
    end
    
    temp_H = fft(temp_h, N_fft); % Compute the H again after the compensation
    % Returns the H and the COH till the fs/2 value
    temp_H = temp_H(1:lengthfreqvect);
    
    % Storing values
    h(:, ch_idx) = temp_h;
    H(:, ch_idx) = temp_H;
    COH(:, ch_idx) = COH_temp;
end% for ch_idx

if PlotFlag
    H_Estimator_Plots(h, H, COH, fs, FigureNameTextStr);
end
end%HEstimator

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% EXAMPLE CODE
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [h, H, COH] = run_example_code_H_estimator()
disp('Running the example code...');
    % Create some fake data
    fs = 48000; % sampling frequency
    n_samples = 500; % length of excitation and recorded signals
    whitenoise = randn(n_samples, 1); % excitation signal
    
    outputtest = zeros(n_samples, 2); % fake 2-ch recording
    outputtest(:, 1) = circshift(whitenoise, 100); % basically a delayed version of the input
    outputtest(:, 2) = circshift(whitenoise, 200);
    
    % Parameters for the Estimation
    N_fft = 300; % duration of the estimated IRs
    N_overlap_perc = 50; % percent of overlap
    AlignmentBoolean = 1;
    PlotFlag = 1;
    FigureNameTextStr = 'Example code';

    % Estimation (with plots)
    [h_test, H_test, COH_test] = HEstimator(whitenoise, outputtest, N_fft, ...
        N_overlap_perc, 'H1', AlignmentBoolean, PlotFlag, FigureNameTextStr, fs);
    
     % Estimation (without plots)
    [h_test, H_test, COH_test] = HEstimator(whitenoise, outputtest, N_fft, ...
        N_overlap_perc, 'H1', AlignmentBoolean);
    
    
    
    disp('End of the example code...');
    
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% SUB-ROUTINES
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function DelayedOutputSignal = delay_sequence(OutputSignal, delay_in_samples)
% Basically a wrapper around Matlab's CIRCSHIFT function - since they
% change the syntax every now and then.

% INPUTS:
% - OUTPUTSIGNAL: the vector N_SAMPLES X 1 of the sequence to be shifted
% - DELAY_IN_SAMPLES: the integer delay in samples
% OUTPUTS:
% - DelayedOutputSignal: the vector N_SAMPLES X 1 of the delayed sequence

% % Testing CIRCSHIFT - uncomment the following 3-lines code snippet to test
% CIRCSHIFT's behaviour
% OutputSignal = [1; zeros(10, 1)]; % Input signal
% delay_in_samples = 3;
% OutputSignalDelayed = circshift(OutputSignal, delay_in_samples)

DelayedOutputSignal = circshift(OutputSignal, delay_in_samples);
end%delay_sequence

function [DelayedOutputSignal, delay_in_samples] = ...
    AlignTwoSequences(InputSignal, OutputSignal)
% OutputSignal is the sequence that is shifted
% [1] CLASSIFICATION AND EVALUATION OF DISCRETE SUBSAMPLE TIME DELAY ESTIMATION ALGORITHMS
InputSignal = InputSignal(:);
OutputSignal = OutputSignal(:);
delay_in_samples = EstimateDelayInSamplesBtwTwoSequences(InputSignal, OutputSignal);
% Shift in the time domain (towards the left-hand side)
DelayedOutputSignal = delay_sequence(OutputSignal, -delay_in_samples);
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


function COH = coherencefunction(InputSignal, OutputSignalProcessed, win, N_overlap)
% By Ferdinando Olivieri
Pxx = xspectrum(InputSignal, InputSignal, win, N_overlap);
Pyx = xspectrum(OutputSignalProcessed, InputSignal, win, N_overlap);
Pyy = xspectrum(OutputSignalProcessed, OutputSignalProcessed, win, N_overlap);
COH = (abs(Pyx).^2)./(Pxx.*Pyy);
end%coherencefunction




function Pxy = xspectrum(x, y, w, N_overlap)
% Cross spectral density of vectors x and y
%
% Sxy is the csd of signal x and signal y sampled at rate Fs based on
% FFTs of segments that are first multiplied by the window
% given in w (the size of this also defines the FFT size).
%
% [Sxy,f] = csd(x,y,w,Fs)
% INPUT:
%    - x, y: input vectors
%    - w: vector containing the window
%    - Fs: sampling frequency
% OUTPUT:
%    - Sxy: CSD of signal x and signal y
%    - f: set of corresponding frequencies

x = x(:); % make sure x is a column vector
y = y(:); % make sure y is a column vector
w = w(:); % make sure w is a column vector

L = min(length(x), length(y));
N = length(w);

if nargin < 4
    N_overlap = 0;
end

D = N - N_overlap;
%D = N_overlap;

% % determine number of averages
% L = length(x)/N;
% L = floor(L); % Truncation
%
% if L == 0 % if N > length(x)
%     L = 1;
% end
% determine number of averages (from matlab cpsd function)
K = (L - N_overlap)/D;
K = floor(K);

lengthfreqvect = N/2 + 1;
% find FFTs and average
Pxy = zeros(lengthfreqvect, 1);

U = sum(w.^2)/N;

for frame_idx = 0:K-1
    
    % Segmentation of the signals in frames
    lowlim = frame_idx*D + 1; % Plus one sample to adapt to the Matlab convention
    upperlim = frame_idx*D + N;
    
    x_idx = x(lowlim:upperlim);
    y_idx = y(lowlim:upperlim);
    
    % Calculates fft for the current frame
    X_idx = fft(w.*x_idx);
    Y_idx = fft(w.*y_idx);
    
    % Half of the spectrum
    X_idx = X_idx(1:lengthfreqvect);
    Y_idx = Y_idx(1:lengthfreqvect);
    
    % Current power spectrum
    curr_Pxy = (X_idx).*conj(Y_idx);
    curr_Pxy = curr_Pxy/(N*U); %Normalisation
    
    % Calculate xpower
    Pxy = Pxy + curr_Pxy;
end

Pxy = Pxy/K;

end

function H_Estimator_Plots(h, H, COH, fs, FigureNameTextStr)
% Plot the result of the multichannel SYS-ID session
if nargin < 4
else
    figure('Name', FigureNameTextStr);
end

N_fft = length(h);

samplevect = (0:(N_fft - 1));
timevect = samplevect/fs;
freqvect = linspace(0, fs/2, N_fft/2+ 1);

subplot(2,2,1); plot(timevect, h); title('Estimated IR');
xlabel('Samples'); ylabel('Amplitude'); grid on;
xlim([timevect(1), timevect(end)]); %ylim([-8*10^-3, 8*10^-3]);

%Ax1 = gca;
%Ax2 = axes('Position', get(Ax1,'Position'), 'XAxisLocation','top');
% plot(samplevect, h, 'color','k','parent', Ax2);
%xlim([samplevect(1), samplevect(end)]);

subplot(2,2,2); semilogx(freqvect, LogDB(abs(H)));
title('Magnitude FRF'); xlabel('Freq, Hz'); ylabel('dB');
grid on; xlim([50, fs/2]);

subplot(2,2,3); semilogx(freqvect, unwrap(angle((H))));
title('Phase (unwrapped) FRF');
xlabel('Freq, Hz');xlim([50, fs/2]);  grid on;

subplot(2,2,4); semilogx(freqvect, COH); title('Coherence'); xlabel('Freq, Hz');
grid on; xlim([50, fs/2]); ylim([0, 1.1]);
end


% function [Sxy,f] = csd(x, y, w, Fs)
% % Cross spectral density of vectors x and y
% %
% % Sxy is the csd of signal x and signal y sampled at rate Fs based on
% % FFTs of segments that are first multiplied by the window
% % given in w (the size of this also defines the FFT size).
% %
% % [Sxy,f] = csd(x,y,w,Fs)
% % INPUT:
% %    - x, y: input vectors
% %    - w: vector containing the window
% %    - Fs: sampling frequency
% % OUTPUT:
% %    - Sxy: CSD of signal x and signal y
% %    - f: set of corresponding frequencies
%
% x = x(:); % make sure x is a column vector
% y = y(:); % make sure y is a column vector
% w = w(:); % make sure w is a column vector
%
% N = length(w);
% %disp(['block length: ',int2str(N)])
% f = (0:N/2)*Fs/N;
%
% % determine number of averages
% m = length(x)/N;
% m = floor(m);
%
% if m == 0 % if N > length(x)
%     m = 1;
% end
% %disp(['number of segments used: ',int2str(m)])
%
% % find FFTs and average
% Sxy = zeros(N/2 + 1, 1);
% for idx = 1:m
%     X = fft(w.* ...
%         (x((idx - 1)*N+1:idx*N)));
%     Y = fft(w.* ...
%         (y((idx - 1)*N+1:idx*N)));
%     Sxy = Sxy + ...
%         2*(conj(X(1:N/2+1)).* ...
%         Y(1:N/2+1))/(N*Fs);
% end
%
% wp = sum(w.^2)/N;
% Sxy = Sxy/(m*wp);
% end