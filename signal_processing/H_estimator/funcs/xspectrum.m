
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