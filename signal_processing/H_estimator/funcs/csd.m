function [Sxy,f] = csd(x, y, w, Fs)
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

N = length(w);
%disp(['block length: ',int2str(N)])
f = (0:N/2)*Fs/N;

% determine number of averages
m = length(x)/N;
m = floor(m);

if m == 0 % if N > length(x)
    m = 1;
end
%disp(['number of segments used: ',int2str(m)])

% find FFTs and average
Sxy = zeros(N/2 + 1, 1);
for idx = 1:m
    X = fft(w.* ...
        (x((idx - 1)*N+1:idx*N)));
    Y = fft(w.* ...
        (y((idx - 1)*N+1:idx*N)));
    Sxy = Sxy + ...
        2*(conj(X(1:N/2+1)).* ...
        Y(1:N/2+1))/(N*Fs);
end

wp = sum(w.^2)/N;
Sxy = Sxy/(m*wp);

end