function H_Estimator_Plots(h, H, COH, fs, FigureNameTextStr)
%PLOTS Summary of this function goes here
%   Detailed explanation goes here
if nargin < 4
else
    figure('Name', [FigureNameTextStr ' WhiteNoise']);
end

N_fft = length(h);

samplevect = (0:(N_fft - 1));
timevect = samplevect/fs;
freqvect = linspace(0, fs/2, N_fft/2+ 1);

subplot(2,2,1); plot(timevect, h); title('Estimated IR');
xlabel('Samples'); ylabel('Amplitude'); grid on;
xlim([timevect(1), timevect(end)]); ylim([-8*10^-3, 8*10^-3]);

%Ax1 = gca;
%Ax2 = axes('Position', get(Ax1,'Position'), 'XAxisLocation','top');
% plot(samplevect, h, 'color','k','parent', Ax2);
%xlim([samplevect(1), samplevect(end)]);

subplot(2,2,2); semilogx(freqvect, LogDB(abs(H))); title('Estimated Magnitude FRF'); xlabel('Freq, Hz'); ylabel('dB');
grid on; xlim([50, fs/2]);

subplot(2,2,3); semilogx(freqvect, unwrap(angle((H)))); title('PHase'); xlabel('Freq, Hz');xlim([50, fs/2]);  grid on;

subplot(2,2,4); semilogx(freqvect, COH); title('Coherence'); xlabel('Freq, Hz');
grid on; xlim([50, fs/2]);
end