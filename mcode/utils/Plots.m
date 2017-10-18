function Plots( TimeDomainWave, Estimated_IR, Estimated_FRF, Coherence, MicNumb, fs, N_fft, FilterTitleStr)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

ffreq = CreateFrequencyVector(fs, N_fft);

% Plotting results
numbrow = 2;
numbcol = 3;

freqlimits = [50, 10000];


FigureTlbx.New(['Measurements from Microphone Num ' num2str(MicNumb) ' Filter ' FilterTitleStr], 0);

%             subplot(numbrow,numbcol,[1 2]);
%             plot(TimeDomainWave); title('Time Domain Recorded Data');

subplot(numbrow,numbcol, 1);
plot(Estimated_IR); title('Estimated IR pressure');

subplot(numbrow,numbcol, 2);
semilogx(ffreq, LogDB(abs(Estimated_FRF)));
title('Magnitude FRFs'); xlim(freqlimits); xlabel(labels.FreqInHertz)

subplot(numbrow,numbcol, 3);
semilogx(ffreq, angle(Estimated_FRF));
title('Phase FRFs'); xlim(freqlimits); xlabel(labels.FreqInHertz)

subplot(numbrow,numbcol, 4);
semilogx(ffreq, Coherence);
title('Coherence'); ylim([0, 1.1]);
xlim(freqlimits); xlabel(labels.FreqInHertz)

subplot(numbrow,numbcol, 5);
plot(TimeDomainWave); title('Time domain recording');

end