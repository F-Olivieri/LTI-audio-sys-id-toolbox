
function COH = coherencefunction(InputSignal, OutputSignalProcessed, win, N_overlap)
% By Ferdinando Olivieri

Pxx = xspectrum(InputSignal, InputSignal, win, N_overlap);
Pyx = xspectrum(OutputSignalProcessed, InputSignal, win, N_overlap);
Pyy = xspectrum(OutputSignalProcessed, OutputSignalProcessed, win, N_overlap);
COH = (abs(Pyx).^2)./(Pxx.*Pyy);

end