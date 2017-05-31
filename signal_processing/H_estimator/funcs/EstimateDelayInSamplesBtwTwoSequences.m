

function [delay_in_samples,  CrossCorrelation] = EstimateDelayInSamplesBtwTwoSequences(InputSignal, OutputSignal)
% Estimates the delay in samples between OutputSignal and the reference sequence InputSignal
% .
% Several methods are available, that are based on cross-correlation and
% generalised cross correlation

% MethodStr = 'Cross-Correlation';


lengthInput = length(InputSignal);
lengthOutput = length(OutputSignal);
CrossCorrelation = xcorr(OutputSignal, InputSignal); %compute cross-correlation between vectors InputSignal and OutputSignal

[~, d] = max(CrossCorrelation); %find value and index of maximum value of cross-correlation amplitude
delay_in_samples = d - max(lengthInput, lengthOutput) + 1; %shift index d, as length(X1)=2*N-1; where N is the length of the signals
delay_in_samples = delay_in_samples - 1;

end