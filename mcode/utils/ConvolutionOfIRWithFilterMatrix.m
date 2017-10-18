function ConvolvedAudioMatrix = ConvolutionOfIRWithFilterMatrix(FilterMatrixIRs, IR)

[a, b] = size(FilterMatrixIRs);
N_fft = max(a,b); % Assuming N_fft is always larger than the number of filters in the matrix
NumberOfFilters = min(a,b);

for filter_idx = 1:NumberOfFilters
    current_speaker_filter = squeeze(FilterMatrixIRs(filter_idx, :));
    FiltTemp = conv(current_speaker_filter, IR);
    
    ConvolvedAudioMatrix(filter_idx, :) = FiltTemp;%ifft(FiltTemp, N_fft, 'symmetric');
    
end

end