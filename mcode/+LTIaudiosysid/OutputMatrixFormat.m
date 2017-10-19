function OutputWAVSignalMatrix = OutputMatrixFormat(InputWavSignal, TotalNumberOfOutputChannels, ChannelsIdxs)
% OUTPUTMATRIXFORMAT adapts the input file to the format required by the multichannel audio function
% InputWavSignal is the input signal (vector SAMPLES*1, or matrix SAMPLES*NCHAN)
% ChannelsIdxs is a vector with the indeces of the output
% channels where the input file is to be reproduced

[n_samples, n_channels] = size(InputWavSignal);

if n_samples == TotalNumberOfOutputChannels % it means that we have a matrix of the form NUMBCH*LENGTH
    InputWavSignal = transpose(InputWavSignal);
end
[n_samples, n_channels] = size(InputWavSignal);

if isvector(InputWavSignal) % The signal is a vector (single channels)
    OutputWAVSignalMatrix = zeros(length(InputWavSignal), TotalNumberOfOutputChannels);
    OutputWAVSignalMatrix(:, ChannelsIdxs) = InputWavSignal;
    
elseif isscalar(InputWavSignal)
    %OutputWAVSignalMatrix = 0;
    error('The Input signal is a scalar. Please choose a different type of input signal');
    
elseif ismatrix(InputWavSignal) % It is a matrix (multichannel audio)
    
    if n_channels == length(ChannelsIdxs)
        OutputWAVSignalMatrix = zeros(n_samples, TotalNumberOfOutputChannels);
        OutputWAVSignalMatrix(:, ChannelsIdxs) = InputWavSignal;
        
    elseif n_channels == TotalNumberOfOutputChannels
        OutputWAVSignalMatrix = InputWavSignal;
    end
end


end%