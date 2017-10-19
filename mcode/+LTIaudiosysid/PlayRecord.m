function OutputFile = PlayRecord(InputSig, out_ch_idx, total_number_of_output_channels, fs, NumberOfMicrophones, MaxGain)
%PLAYRECORD is an abstraction to pa_wavplayrecord

% NumberOfMicrophones is a vector with the channel to be used. Example: use
% channel 1 to 16, NumberOfMicrophones = [1,16];
% If you want to use a specific mic single channel, use NumberOfMicrophones
% = scalar (e.g., NumberOfMicrophones = 1)

if nargin < 5 
    NumberOfMicrophones = [1, 1];
end %It will be a single-channel recording

if size(NumberOfMicrophones) == 1
    NumberOfMicrophones = [NumberOfMicrophones, NumberOfMicrophones];
end % it is not a vector

RecordChanStart = NumberOfMicrophones(1); RecordChanStop = NumberOfMicrophones(2);

% Adapting the input file to the format required by PA_WAVPLAYRECORD (it needs a matrix)
InputWAVSignalMatrix = OutputMatrixFormat(InputSig, total_number_of_output_channels, out_ch_idx);
%InputWAVSignalMatrix = Normalize(InputWAVSignalMatrix, MaxGain);
% InputWAVSignalMatrix = Normalize(InputWAVSignalMatrix, MaxGain);

InputWAVSignalMatrix = Normalize(InputWAVSignalMatrix, MaxGain); % Normalization before playback
% Recording the signal
OutputFile = ...
    pa_wavplayrecord(InputWAVSignalMatrix, MultiChAudioTlbx.Dev, ...
    fs, length(InputSig), RecordChanStart, RecordChanStop, ...
    MultiChAudioTlbx.Dev, MultiChAudioTlbx.Drv);
end% PlayRecord