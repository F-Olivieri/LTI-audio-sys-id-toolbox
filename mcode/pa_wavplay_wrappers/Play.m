function Play(InputFile, out_ch_idx, TotalNumbOutputChannels, fs, MaxGain, NumberOfRepeats)
% PLAY is an abstraction to pa_wavplay.
% INPUTS:
% - INPUTFILE is a vector with the input signal
% - OUT_CH_IDX is a scalar indicating the output channel where
% we want to play INPUTFILE
% - TOTALNUMBEROUTPUTCHANNELS is a scalar indicating the total
% number of output channels
% - FS is the sampling frequency (scalar)
% - MAXGAIN is a scalar indicating the maximum value for the
% normalization of the input signal INPUTFILE (i.e., the
% maximum value of INPUTFILE will be equal to MAXGAIN). If
% MaxGain == NaN or MaxGain == 0 there will be no
% normalization. Careful as, in practice, resonable values of
% MaxGain are something like 0.05 or so.
% - NUMBEROFREPEATS is a scalar indicating the total number of
% times the playback of INPUTFILE will be repeated.

if nargin < 6, NumberOfRepeats = 1; end;
if nargin < 5, MaxGain = MultiChAudioTlbx.MaxGainDefault; end;
if nargin < 4, fs = MultiChAudioTlbx.fsDefault; end;

% Adapting the input file to the format required by PA_WAVPLAYRECORD (it needs a matrix)
InputWAVSignalMatrix =  MultiChAudioTlbx.OutputMatrixFormat(InputFile, TotalNumbOutputChannels, out_ch_idx);
InputWAVSignalMatrix = Normalize(InputWAVSignalMatrix, MaxGain);

repeat_idx = 0;
while repeat_idx ~= NumberOfRepeats
    pa_wavplay(InputWAVSignalMatrix, fs, MultiChAudioTlbx.Dev, MultiChAudioTlbx.Drv); % Reproducing the signal
    repeat_idx = repeat_idx + 1;
end

end% Play