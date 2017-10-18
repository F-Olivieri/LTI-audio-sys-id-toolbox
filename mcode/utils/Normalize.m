function InputWAVSignalMatrix = Normalize(InputWAVSignalMatrix, MaxGain)
if MaxGain ~= 0
    InputWAVSignalMatrix = AudioTlbx.NormalizeLevel(InputWAVSignalMatrix, MaxGain); % Normalization before playback
else
    InputWAVSignalMatrix = 1*InputWAVSignalMatrix;
end

end