function TestOutputChannels(WavSignal, ChosenSpeakerIdx, TotalNumbOfOutChannels, fs, MaxGain, TestOptStr, NumbTimesRepeatTest)

if nargin < 7, NumbTimesRepeatTest = 1; end;
if nargin < 5, MaxGain = MultiChAudioTlbx.MaxGainDefault; end;
if nargin < 6, TestOptStr = 'SingleChannels'; end;

PauseBtwEachSpeakerTest = 0.5;
display('Test of the output channels...');
test_idx = 0;
while test_idx ~= NumbTimesRepeatTest
    
    switch TestOptStr
        
        case 'SingleChannels'
            for loudsp_idx = ChosenSpeakerIdx
                MultiChAudioTlbx.Play(WavSignal, loudsp_idx, TotalNumbOfOutChannels, fs, MaxGain);
            end
            
        case 'AllOutputs-OneAtATime'
            
            for loudsp_idx = 1:TotalNumbOfOutChannels
                MultiChAudioTlbx.Play(WavSignal, loudsp_idx, TotalNumbOfOutChannels, fs, MaxGain);
            end
            
        case 'AllOutputs-AtSameTime'
            
            OutputWAVSignalMatrix = ...
                MultiChAudioTlbx.OutputMatrixFormat(WavSignal, TotalNumbOfOutChannels, [1:TotalNumbOfOutChannels]);
            MultiChAudioTlbx.Play(OutputWAVSignalMatrix, 1, TotalNumbOfOutChannels, fs, MaxGain);
        otherwise, ErrorDisplay('Not a valid Playback choice');
    end
    pause(PauseBtwEachSpeakerTest) % If you want to check speaker by speaker...
    test_idx = test_idx + 1;
    
end% while

end%