function  TestHardware(MaxGain_INIT, TotalNumOutputChan, MicIdxs)


[y, fs] = audioread('whitenoise_6s_48kHz.wav');

% Testing output audio (if needed)
% Play all the outputs - one at a time (to test speakers)
display('Testing all the speakers, one at a time')
MultiChAudioTlbx.TestOutputChannels(y(1:2*fs), 1, TotalNumOutputChan, fs, MaxGain_INIT, 'AllOutputs-OneAtATime');
%
% Testing output audi: Play Some Channels
ChannelsToPlay = [1:3];
display(['Playing channels' num2str(ChannelsToPlay)])
MultiChAudioTlbx.TestOutputChannels(y, ChannelsToPlay, TotalNumOutputChan, fs, MaxGain_INIT);

% Testing the inputs
if MicIdxs ~= 0
    % Record (To test microphones)
    
    display('Testing microphones')
    q = zeros(length(y), TotalNumOutputChan);
    q(:, 1) = y; % TO record noise level comment this line
    OutputSignal = MultiChAudioTlbx.PlayRecord(q, 1, TotalNumOutputChan, fs, MicIdxs, MaxGain_INIT);
    
    figure;
    subplot(1,2,1); plot(y);
    title('Signal reproduced by the speakers.')
    subplot(1,2,2); plot(OutputSignal);
    title('Signal recorded by the microphones')
    
    fffff = fft(OutputSignal);
    aaa = CreateFrequencyVector(fs, length(OutputSignal));
    fffff = fffff(1:length(OutputSignal)/2 +1);
    figure;
    semilogx(aaa, LogDB(abs(fffff)));
    
    pwelch(OutputSignal, [], [], [], fs, 'twosided'); % Uses default window, overlap & NFFT.
end



end