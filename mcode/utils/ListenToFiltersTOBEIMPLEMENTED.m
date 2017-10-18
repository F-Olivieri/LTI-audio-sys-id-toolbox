function ListenToFiltersTOBEIMPLEMENTED

MaxGain_INIT = 0.1;

TotalNumOutputChan = 16;
aa = qWN; %a = qWN{1};
%aa = FilterTlbx.ApplyModellingDelayToFilterMatrix(aa);
pause(10)
%bb = circshift(1:32, [0 -16])
MultiChAudioTlbx.Play(aa, 1, TotalNumOutputChan, fs, MaxGain_INIT, 12);

end


