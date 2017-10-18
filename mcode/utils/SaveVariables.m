function SaveVariables(TitleSession, curr_timedomain_file)
% Saving into big matrices for later usage
%TimeDomainRecordings = curr_timedomain_file;

NameFile = [TitleSession];
save(NameFile,'curr_timedomain_file');     % 'IR_estimated','FRF_estimated','COH_estimated',
end