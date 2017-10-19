%% Script to measure transfer functions
%% F Olivieri

%% Initialize the turntable
% need to modify this to accommodate your turntable
serial_port = 'COM4';
s = LT360_Turntable_RS232_2_Matlab.init_connection(serial_port);

%%
total_number_of_output_channels = 8;
fs = 48000; % sampling frequency
NumberOfMicrophones  = 1;
MaxGain_Sigs = 0.02;

%% Take measurements
angle_ses = [0:5:180];

Num_angles = length(angle_ses);
for angle_idx = 1:Num_angles
    
    %
    
    % take the measurement
    OutputFile = LTIaudiosysid.PlayRecord(InputSig, out_ch_idx, ...
        total_number_of_output_channels, fs, NumberOfMicrophones, MaxGain_Sigs);
    
    % Prompt
    RepeatFlag = LTIaudiosysid.AcceptOrRepeatMeasurement_Prompt;
    
    switch RepeatFlag
        case 0
            % move turntable
        case 1
            % move
    end
end