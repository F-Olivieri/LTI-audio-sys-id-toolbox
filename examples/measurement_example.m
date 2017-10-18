%% F Olivieri

%% Take measurements
angle_ses = [0:5:180];

Num_angles = length(angle_ses);
for angle_idx = 1:Num_angles
    
    % take the measurement
    
    % Prompt
    RepeatFlag = AcceptOrRepeatMeasurement_Prompt;
    
    switch RepeatFlag
        case 0
        case 1
    end
end