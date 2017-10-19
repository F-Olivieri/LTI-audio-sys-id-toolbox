function RepeatFlag = AcceptOrRepeatMeasurement_Prompt
% By Ferdinando Olivieri
% It prompts the user on whether he/she wants to accept of repeat the
% current measurement

AcceptString = input('Do you want to [A]ccept the measurements or [R]epeat them? [A/R] ', 's');

while ~(strcmp(AcceptString, 'A') || strcmp(AcceptString, 'R'))
    disp('Please input A or R.')
    AcceptString = input('Do you want to [A]ccept the measurements or [R]epeat them? [A/R] ', 's');
end

switch AcceptString
    case 'A', RepeatFlag = 0;
    case 'R', RepeatFlag = 1;
end
end%%%