function [DelayedOutputSignal, delay_in_samples] = ...
                AlignTwoSequences(InputSignal, OutputSignal)
            % OutputSignal is the sequence that is shifted
            
            % if size(InputSignal) ~= size(OutputSignal)
            %     display('The input and output sequences do not have the same size. Check their dimensions.');
            %     return;
            % end
            
            if nargin < 4, FracDelayOptionFlag = 0; freqvect = 0; fs = 0; N_fft = 0; end
            InputSignal = InputSignal(:);
            OutputSignal = OutputSignal(:);
            delay_in_samples = HEstimatorTlbx.EstimateDelayInSamplesBtwTwoSequences(InputSignal, OutputSignal);
            % Shift in the time domain
            DelayedOutputSignal = circshift(OutputSignal, -delay_in_samples);
        end
        
        % [1] CLASSIFICATION AND EVALUATION OF DISCRETE SUBSAMPLE TIME DELAY ESTIMATION ALGORITHMS