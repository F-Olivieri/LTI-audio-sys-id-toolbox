classdef multiamt
    % multiamt = Multichannel Audio Measurement Toolbox for Windows (Vista
    % or later).
    % A Toolbox for the measurements of impulse responses and frequency
    % response functions of multichannel audio systems.
    
    % It requires on PAWAVPLAY - https://github.com/jgdsens/pa-wavplay
    
    % Author: F Olivieri - ferdinando.olivieri@me.com
    
    % CHOOSE A LICENSE, possibly one that imposes to give back
    % contributions
    
    properties (Constant)
        
        AcceptRepeatStr = 'Do you want to [A]ccept or [R]epeat the measurements? [A/R] ';
        ErrorSessionTypeStr = 'Choose one valid option for the workflow. Type multiamt.help for more info';
        AreYouReadyStr = 'Connect the cable and press return when you are ready.';
        
        labelsFreqInHertz = 'Freq, Hz';
        
        Dev = 0;
        Drv = 'asio';
        MaxGainDefault = 0.05;
        fsDefault = 48000; % Default sampling frequency
        
        TestSignalStr_Default = 'WN'; % White noise
        
        % Workflows
        OneOut2OneInStr = 'OneOut2OneIn';
        OneOut2ManyInStr = 'OneOut2ManyIn'; % Default
        ManyOut2OneInStr = 'ManyOut2OneIn';
        
    end
    
    
    properties(GetAccess = 'public', SetAccess = 'public')
        % public read access, public write access.
        
        SessionNameStr % The title of the session (e.g., MicArrayTFs08082016)
        TestSignalTypeStr % SS (Sine Sweep) / WN (White Noise, default)
        SessionTypeStr % OneOut2OneIn, OneOut2ManyIn (default), ManyOut2OneIn
        
        TotNumIn % The total number of Input Channels on the sound card
        TotNumOut  % The total number of Output Channels on the sound card
        InCh % The input channels to be considered in the measurements
        OutCh % The ouput channels to be considered in the measurements
        
        MaxGain_INIT % The maximum gain in output
        
        TestSignalFileName
        InverseFilterSSFileName % Filename for the inverse filter for SS
        
        SamplingFrequency % In Hz
        DurationTestSignal
        
        ConvolutionLength % Must be a power of 2 (for FFT purposes)
        N_fft % If ConvolutionLength is a power of 2 then N_fft = ConvolutionLength/2 + 1
        
        info % struct that is used to store info
        
    end
    
    
    
    methods%Private Methods
        %         function obj = multiamt(InputSignal, OutputSignal, N_fft, fs, EstimatorType, varargin)
        %             % Returns a SYSTEMRESPONSE object (see systemresponse.m for
        %             % details of the properties and methods) after the estimation
        %             % of the System Response (performed via the HEstimator method)
        %
        %             % Initialise the VARARGIN options
        %             AlignmentBoolean = 1; %perform the estimation with the alignment of the IN/OUT signals and then shift the IR back
        %             GetTheMinimumPhase = 0; %calculates the Minimum Phase component of the IR
        %
        %             if (~isempty(varargin))
        %                 for c = 1:length(varargin)
        %                     switch varargin{c}
        %                         case {'NoInputOutputAlignment'}
        %                             AlignmentBoolean = 0;
        %
        %                         otherwise
        %                             error(['Invalid optional argument, ', ...
        %                                 varargin{c}]);
        %                     end % switch
        %                 end % for each varargin entry
        %             end % if varargin is not empty
        %
        %             if nargin < 3
        %                 obj.N_fft = 2^nextpow2(length(h));
        %             else
        %                 obj.N_fft = N_fft;
        %             end
        %
        %             [h, H, coherence, f, t] = SystemResponseToolbox.HEstimator(InputSignal, OutputSignal, N_fft, fs, EstimatorType, AlignmentBoolean);
        %             %obj = SystemResponse(h, H, t, f, fs, N_fft);% Creates a new SYSTEMRESPONSE object with the new estimated data
        %
        %             obj.IRcoefficients = h;%the coefficients of the Impulse response
        %
        %             if GetTheMinimumPhase
        %                 obj.IR_MinimumPhasePart = Get_Minimum_Phase(h, N_fft); %Calculates the Minimu Phase component of the Impulse response
        %             end
        %
        %             obj.time_vector = transpose(t);
        %
        %             obj.FRF = H; %The frequency response function (FRF)
        %             obj.FRFMagnitude_dB = ConvertIndB(H);
        %             obj.FRFPhase_rads = angle(H);
        %             obj.FRFPhase_degs = rad2deg(angle(H));
        %
        %             obj.fs = fs;
        %             obj.frequency_vector = transpose(f);%(0:obj.N_fft/2)/fs;
        %             obj.COH = coherence;
        %         end
        %
        function obj = multiamt(info)
            
            %             % Default choices
            %             if nargin < 1, SessionNameStr = 'Default'; end
            %             if nargin < 2, TestSignalTypeStr = TestSignalStr_Default; end
            %             if nargin < 3, SessionTypeStr = multiamt.OneOut2ManyInStr; end
            %             %             if nargin < 4, SamplingFrequency = multiamt.fsDefault; end
            
            obj.info = info;
            
            % Create the object
            %             obj.info.SessionNameStr = info.SessionNameStr;
            %             obj.info.TestSignalTypeStr = TestSignalTypeStr;
            %             obj.info.SessionTypeStr = SessionTypeStr;
            
            
            % Input/Output configuration
            %             obj.info.TotNumIn = TotNumIn;
            %             obj.info.TotNumOut = TotNumOut;
            %             obj.info.InCh = InCh;
            %             obj.info.OutCh  = OutCh;
            %
            %             obj.info.MaxGain_INIT = MaxGain_INIT;
            
            % Test signal configuration
            if iscellstr(info.TestSignalFileName)
                % for the SS
                obj.info.TestSignalFileName = cell2mat(info.TestSignalFileName(1));
                obj.info.InverseFilterSSFileName  = cell2mat(info.TestSignalFileName(2));
            else % for the WN (a single file name)
                obj.info.TestSignalFileName =  info.TestSignalFileName;
            end
            
        end
        
        %%
        
        
        
        
    end%%Private Methods
    
    
    
    
    
    
    methods (Static)
        
        function install
            % Run once
            
            %https://github.com/jgdsens/pa-wavplay/archive/master.zip
            url ='https://github.com/jgdsens/pa-wavplay/archive/master.zip';
            display(['Installing PA-WAVPLAY from: ' url ' ...']);
            
            foldName = 'pawavplay';
            [s, mess, messid] = mkdir(foldName, 'newFolder');
            
            ncmFil = unzip(url, foldName);
            
            addpath(foldName);
            
        end
        
        function Prompt_AreYouReady
            
            display(multiamt.AreYouReadyStr);
            pause;
            
        end
        
        
        %% multiamt WORKFLOWS
        
        function obj = RunWorkflow(obj)
            % Performs the multiamt Workflow selected in obj.SessionTypeStr
            
            addpath('pawavplay');
            
            % Run The Workflow
            
            obj.info.SessionTypeStr
            switch obj.info.SessionTypeStr
                
                case multiamt.OneOut2OneInStr
                    % TO BE IMPLEMENTED
                    %a = 1
                case multiamt.OneOut2ManyInStr
                    
                    multiamt.Workflow_OneOut2ManyIn(obj);
                case multiamt.ManyOut2OneInStr
                    % TO BE IMPLEMENTED
                    %a = 3
                otherwise
                    display(multiamt.ErrorSessionTypeStr);
            end
        end
        
        %%
        function Workflow_OneOut2ManyIn(obj)
            display('Workflow_OneOut2ManyIn')
            
            [TestSignal, obj.info.SamplingFrequency]  = audioread(obj.info.TestSignalFileName);
            
            DurationTestSignal = length(TestSignal);
            obj.info.DurationTestSignal = DurationTestSignal;
            obj.info.TotNumIn
            obj.info.TotNumOut
            
            
            
            
            
            switch obj.info.TestSignalTypeStr
                case 'WN'
                    %                     IRs = MatrixOfZeros;
                    %                     FRFs = MatrixOfZeros;
                case 'SS'
                    %% TODO: IRs duration power of Two
                    
                    
                    %                     TestSignal = MakeSignalLengthPowerOfTwo(TestSignal);
                    
                    InverseFilter  = audioread(obj.info.InverseFilterSSFileName);
                    InverseFilter = MakeSignalLengthPowerOfTwo(InverseFilter);
                    
                    
                    ConvolutionLength = max(2^nextpow2(obj.info.DurationTestSignal), length(InverseFilter));
                    
                    obj.info.N_fft = ConvolutionLength/2 + 1;
                    
                    
                    IRs = zeros(obj.info.TotNumIn, obj.info.TotNumOut, ConvolutionLength);
                    FRFs = zeros(obj.info.TotNumIn, obj.info.TotNumOut, obj.info.N_fft);
                    
                    % TODO:check fs of test signal is equal to that of the
                    % inverse filter
            end
            DateTimeStart = datetime('now');
            obj.info.DateTimeStart = DateTimeStart;
            display(['Measurements started on ' datestr(DateTimeStart)]);
            
            for out_idx = 1:length(obj.info.OutCh)  % each combination
                curr_out_ch = obj.info.OutCh(out_idx);
                TitleStr = ['out' num2str(curr_out_ch)];
                display(TitleStr);
                %multiamt.Prompt_AreYouReady; %% TODO Make it optional
                
                
                RepeatFlag = 1;
                
                while RepeatFlag
                    close all; % Closing all the active figures
                    
                    % Recordings
                    TimeDomainFiles = transpose(multiamt.PlayRecord(TestSignal, curr_out_ch, ...
                        obj.info.TotNumOut, obj.info.SamplingFrequency, obj.info.InCh, obj.info.MaxGain_INIT));
                    
                    % Estimation
                    switch obj.info.TestSignalTypeStr
                        case 'WN'
                            % TODO
                        case 'SS'
                            [IR_estimated, FRF_estimated] = multiamt.SS_InvFilt(TimeDomainFiles, InverseFilter);
                            multiamt.Plots_SS(IR_estimated, FRF_estimated, obj.info.SamplingFrequency, TitleStr);
                        otherwise
                            display('not a valid option');
                    end
                    
                    
                    
                    IRs(:, out_idx, :) = IR_estimated;
                    FRFs(:, out_idx, :) = FRF_estimated;
                    
                    
                    
                    SaveTimeDomainFile([obj.info.SessionNameStr '_out' num2str(out_idx)], TimeDomainFiles);
                    
                    RepeatFlag = multiamt.Prompt_AcceptOrRepeatMeasurement; % ACCEPT MEASUREMENTS? PROMPT
                    
                end% RepeatFlag
                
            end%for each combination
            
            % Save all the data
            DateTimeStart = datetime('now');
            obj.info.DateTimeEnd = DateTimeStart;
            display(['Measurements completed on '  datestr(DateTimeStart)]);
            SaveIRsAndFRFs(obj.info.SessionNameStr, IRs, FRFs, obj.info);
        end
        
        %%
        
        
        function Plot_Estimation_Results(obj)
            % Plotting the results of the HEstimator
            
            %Preset = get(handles.PreSetAxisChk,'value');
            figure('name','System Response Estimation');
            subplot(2,2,1); obj.PlotIR(obj); title('Estimated IR'); grid on;
            subplot(2,2,2); semilogx(obj.frequency_vector, obj.COH); title('Coherence'); xlim([50 (obj.fs)/2]); grid on;
            subplot(2,2,3); obj.Plot_Magnitude_FRF(obj); title('Estimated Magnitude FRF, dB');   xlim([50 (obj.fs)/2]); grid on;
            subplot(2,2,4); obj.Plot_Phase_FRF(obj); title('Estimated Phase FRF, rads');  xlim([50 (obj.fs)/2]); grid on;
        end%ploteverything
        
        %% Methods for the IR
        function PlotIR(obj, x_axis_range)
            
            plot(obj.IRcoefficients);
            
            if nargin < 2
                xlim([0 length(obj.IRcoefficients) - 1]);
            else
                xlim([x_axis_range(1) x_axis_range(2)]);
            end
            
            xlabel('Samples'); ylabel('Impulse Response Amplitude');
        end%plotIR
        
        
        
        
        
        %% Methods for the FRF
        function Plot_Magnitude_FRF(obj)
            semilogx(obj.frequency_vector, obj.FRFMagnitude_dB);
            xlim([50 obj.fs/2]);
            xlabel('Frequency, Hz'); ylabel('Magnitude FRF');
        end%Plot_Magnitude_FRF
        
        function Plot_Phase_FRF(obj, UnwrappedBoolean)
            if nargin < 2
                UnwrappedBoolean = 0;
            end
            
            string1 = 'semilogx(obj.frequency_vector, ';
            
            if UnwrappedBoolean
                string2 = 'unwrap(obj.FRFPhase_rads)';
            else
                string2 = 'obj.FRFPhase_rads';
                
            end
            
            string3 = ' );';% closes the semilogx function in string1
            eval([string1 string2 string3]);
            xlim([50 obj.fs/2]);
            xlabel('Frequency, Hz'); ylabel('Phase FRF, rads');
        end%Plot_Phase_FRF
        
        function Plots2(MethodStr, h, H, fs, COH, FigureNameTextStr)
            %PLOTS Summary of this function goes here
            % MethodStr can be 'SineSweep' 'HEstimator'
            if nargin < 6
            else
                figure('Name', [FigureNameTextStr ' WhiteNoise']);
            end
            
            N_fft = length(h);
            
            switch MethodStr
                case 'HEstimator'
                    NumberOfPlots = 4; % IR, Magnitude FRF, Phase FRF, COH
                case  'SineSweep'
                    NumberOfPlots = 3; % Same as above but it does not include Coherence
            end
            
            
            samplevect = (0:(N_fft - 1));
            timevect = samplevect/fs;
            freqvect = linspace(0, fs/2, N_fft/2 + 1);
            
            % IR
            subplot(NumberOfPlots, 1, 1);
            plot(samplevect, h); title('Estimated IR'); xlabel('Samples'); ylabel('Amplitude');
            xlim([samplevect(1), samplevect(end)]);
            
            % Code to plot a double x-axis (samples and time)
            %             Ax1 = gca;
            %             Ax2 = axes('Position', get(Ax1,'Position'), 'XAxisLocation','top');
            %             plot(timevect, h, 'color','k','parent', Ax2);
            %             xlim([timevect(1), timevect(end)]);
            grid on;
            
            % Magnitude
            subplot(NumberOfPlots, 1, 2); semilogx(freqvect, LogDB(abs(H)));
            title('Estimated Magnitude FRF'); xlabel('Freq, Hz'); ylabel('dB');
            grid on; xlim([50, fs/2]);
            
            % Phase
            subplot(NumberOfPlots, 1 ,3);
            semilogx(freqvect, unwrap(angle((H))));
            title('Estimated FRF Unwrapped Phase'); xlabel('Freq, Hz');xlim([50, fs/2]);  grid on;
            
            if strcmp(MethodStr, 'HEstimator') % then plot the COH
                
                subplot(NumberOfPlots, 1, 4); semilogx(freqvect, COH); title('Coherence'); xlabel('Freq, Hz');
                grid on; xlim([50, fs/2]);
            end
        end
        
        
        %% MULTI-CHANNEL AUDIO TOOLBOX
        
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
            if nargin < 5, MaxGain = multiamt.MaxGainDefault; end;
            if nargin < 4, fs = multiamt.fsDefault; end;
            
            
            % Adapting the input file to the format required by PA_WAVPLAYRECORD (it needs a matrix)
            InputWAVSignalMatrix =  OutputMatrixFormat(InputFile, TotalNumbOutputChannels, out_ch_idx);
            InputWAVSignalMatrix = Normalize(InputWAVSignalMatrix, MaxGain);
            
            repeat_idx = 0;
            while repeat_idx ~= NumberOfRepeats
                pa_wavplay(InputWAVSignalMatrix, fs, MultiChAudioTlbx.Dev, MultiChAudioTlbx.Drv); % Reproducing the signal
                repeat_idx = repeat_idx + 1;
            end
            
        end% Play
        
        %%
        function OutputFile = PlayRecord(InputFile, out_ch_idx, total_number_of_output_channels, fs, NumberOfMicrophones, MaxGain)
            %PLAYRECORD is an abstraction to pa_wavplayrecord
            
            % NumberOfMicrophones is a vector with the channel to be used. Example: use
            % channel 1 to 16, NumberOfMicrophones = [1,16];
            % If you want to use a specific mic single channel, use NumberOfMicrophones
            % = scalar (e.g., NumberOfMicrophones = 1)
            
            if nargin < 5, NumberOfMicrophones = [1, 1]; end; %It will be a single-channel recording
            
            if size(NumberOfMicrophones) == 1, NumberOfMicrophones = [NumberOfMicrophones, NumberOfMicrophones]; end; % it is not a vector
            
            RecordChanStart = NumberOfMicrophones(1); RecordChanStop = NumberOfMicrophones(2);
            
            % Adapting the input file to the format required by PA_WAVPLAYRECORD (it needs a matrix)
            InputWAVSignalMatrix = OutputMatrixFormat(InputFile, total_number_of_output_channels, out_ch_idx);
            %InputWAVSignalMatrix = Normalize(InputWAVSignalMatrix, MaxGain);
            
            
            % Recording the signal
            OutputFile = pa_wavplayrecord(InputWAVSignalMatrix, multiamt.Dev, ...
                fs, length(InputFile), RecordChanStart, RecordChanStop, ...
                multiamt.Dev, multiamt.Drv);
            
        end% PlayRecord
        
        %%
        
        
        
        
        %%
        function TestOutputChannels(WavSignal, ChosenSpeakerIdx, TotalNumbOfOutChannels, fs, MaxGain, TestOptStr, NumbTimesRepeatTest)
            
            if nargin < 7, NumbTimesRepeatTest = 1; end;
            if nargin < 5, MaxGain = MultiChAudioTlbx.MaxGainDefault; end;
            if nargin < 6, TestOptStr = 'SingleChannels'; end;
            MaxGain
            PauseBtwEachSpeakerTest = 0.5;
            
            test_idx = 0;
            while test_idx ~= NumbTimesRepeatTest
                
                switch TestOptStr
                    
                    case 'SingleChannels'
                        for loudsp_idx = ChosenSpeakerIdx
                            multiamt.Play(WavSignal, loudsp_idx, TotalNumbOfOutChannels, fs, MaxGain);
                        end
                        
                    case 'AllOutputs-OneAtATime'
                        
                        for loudsp_idx = 1:TotalNumbOfOutChannels
                            multiamt.Play(WavSignal, loudsp_idx, TotalNumbOfOutChannels, fs, MaxGain);
                        end
                        
                    case 'AllOutputs-AtSameTime'
                        
                        OutputWAVSignalMatrix = ...
                            OutputMatrixFormat(WavSignal, TotalNumbOfOutChannels, [1:TotalNumbOfOutChannels]);
                        multiamt.Play(OutputWAVSignalMatrix, 1, TotalNumbOfOutChannels, fs, MaxGain);
                    otherwise, error('Not a valid Playback choice');
                end
                pause(PauseBtwEachSpeakerTest) % If you want to check speaker by speaker...
                test_idx = test_idx + 1;
                
            end% while
            
        end%
        
        function ConvolvedAudioMatrix = ConvolutionOfIRWithFilterMatrix(FilterMatrixIRs, IR)
            
            [a, b] = size(FilterMatrixIRs);
            N_fft = max(a, b); % Assuming N_fft is always larger than the number of filters in the matrix
            NumberOfFilters = min(a,b);
            
            for filter_idx = 1:NumberOfFilters
                current_speaker_filter = squeeze(FilterMatrixIRs(filter_idx, :));
                FiltTemp = conv(current_speaker_filter, IR);
                
                ConvolvedAudioMatrix(filter_idx, :) = FiltTemp;%ifft(FiltTemp, N_fft, 'symmetric');
                
            end
            
        end
        
        
        %%%%
        
        %%
        function PrintAcronyms()
            % Prints a list of acronyms used in the making of the toolbox
            idx = 1;
            list(idx).acr = 'SS'; list(idx).meaning = 'Sine Sweep'; idx = idx + 1;
            list(idx).acr = 'IR'; list(idx).meaning = 'Impulse Response'; idx = idx + 1;
            list(idx).acr = 'FRF'; list(idx).meaning = 'Frequency Response Function'; idx = idx + 1;
            list(idx).acr = 'WN'; list(idx).meaning = 'White Noise'; idx = idx + 1;
            
            
            for id = 1:length(list)
                display([list(id).acr ' = ' list(id).meaning]);
            end
        end
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% WHITE NOISE (WN) MEASUREMENTS
        
        
        
        function Plots_WN(h, H, COH, fs, FigureNameTextStr)
            %PLOTS Summary of this function goes here
            %   Detailed explanation goes here
            if nargin < 4
            else
                figure('Name', [FigureNameTextStr ' WhiteNoise']);
            end
            
            N_fft = length(h);
            
            samplevect = (0:(N_fft - 1));
            timevect = samplevect/fs;
            freqvect = linspace(0, fs/2, N_fft/2+ 1);
            
            subplot(2,2,1); plot(timevect, h); title('Estimated IR');
            xlabel('Samples'); ylabel('Amplitude'); grid on;
            xlim([timevect(1), timevect(end)]); ylim([-8*10^-3, 8*10^-3]);
            
            %Ax1 = gca;
            %Ax2 = axes('Position', get(Ax1,'Position'), 'XAxisLocation','top');
            % plot(samplevect, h, 'color','k','parent', Ax2);
            %xlim([samplevect(1), samplevect(end)]);
            
            subplot(2,2,2); semilogx(freqvect, LogDB(abs(H))); title('Estimated Magnitude FRF'); xlabel('Freq, Hz'); ylabel('dB');
            grid on; xlim([50, fs/2]);
            
            subplot(2,2,3); semilogx(freqvect, unwrap(angle((H)))); title('PHase'); xlabel('Freq, Hz');xlim([50, fs/2]);  grid on;
            
            subplot(2,2,4); semilogx(freqvect, COH); title('Coherence'); xlabel('Freq, Hz');
            grid on; xlim([50, fs/2]);
        end
        
        %%
        
        
        
        
        %%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% SINE SWEEP (SS) MEASUREMENTS
        
        % Using a sine sweep measurement
        %[h, H] = SineSweep(SineSweepInputSignalWAV, OutputSignalRecorded, InverseFilterWAV, N_fft, fs);
        function [h_IR, H_fft] = SS_InvFilt(RecordedSineSweep, InverseFilter, NormalizationFactor)
            % Apply_SS_InvFilt: Returns the IRs and FRFs  Summary of this function goes here
            %   Detailed explanation goes here
            
            % If RecordedSineSweep is a multichannel audio matrix, it must be NUMFILTER*TIMEDURATION
            
            if nargin < 3
                NormalizationFactor = 1;
            end
            
            %NumDimsMatrixSineSweep = ndims(RecordedSineSweep);
            [NumFilter, ~] = size(RecordedSineSweep); % it is assumed to be the minimum
            
            % Zero Padding of the recorded signal
            for idx = 1:NumFilter
                RecordedSineSweep2(idx, :) =  MakeSignalLengthPowerOfTwo(RecordedSineSweep(idx, :));
            end
            RecordedSineSweep = RecordedSineSweep2;
            clear RecordedSineSweep2;
            
            [~, TimeDuration] = size(RecordedSineSweep); % it is assumed to be the minimum
            
            lengthConvolution = TimeDuration; % Because of the frequency domain convolution
            h_IR = zeros(NumFilter, lengthConvolution);
            Nfft = lengthConvolution/2 + 1;
            H_fft = zeros(NumFilter, Nfft);
            for idx = 1:NumFilter
                
                [temph, tempH] = fconv(squeeze(RecordedSineSweep(idx, :)), InverseFilter);
                %temph = conv(curr_rec, InverseFilter);
                
                temph = temph/NormalizationFactor;
                tempH = tempH/NormalizationFactor;
                
                
                h_IR(idx, :) = temph; % Normalising by the sampling frequency
                H_fft(idx, :) = tempH;
                
            end
            
            %% TODO: complex FFT at the Nyquist freq
            H_fft(:, end) % FRF at Nyquist freq.
            
        end
        
        function Plots_SS(h, H, fs, FigureNameTextStr)
            
            %PLOTS Summary of this function goes here
            %   Detailed explanation goes here
            if nargin < 4
            else
                figure('Name', [FigureNameTextStr ' WhiteNoise']);
            end
            
            N_fft = length(h);
            
            samplevect = (0:(N_fft - 1));
            timevect = samplevect/fs;
            freqvect = linspace(0, fs/2, N_fft/2 + 1);
            
            subplot(3,1,1); plot(timevect, h); title('Estimated IR'); xlabel('Samples'); ylabel('Amplitude'); grid on;
            xlim([timevect(1), timevect(end)]);
            
            Ax1 = gca;
            Ax2 = axes('Position', get(Ax1,'Position'), 'XAxisLocation','top');
            plot(samplevect, h, 'color','k','parent', Ax2);
            xlim([samplevect(1), samplevect(end)]);
            
            subplot(3,1,2); semilogx(freqvect, LogDB(abs(H))); title('Estimated Magnitude FRF'); xlabel('Freq, Hz'); ylabel('dB');
            grid on; xlim([50, fs/2]);
            
            subplot(3,1,3); semilogx(freqvect, unwrap(angle((H)))); title('PHase'); xlabel('Freq, Hz');xlim([50, fs/2]);  grid on;
            
            
        end
        
        
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %%
        
        
        %%
        function  TestHardware(MaxGain_INIT, TotalNumOutputChan, MicIdxs)
            
            [y, fs] = audioread('whitenoise_6s_48kHz.wav');
            
            % Testing audio (if needed)
            % Play all the outputs - one at a time (to test speakers)
            display('Testing all the speakers, one at a time')
            multiamt.TestOutputChannels(y(1:2*fs), 1, TotalNumOutputChan, fs, MaxGain_INIT, 'AllOutputs-OneAtATime');
            %
            % Play Some Channels
            ChannelsToPlay= [1:3];
            display(['Playing channels' num2str(ChannelsToPlay)])
            multiamt.TestOutputChannels(y,ChannelsToPlay, TotalNumOutputChan, fs, MaxGain_INIT);
            
            if MicIdxs ~= 0
                % Record (To test microphones)
                
                display('Testing microphones')
                q_LS = zeros(length(y), TotalNumOutputChan);
                q_LS(:, 1) = y; % TO record noise level comment this line
                OutputSignal = multiamt.PlayRecord(q_LS, 1, TotalNumOutputChan, fs, MicIdxs, MaxGain_INIT);
                
                figure;
                subplot(1,2,1); plot(y);
                title('Signal reproduced by the speakers.')
                subplot(1,2,2); plot(OutputSignal);
                title('Signal recorded by the microphones')
                
                fffff = fft(OutputSignal);
                aaa = CreateFrequencyVector(fs, length(OutputSignal));
                fffff = fffff(1:length(OutputSignal)/2 +1);
                figure;
                semilogx(aaa, 20*log10(abs(fffff)));
                
                pwelch(OutputSignal,[],[],[],48000,'twosided'); % Uses default window, overlap & NFFT.
                
                %coherencefunction(
            end
            
            
            
        end
        
        %%
        function Plots( TimeDomainWave, Estimated_IR, Estimated_FRF, Coherence, MicNumb, fs, N_fft, FilterTitleStr)
            %UNTITLED2 Summary of this function goes here
            %   Detailed explanation goes here
            
            ffreq = CreateFrequencyVector(fs, N_fft);
            
            % Plotting results
            numbrow = 2;
            numbcol = 2;
            
            freqlimits = [50, 10000];
            
            FigureNew(['Time domain Measurements from Microphone Num ' num2str(MicNumb) ' Filter ' FilterTitleStr], 0);
            plot(TimeDomainWave); title('Time domain recording');
            
            FigureNew(['Measurements from Microphone Num ' num2str(MicNumb) ' Filter ' FilterTitleStr], 0);
            
            %             subplot(numbrow,numbcol,[1 2]);
            %             plot(TimeDomainWave); title('Time Domain Recorded Data');
            
            subplot(numbrow,numbcol, 1);
            plot(Estimated_IR); title('Estimated IR pressure');
            
            subplot(numbrow,numbcol, 2);
            semilogx(ffreq, LogDB(abs(Estimated_FRF)));
            title('Magnitude FRFs'); xlim(freqlimits); xlabel(multiamt.labelsFreqInHertz)
            
            subplot(numbrow,numbcol, 3);
            semilogx(ffreq, angle(Estimated_FRF));
            title('Phase FRFs'); xlim(freqlimits); xlabel(multiamt.labelsFreqInHertz)
            
            subplot(numbrow,numbcol, 4);
            semilogx(ffreq, Coherence);
            title('Coherence'); ylim([0, 1.1]);
            xlim(freqlimits); xlabel(multiamt.labelsFreqInHertz)
            
        end
        
        
        
        %%
        
        function RepeatFlag = Prompt_AcceptOrRepeatMeasurement
            % Prompts the user with an option to accept or repeat the measurements
            
            AcceptString = input(multiamt.AcceptRepeatStr, 's');
            while ~(strcmp(AcceptString, 'A') || strcmp(AcceptString, 'R'))
                display('Please input A or R.')
                AcceptString = input(multiamt.AcceptRepeatStr, 's');
            end
            
            switch AcceptString
                case 'A', RepeatFlag = 0;
                case 'R', RepeatFlag = 1;
            end
        end
    end
    
end


%% PRIVATE FUNCTIONS

%%
function SaveTimeDomainFile(TitleSession, curr_timedomain_file)
% Saving into big matrices for later usage
%TimeDomainRecordings = curr_timedomain_file;
save(TitleSession, 'curr_timedomain_file');     % 'IR_estimated','FRF_estimated','COH_estimated',
end


function SaveIRsAndFRFs(TitleSession, IR_estimated, FRF_estimated, info)
% Saving into big matrices for later usage
%TimeDomainRecordings = curr_timedomain_file;
save(TitleSession, 'IR_estimated','FRF_estimated', 'info');     % ,'COH_estimated',
end

function [y, Y] = fconv(Signal1, Signal2)
%FCONV Fast Convolution
%   [y] = FCONV(x, h) convolves x and h
%
%      x = input vector
%      h = input vector
Signal1 = Signal1(:);
Signal2 = Signal2(:);

Dim1 = length(Signal1);
Dim2 = length(Signal2);

if Dim1 > Dim2
    Signal2 = [Signal2; zeros(Dim1 - Dim2, 1)];
elseif Dim2 > Dim1
    Signal1 = [Signal1; zeros(Dim2 - Dim1, 1)];
end

X = FFTSingleSide(Signal1);
H = FFTSingleSide(Signal2);
Y = X.*H;        	           %
y = ifft(Y, max(length(Signal1), length(Signal2)), 'symmetric');
end
%y = y(1:Ly);               % Take just the first N elements
%Ly = length(Signal1) + length(Signal2) - 1;  %
%Ly2 = pow2(nextpow2(Ly));    % Find smallest power of 2 that is > Ly


%     properties(GetAccess = 'public', SetAccess = 'private')
%         % public read access, but private write access.
%
%         IRcoefficients
%         FRF
%         time_vector
%         frequency_vector

%
%         % USEFUL PROPERTIES FOR PROCESSING
%         % These properties are included to facilitate the access to
%         % the data from external script. Example: in order to plot the
%         % magnitude of the FRF use
%         % >> plot(ObjectName.FRFMagnitude_dB)
%         % instead of
%         % >> plot(20*log10(abs(ObjectName.FRF)))
%
%         %         COH; %Coherence of the estimation
%         %         FRFMagnitude_dB % The 20*log10(abs(FRF))
%         %         FRFPhase_rads % angle(FRF)
%         %         FRFPhase_degs % rad2deg(angle(FRF))
%         %         IR_MinimumPhasePart % The minimum phase part of the IR (see Get_Minimum_Phase method)
%         %         IRpeakvector % The vector of the peaks of the IR
%
%         % TO BE IMPLEMENTED
%         % Group Delay Calculation [Gd,F] = grpdelay(SystemRespObj.IRcoefficients,1,SystemRespObj.N_fft,SystemRespObj.fs);
%     end



%
%     properties(Constant, Access = 'protected') %(GetAccess = 'protected', SetAccess = 'protected') % private read and write access
%
%
%     end

function [ output_args ] = LogDB( input_args, MultiplyingFactor, reference_value, RoundingFlag)
%
% input_args is the value to be converted in dB
% MultiplyingFactor could be 10 or 20 depending on the conversion needed
% (default = 20)
% reference_value is the reference value for the dB conversion (default =
% 1)

if nargin < 4, RoundingFlag = 0; end;
if nargin < 2, MultiplyingFactor = 20; end;
if nargin < 3, reference_value = 1; end;

output_args = MultiplyingFactor*log10(input_args/reference_value);
if RoundingFlag
    output_args = roundn(output_args, -2);
end
end


function [ Signal ] = MakeSignalLengthPowerOfTwo(Signal)
% Rounds the length of the Signal vector to the closest power of two
Signal = Signal(:);
lS = length(Signal);
aaa = 2^nextpow2(lS);
Signal = [Signal; zeros(aaa - lS, 1)];
end

function OutWAVSigMatrix = ...
    OutputMatrixFormat(InWavSig, TotNumOutChans, ChanIdxs)
% OUTPUTMATRIXFORMAT adapts the input file to the format required by the multichannel audio function
% InputWavSignal is the input signal (vector SAMPLES*1, or matrix SAMPLES*NCHAN)
% ChannelsIdxs is a vector with the indeces of the output
% channels where the input file is to be reproduced

[n_samples, n_channels] = size(InWavSig);

if n_samples == TotNumOutChans % it means that we have a matrix of the form NUMBCH*LENGTH
    InWavSig = transpose(InWavSig);
end
[n_samples, n_channels] = size(InWavSig);

if isvector(InWavSig) % The signal is a vector (single channels)
    OutWAVSigMatrix = zeros(length(InWavSig), TotNumOutChans);
    
    for ch_idx = ChanIdxs
        OutWAVSigMatrix(:, ch_idx) = InWavSig;
    end
    
elseif isscalar(InWavSig)
    OutWAVSigMatrix = 0;
    error('The Input signal is a scalar. Please choose a different type of input signal');
    
elseif ismatrix(InWavSig) % It is a matrix (multichannel audio)
    
    if n_channels == length(ChanIdxs)
        OutWAVSigMatrix = zeros(length(InWavSig), TotNumOutChans);
        for ch_idx = 1:length(ChanIdxs)
            OutWAVSigMatrix(:, ch_idx) = InWavSig(ch_idx, :);
        end
    elseif n_channels == TotNumOutChans
        OutWAVSigMatrix = InWavSig;
    end
end


end%

function InputWAVSignalMatrix = Normalize(InputWAVSignalMatrix, MaxGain)
if MaxGain ~= 0
    InputWAVSignalMatrix = NormalizeLevel(InputWAVSignalMatrix, MaxGain); % Normalization before playback
else
    InputWAVSignalMatrix = 1*InputWAVSignalMatrix;
end

end


function [ Freq, Omega ] = CreateFrequencyVector( fs, N_fft )
%UNTITLED14 Summary of this function goes here
%   Detailed explanation goes here


Freq = linspace(0, fs/2, LengthFreqVect(N_fft)); % Frequency Vector
Omega = 2*pi*Freq; % Angular frequency Vector

end

function [AudioSignal, gain] = NormalizeLevel(AudioSignal, max_gain_linear)
% AudioSignal can be a matrix or a vector of dimensions samples X
% N_channels

if nargin < 2
    max_gain_linear = 0.96;
end

[DurationSamples, N_ch] = size(AudioSignal);

% Calculating the normalization gain. If the AudioSignal is a very large
% matrix, Matlab trhows an error of out of memory. hence the for loop.
% Initially it used to be
%max_value = MatrixMaxValue(abs(AudioSignal(:))); % Get the max of the magnitude of the filters
% but now I implemetn it with a for loop
max_value_vect = zeros(1, N_ch);
for ch_idx = 1:N_ch
    max_value_vect(ch_idx) = MatrixMaxValue(abs(AudioSignal(:, ch_idx))); % Get the max of the magnitude of the filters
end
%AudioSignal = AudioSignal; % First normalization. After this step max(AudioSignal) = 1;
max_value = max(abs(max_value_vect));

gain = max_gain_linear/max_value;

% Same as above. If matrix if very big, Matlab 32 bit runs out of memory
% hence the for loop. The original command was
AudioSignal = gain*AudioSignal; % Apply the max linear gain
% for ch_idx = 1:N_ch
%     AudioSignal(:, ch_idx) = gain*(AudioSignal(:, ch_idx)); % Get the max of the magnitude of the filters
% end

end

function fighandle = FigureNew(FigureNameStr, FullScreenFlag)
% Creates an empty figure with a given name and white
% background. The fullScreenFlag will create a full screen
% figure
if nargin < 2, FullScreenFlag = 1; end;
if nargin < 1, FigureNameStr = ''; end;

%figobj.handle = figure;
fighandle = figure;
%              ReturnHandle(figobj);

%             FigureTlbx.SetName(FigureNameStr, fighandle);

set(figHandle, 'name', FigureNameStr);
if FullScreenFlag,  set(figHandle,'units','normalized','outerposition',[0 0 1 1]); end;
%             FigureTlbx.SetBackgroundColour('', fighandle);

set(figHandle, 'color', 'w');

%figobj.name = FigureNameStr;
end


function SignalFFT = FFTSingleSide( Signal )
% Single-sided FFT spectrum of the time domain sequence Signal

N_fft = length(Signal);
if is_odd(N_fft)
    error('Please give an input signal which has a length that is power of two')
else
    
    lengthfreqvect = N_fft/2 + 1;
    
    SignalFFT = fft(Signal, N_fft);
    SignalFFT = SignalFFT(1:lengthfreqvect);
    
    
end

end

function lengthfreqvector = LengthFreqVect(N_fft)
%This function serves to ensure that all the length of freq vectors in the
%toolbox are the same for all the functions

lengthfreqvector = N_fft/2 + 1;

end

function [ boolean ] = is_odd( number )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

if mod(number, 2) == 0
    %number is even
    boolean = 0;
else
    %number is odd
    boolean = 1;
end

end

function [max_value] = MatrixMaxValue(Matrix)
% Recursive function that returns the maximum value in a matrix
% F Olivieri, 20 Dec 2013



if numel(Matrix) ~= 1
    temp_max = max(Matrix);
    max_value = MatrixMaxValue(temp_max);
else
    max_value = max(Matrix);
end


end



