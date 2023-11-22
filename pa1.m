function pa1
%loading data from the file

fileName1 = 'C:\Users\Puviarasu N P\OneDrive\Desktop\qpsk_I_seqLen65472.mat';
loadfile1 = load(fileName1);
table1 = struct2cell(loadfile1);
data_total_I = cell2mat(table1);

fileName2 = 'C:\Users\Puviarasu N P\OneDrive\Desktop\qpsk_Q_seqLen65472.mat';
loadfile2 = load(fileName2);
table2 = struct2cell(loadfile2);
data_total_Q = cell2mat(table2);

len = length(data_total_I);
data_total = zeros(1,len);
for i = 1:len
  data_total(i) = data_total_I(i) + 1j*data_total_Q(i);
end


fprintf('Loaded data\n'); 
%input scaling in dB
scale_val = max(abs(data_total));
data_total = data_total/scale_val;
fprintf('Input scaling done\n');


[x_train,y_train] =  Generate_Data(); 

%{Lambda =  x - 0.1*K.sum(x*x,axis =1,keepdims=True)*x))


layers = [ ...
    sequenceInputLayer(2)
    lstmLayer(12,'OutputMode','sequence')
    fullyConnectedLayer(7)
    fullyConnectedLayer(2)
    regressionLayer];


options = trainingOptions('adam', ...
    'MaxEpochs',4000, ...
    'MiniBatchSize', 100, ...
    'InitialLearnRate', 0.005, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);
net = trainNetwork(y_train',x_train',layers,options)

[x_test,y_test] =  load_data(data_total);     

classes = predict(net,x_test','MiniBatchSize',100);
classes = classes';
output = pa(Convert2Complex(classes(1:4001,:)));
figure(1);
plot(abs(Convert2Complex(x_test(1:4001,:))),abs(output),'-r','LineWidth',4)
hold on 
plot(abs(Convert2Complex(x_test)),abs(Convert2Complex(x_test)),'-b')
plot(abs(Convert2Complex(x_test)),abs(Convert2Complex(y_test)),'-k')
hold off
xlabel('Data Input')
legend('Compensated PA Output(w)','Data Input(x)','PA Output(y)')
figure(2);
plot_spec(Convert2Complex(y_test))
hold on
plot_spec(Convert2Complex(x_test))
plot_spec(output);
hold off
xlabel('Frequency ( in MHz)')
ylabel('Magnitude in dB')
legend('Output of PA','Input Data','Output after PD and PA')

end

function output = Convert2Feature(input)
  size = length(input);
  output = zeros(size,2);
  for i = 1:size
    output(i,1) = real(input(i));
    output(i,2) = imag(input(i));
  end
end

function output = Convert2Complex(input)
  shape = size(input);
  len = shape(1);
  output = zeros(1,len);
  for i = 1:len
    output(i) = input(i,1)+1j*input(i,2);
  end
end
 
function output = Non_Linearity(input)
  shape = size(input);
  len = shape(1);
  input1 = zeros(1,len);
  for i = 1:len
    input1(i) = input(i,1)+1j*input(i,2);
  end
  output1 = pa(input1);
  output = Convert2Feature(output1);
end

function output = pa(input)
      time = linspace(0,1,length(input));
      input1.time=time';
      input1.signals.values=input';
      Ts1 = 1/(length(input)-1);
      assignin('base', 'input1',input1);
      assignin('base', 'Ts1',Ts1);
      sim('demosim1.slx');
      assignin('base','output1',output1);
      output=(output1.signals.values)';
end       
 
function [x_train,y_train] = Generate_Data()
	x_train1 = rand(1,20002)*2.1-1.05;
    x_train = reshape(x_train1,[],2);
	y_train = Non_Linearity(x_train);
end

function [x_test,y_test] = load_data(x)
    
    y_test1 = pa(x((length(x)-5000):end));
    y_test = Convert2Feature(y_test1);
    x_test = Convert2Feature(x((length(x)-5000):end));
end 

function plot_spec(spec_input)    
    % Spectral analyses of PA input and PA output
    Fs = 7.86e6;
    N = min([1024,length(spec_input)]);
    pwin = hann(N);
    f = linspace(-3e6,3e6,1000000);
    [spec_input_ps,f_axis] = pwelch(spec_input,pwin,[],f,Fs);
    spec_input_ps_dB = zeros(1,length(spec_input_ps));
    for i =  1:length(spec_input_ps)
      spec_input_ps_dB(i) =  10*log10(abs(spec_input_ps(i)));
    end
    % Plots
    for i = 1:length(f_axis)
      f_axis(i) = f_axis(i)/1000000 ;
    end
    plot(f_axis,spec_input_ps_dB)
end
