function mem_pa_dB
%loading data from the file

fileName1 = '/home/apuvi/qpsk_I_seqLen65472.mat';
loadfile1 = load(fileName1);
table1 = struct2cell(loadfile1);
data_total_I = cell2mat(table1);

fileName2 = '/home/apuvi/qpsk_Q_seqLen65472.mat';
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
scale_val = max(abs(data_total))
data_total = data_total/scale_val;
cdf_point = 0.884282;
dB_point = 0.65163;
%cdf_point = 1;
%dB_point = 1;
data_total = (data_total*dB_point)/cdf_point; 
fprintf('Input scaling done\n');
[x_train,y_train] =  Generate_Data(data_total); %y_train is vector
gamma = max(abs(Convert2Complex(y_train)));
y_train = (y_train)/gamma;
%gamma = sqrt(sum(abs(Convert2Complex(y_train)).^2)/sum(abs(Convert2Complex(x_train)).^2));
%gamma_phase = exp(-1*1j*atan(dot(Convert2Complex(x_train),conj(Convert2Complex(y_train)))));
%y_train = Convert2Feature((Convert2Complex(y_train)*gamma_phase)/gamma);
%{Lambda =  x - 0.1*K.sum(x*x,axis =1,keepdims=True)*x))

layers = [ ...
    sequenceInputLayer(2)
    lstmLayer(12,'OutputMode','sequence')
    fullyConnectedLayer(7)
    reluLayer
    fullyConnectedLayer(5)
    reluLayer
    fullyConnectedLayer(2)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',6001, ...
    'MiniBatchSize', 100, ...
    'InitialLearnRate', 0.005, ...
    'ExecutionEnvironment',"auto",...
    'plots','training-progress', ...
    'Verbose',false);

net = trainNetwork(y_train',x_train',layers,options)

[x_test,y_test] =  load_data(data_total);
delta = max(abs(Convert2Complex(y_test)));
y_test = (y_test)/delta;
classes = predict(net,x_test','MiniBatchSize',100);
classes = classes';
output = Convert2Feature(pa(Convert2Complex(classes)));
beta = max(abs(Convert2Complex(output)));
output = (output)/beta;
%beta = sqrt(sum(abs(Convert2Complex(output)).^2)/sum(abs(Convert2Complex(x_test)).^2));
%beta_phase = exp(-1*1j*atan(dot(Convert2Complex(x_test),conj(Convert2Complex(output)))));
%output = Convert2Feature((Convert2Complex(output)*beta_phase)/beta);
figure(1);
scatter(20*log10(abs(Convert2Complex(x_test))),20*log10(abs(Convert2Complex(output)))-2.7,2,'k')
hold on 
scatter(20*log10(abs(Convert2Complex(x_test))),20*log10(abs(Convert2Complex(x_test))),2,'b')
scatter(20*log10(abs(Convert2Complex(x_test))),20*log10(abs(Convert2Complex(y_test)))-2.7,2,'r')
hold off
xlabel('Data Input')
legend('Compensated PA Output(w)','Data Input(x)','PA Output(y)')
figure(2);
plot_spec(Convert2Complex(y_test),'r')
hold on
plot_spec(Convert2Complex(x_test),'b')
plot_spec(Convert2Complex(output),'k');
hold off
xlabel('Frequency ( in MHz)')
ylabel('Magnitude in dB')
legend('Output of PA','Input Data','Output after PD and PA')
figure(3);
scatter(20*log10(abs(Convert2Complex(x_test))),angle(Convert2Complex(y_test)./Convert2Complex(x_test)),2,'r')
hold on
scatter(20*log10(abs(Convert2Complex(x_test))),angle(Convert2Complex(output)./Convert2Complex(x_test)),2,'k')
hold off
ylabel('Phase difference in radians')
xlabel('Input Data')
legend('Arg(PA Output(y))-Arg(Input data(x))','Arg(Compensated PA Output(y))-Arg(Input data(x))')
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

%LTI filter
function Y = lti(X,num,denom)
    Y = filter(num,denom,X);
end

function output = mem_poly(input,C)
      time = linspace(0,1,length(input));
      input1.time=time';
      input1.signals.values=input';
      Ts1 = 1/(length(input)-1);
      assignin('base', 'input1',input1);
      assignin('base', 'Ts1',Ts1);
      assignin('base', 'C',C);
      sim('demosim1.slx');
      assignin('base','output1',output1);
      output=(output1.signals.values)';
end

function output = pa(x)
    %v_test1 = lti(x,[1],[1]);
    %v_test2 = lti(x,[1 0.3],[1 -0.1]);
    %v_test3 = lti(x,[1 -0.2],[1 -0.4]);
    %y_test1 = mem_poly(v_test1,[0.5054+0.0429j 0 0.04395-0.07915j 0 -0.5496-0.44455j]);
    %y_test2 = mem_poly(v_test2,[0.05895+0.0002j 0 -0.0909+0.01955j 0 0.0842+0.0017j]);
    %y_test3 = mem_poly(v_test3,[0.02365-0.0029j 0 0.01975+0.01415j 0 -0.05075-0.0098j]);  
    output = mem_poly(x,[0.52565+0.0452j 0 -0.0271-0.145j 0 -0.48285-0.3514j;-0.034-0.00115j 0 0.1117+0.11585j 0 -0.12255-0.18675j;0.01445-0.0027j 0 -0.03105-0.0466j 0 0.06145+0.0754j]);
    %output = y_test1;
end   
 
function [x_train,y_train] = Generate_Data(data)
    %x_t = [];
    %for i = 1:8000
    %    if (20*log10(abs(data(i))) <= -2.31 && length(x_t) < 4000)
    %        x_t = [x_t,data(i)];
    %    end
    %end
    %length(x_t)
    x_t = data(1:2501);
    x_train = Convert2Feature(x_t);
	y_train = Non_Linearity(x_train);
end

function [x_test,y_test] = load_data(x) 
    %x_te = [];
    %for i = (length(x)-8000):length(x)
    %    if (20*log10(abs(x(i))) <= -2.31 && length(x_te) < 4001)
    %        x_te = [x_te,x(i)];
    %    end
    %end
    %length(x_te)c
    x_te = x((length(x)-2500):end);
    x_test = Convert2Feature(x_te);    
    y_test1 = pa(x_te);
    y_test = Convert2Feature(y_test1);
end 

function plot_spec(spec_input,c)    
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
    plot(f_axis,spec_input_ps_dB,c)
end