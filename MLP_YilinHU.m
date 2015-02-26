%function[TrainingData,ValidData] =SplitData()
%Split the Data into Training set and validation set.

clear; 
clc;
load mp_3-5_data.mat;
InitXTrain=Xtrain; %
InitYTrain=Ytrain;

NumInput=size(InitXTrain,1);
R=randperm(NumInput);
TrainIndices=R(1:NumInput*(2/3));% Sample the data randomly(Number=40)
ValidIndices=R(NumInput*(2/3)+1:NumInput*(3/3)); %Number=20

TrainSet=InitXTrain(TrainIndices,:);%Sample the data ramdomly 
ValidSet=InitXTrain(ValidIndices,:);

%TrainSet=imresize(TrainSet,[NumInput*(2/300),64]);   %imresize the size of images to 8*8
%ValidSet=imresize(ValidSet,[NumInput*(1/300),64]);

TrainYSet=InitYTrain(TrainIndices);  %Sample the corresponding Y data
ValidYSet=InitYTrain(ValidIndices);

%K=size(TrainingSet);
%K2=size(ValidSet); 

%Preprocess
Maximum1=max(max(TrainSet));
Minimum1=min(min(TrainSet));
TrainSet=(TrainSet-ones(size(TrainSet)).*Minimum1)./(Maximum1-Minimum1); %Training Data
Maximum2=max(max(ValidSet));
Minimum2=min(min(ValidSet));  %!!!!!可能不用分开
ValidSet=(ValidSet-ones(size(ValidSet)).*Minimum1)./(Maximum1-Minimum1);
%Initialization
TrainSize=size(TrainSet);
ValidSize=size(ValidSet);
H1= 16;    %%(1/2)*TrainSize(2);                         %the number of hidden layers
h1=H1/2;
W1=normrnd(0,sqrt(1/TrainSize(2)),H1,TrainSize(2));   %Initialize hidden layer parameters.
W2=normrnd(0,sqrt(1/h1),1,h1);
Bias1=normrnd(0,sqrt(1/TrainSize(2)),H1,1);
Bias2=normrnd(0,sqrt(1/TrainSize(2)),1,1);
r1=zeros(H1,1);
%Bias1=ones(H1,1);
%Bias2=1;
%delta_W1=zeros(H1,TrainSize(2));

tri_W1=zeros(H1,TrainSize(2));
tri_W2=zeros(1,h1);
tri_Bias1=zeros(H1,1);
tri_Bias2=0;
full_loop=100; % upbounded full loop number

%Logistic_training_error=0;
Logistic_training_error_all=zeros(1,full_loop);
%Logistic_validation_error=0;
Logistic_validation_error_all=zeros(1,full_loop);
%Iteration ramdomly one datapoints.

for loop=1 : full_loop;
    rate=0.01;
    mu=0.5;
    IterIndex=randperm(TrainSize(1));

  for i=1:TrainSize(1)
      
     x=TrainSet(IterIndex(i),:)';
     Activ1=W1*x+Bias1;
     Activ1odd=Activ1(1:2:end);
     Activ1even=Activ1(2:2:end);
     z=Activ1odd.*(1./(1+exp(-Activ1even)));
     Activ2=W2*z+Bias2;
     t=TrainYSet(IterIndex(i));
     r2=(-t)./(1+exp(t*Activ2));

     r1(1:2:H1)=r2*W2'.*(1./(1+exp(-Activ1(2:2:end))));
     r1(2:2:H1)=r2*W2'.*Activ1(1:2:end).*(1./(1+exp(-Activ1(2:2:end)))).*(1-(1./(1+exp(-Activ1(2:2:end)))));
     delta_W2=r2*z';
     delta_b2=r2;
     delta_W1=r1*x';
     delta_b1=r1; 
     tri_W2 = ( -rate*(1-mu)).*delta_W2+mu.*tri_W2;
     W2 = W2 + tri_W2;
     tri_W1 =  (-rate*(1-mu)).*delta_W1+mu.*tri_W1;
     W1 = W1 + tri_W1;
     tri_Bias1 = ( -rate*(1-mu)).*delta_b1+mu.*tri_Bias1;
     tri_Bias2 = ( -rate*(1-mu)).*delta_b2+mu.*tri_Bias2;
     Bias1 = Bias1 + tri_Bias1;
     Bias2 = Bias2 + tri_Bias2;
  end
  % Compute the logistical error one all Trainiing data points 
  Activ1_T=W1*(TrainSet')+Bias1*ones(1,TrainSize(1));   % size: H1*n
  Activ1odd_T=Activ1_T(1:2:end,:);   % size(h1*n)
  Activ1even_T=Activ1_T(2:2:end,:); % size(h1*n)
  z=Activ1odd_T.*(1./(1+exp(-Activ1even_T))); %size(h1*n)
  Activ2_T=W2*z+Bias2*ones(1,TrainSize(1)); %size1*n
   
   tmp = -TrainYSet .* Activ2_T';
   
   positive_index = find(tmp>=0);
   tmp_pos = tmp(positive_index, :);
   negative_index = find(tmp<0);
   tmp_neg = tmp(negative_index, :);
   error_pos = log(1 + exp(-tmp_pos)) + tmp_pos;
   error_neg = log1p(exp(tmp_neg));
   Logistic_training_error_all(loop)= (sum(error_pos)+sum(error_neg))/4000;

  
  % Compute the logistical error one all Validation data points 
   Activ1_V=W1*(ValidSet')+Bias1*ones(1,ValidSize(1));   % size: H1*n
   Activ1odd_V=Activ1_V(1:2:end,:);   % size(h1*n)
   Activ1even_V=Activ1_V(2:2:end,:); % size(h1*n)
   z=Activ1odd_V.*(1./(1+exp(-Activ1even_V))); %size(h1*n)
   Activ2_V=W2*z+Bias2*ones(1,ValidSize(1)); %size1*n
   tmp = -ValidYSet .* Activ2_V';
   positive_index = find(tmp>=0);
   tmp_pos = tmp(positive_index, :);
   negative_index = find(tmp<0);
   tmp_neg = tmp(negative_index, :);
   error_pos = log(1 + exp(-tmp_pos)) + tmp_pos;
   error_neg = log1p(exp(tmp_neg));
   Logistic_validation_error_all(loop)= (sum(error_pos)+sum(error_neg))/2000;
 
   tmp=size(find(ValidYSet.*Activ2_V'<=0))./2000;
   zero_error(loop)= tmp(1);

    fprintf('In the %d full loop \n',loop);
    fprintf('training error %d \n ',Logistic_training_error_all(loop));
    fprintf('validation error %d \n',Logistic_validation_error_all(loop));
end 

 loop=1:full_loop;
 figure(1)
 plot(loop,Logistic_training_error_all,'r' ,loop,Logistic_validation_error_all,'b'), title(' H=50 '),legend('training error','validation error'),axis([0,50,0,0.3]);
 figure(2)
 plot(loop,zero_error), title(' Zero/one error of validation set'),axis([0,50,0,0.5]);
%% compute test logistic error 
 Maximum=max(max(Xtest));
 Minimum=min(min(Xtest));
 TestSet=(Xtest-ones(size(Xtest)).*Minimum)./(Maximum-Minimum);
  Activ1_Te=W1*(TestSet')+Bias1*ones(1,1902);   % size: H1*n
  Activ1odd_Te=Activ1_Te(1:2:end,:);   % size(h1*n)
  Activ1even_Te=Activ1_Te(2:2:end,:); % size(h1*n)
  z=Activ1odd_Te.*(1./(1+exp(-Activ1even_Te))); %size(h1*n)
  Activ2_Te=W2*z+Bias2*ones(1,1902); %size1*n
  Logistic_TEST_error_all=mean(log(1+exp((-Ytest').*Activ2_Te)));
  std= std(log(1+exp((-Ytest').*Activ2_Te)));

%end 
     
     