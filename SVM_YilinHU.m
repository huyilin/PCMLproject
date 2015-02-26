%% Initialization
clear;
clc;
format long
load mp_4-9_data.mat;
AllIndices =1:6000;
Indices = randperm(6000);
InitXtrain=Xtrain(Indices,:);
InitYtrain=Ytrain(Indices);
maximum = max(max(InitXtrain));
minimum = min(min(InitXtrain));
TrainSet=1/(maximum - minimum).*(InitXtrain-minimum.*(ones(size(InitXtrain)))); %normailization
maximum=max(max(Xtest));
minimim=min(min(Ytest));
Xtest=1/(maximum - minimum).*(Xtest-minimum.*(ones(size(Xtest))));
%% Parameters 
%M=10;   % Parameters 
C_test=4;%  Parameters
tau_test=0.1; %parameters
M=10;
%C_test=0.5*[1 2 4 8 16 32 64 128 256 512];
%tau_test=0.05*[1 2 4 8 16 32 64 128 256 512];
TrainSize=size(TrainSet,1);
Xtrainsize=TrainSize*(1-1/M);  
%C_error=zeros(size(C_test,2),size(tau,2));
error_save=zeros(size(tau_test,2),size(C_test,2));
final_sign=0;
for tau_loop=1:size(tau_test,2)+1
    if tau_loop == size(tau_test,2)+1
        final_sign=1;
        %C=C_test(find(C_error==min(C_error)));
        %C=C(1);
        para_best=find(error_save==min(min(error_save)));
        para_best=para_best(1);
        C_best=mod(para_best,size(tau_test,2)); 
        if C_best==0
             C_best=size(C_test,2);
        end 
           tau_best=(para_best-C_best)/10+1;
        if tau_best==0
             tau_best=size(tau_test,2);
        end
         C=C_test(C_best);
         tau=tau_test(tau_best);
    else
        tau=tau_test(tau_loop);
    end 
    %% Computaion of Kernel
    K1=sum(TrainSet.*TrainSet,2)*ones(1,TrainSize); 
    K2=-TrainSet*TrainSet';
    K_all=exp(-(tau)*((1/2).*K1+K2+(1/2).*K1'));
for loop=1:size(C_test,2)       %   Cross validation to optimize parameters
    if ~final_sign
          C=C_test(loop);
    end
    error=zeros(M,1); 
    for k=1:M
        alpha=zeros(Xtrainsize,1);
        valid_indices = (TrainSize/M*(k-1)+1:(TrainSize/M*k));
        train_indices = setdiff(AllIndices, valid_indices);    
        if final_sign  %Using the last time loop to learn alpha one the whole Dataset
            train_indices=1:TrainSize;
            valid_indices=1:TrainSize;
            alpha=zeros(TrainSize,1);
            criterion=0;
            convergence=[0;0];
            iteration=1;
        end 
        K=K_all(train_indices,train_indices);
       %% SMO Algorithm
        Xtrainset=TrainSet(train_indices,:);
        Ytrainset=InitYtrain(train_indices);
        tau1=1e-8;
        I0=find(alpha>0&alpha<C);
        Iplus=find((Ytrainset==1&alpha==0)|(Ytrainset==-1&alpha==C));
        Iminor=find((Ytrainset==-1&alpha==0)|(Ytrainset==1&alpha==C));
        Iup=[I0;Iplus];
        Ilow=[I0;Iminor];
        f=-Ytrainset; %
        i=0;j=0;       
        i_memo=-1;j_memo=-1;
      while(true)    
        %% Compute index Set Ilow and Iup;
        b_up=min(f(Iup));
        i_up_indices=find(f(Iup)==b_up);
        i_up=Iup(i_up_indices);
        i_up=i_up(1);         %If there are multiple i_up, choose the first one
        b_low=max(f(Ilow));
        i_low_indices=find(f(Ilow)==b_low); % Find the most violated one
        i_low=Ilow(i_low_indices);
        i_low=i_low(1);
       %% Prevent Deadlock
        if i_low==i_memo && i_up==j_memo  % Preventing Index swing in two points.
            i=randi(Xtrainsize);
            j=randi(Xtrainsize);            
        elseif i_low==i && i_up==j  % If the index is equal to last one, find the second violated one.
            i_memo=i;
            j_memo=j;
            Iup_temp=setdiff(Iup,i_up);
            Ilow_temp=setdiff(Ilow,i_low);
            b_up=min(f(Iup_temp));          
            i_up_indices=find(f(Iup)==b_up);
            i_up=Iup(i_up_indices);
            j=i_up(1);         %If there are multiple i_up, choose the first one
            b_low=max(f(Ilow_temp));
            i_low_indices=find(f(Ilow)==b_low); % Find the most violated one
            i_low=Ilow(i_low_indices);
            i=i_low(1);
        else
            i=i_low;
            j=i_up;
        end     
        %% Compute the Criterion for training
         if final_sign
             if mod(iteration,20)==0
            criterion=[criterion,(1/2).*sum(sum((alpha*alpha').*K.*(Ytrainset*Ytrainset')))-sum(alpha)];    
            convergence=[convergence,[b_up;b_low]];
             end
            iteration=iteration+1;
         end        
       %% Check for optimality 
        if(f(i_low)<= (f(i_up)+2*tau1))    
            break;
        end
        alpha_j_previous = alpha(j);
        alpha_i_previous = alpha(i);
        sigma=Ytrainset(i)*Ytrainset(j);
        w=alpha_i_previous + sigma*alpha_j_previous;
        L=max(0,sigma*w-(sigma==1)*C);
        H=min(C,sigma*w+(sigma==-1)*C);
        eta=K(i,i)+K(j,j)-2*K(i,j);
       %% Update the Alpha
        if eta > 1e-15
             alpha_j_unc = alpha_j_previous + (Ytrainset(j)*(f(i)-f(j)))/eta;
             if alpha_j_unc<L
                alpha(j)=L;
             elseif alpha_j_unc>H
                alpha(j)=H;
             else 
                 alpha(j)=alpha_j_unc;
             end 
        else 
            L_i=w-sigma*L;
            H_i=w-sigma*H;
            v_i=f(i)+Ytrainset(i)-alpha_i_previous*Ytrainset(i)*K(i,i)-alpha_j_previous*Ytrainset(j)*K(i,j);    
            v_j=f(j)+Ytrainset(j)-alpha_i_previous*Ytrainset(i)*K(i,j)-alpha_j_previous*Ytrainset(j)*K(j,j);
            Phi_L=(1/2)*(K(i,i)*L_i*L_i+K(j,j)*L*L)+sigma*K(i,j)*L_i*L+Ytrainset(i)*L_i*v_i+Ytrainset(j)*L*v_j-L_i-L;
            Phi_H=(1/2)*(K(i,i)*H_i*H_i+K(j,j)*H*H)+sigma*K(i,j)*H_i*H+Ytrainset(i)*H_i*v_i+Ytrainset(j)*H*v_j-H_i-H;
            if Phi_L>Phi_H
                   alpha(j)=H;
            else 
                   alpha(j)=L;
            end
        end 
        alpha(i) = alpha_i_previous + sigma*(alpha_j_previous-alpha(j));
        f=f+Ytrainset(i)*(alpha(i)-alpha_i_previous)*K(:,i)+Ytrainset(j)*(alpha(j)-alpha_j_previous)*K(:,j);
        I0=find(alpha>0&alpha<C);
        Iplus=find((Ytrainset==1&alpha==0)|(Ytrainset==-1&alpha==C));
        Iminor=find((Ytrainset==-1&alpha==0)|(Ytrainset==1&alpha==C));
        Iup=[I0;Iplus];
        Ilow=[I0;Iminor];
        %fprintf('i j  %d %d \n ',i,j);
      end 
      %% Update b   
      I_zero=find(alpha>0&alpha<C);
      size_S=size(I_zero,1);
      K_b=K(I_zero,:);
      y_b= K_b*(alpha.*Ytrainset);
      b=(1/size_S).*sum(y_b); 
      %% Cross Validation 
      K_Valid=K_all(valid_indices,train_indices);
      Yvalidset = InitYtrain(valid_indices);
      y= K_Valid*(alpha.*Ytrainset)+b.*ones(TrainSize/(1+(M-1)*(~final_sign)),1);
      error(k)=sum(Yvalidset.*y<0)/size(Yvalidset,1);
      %fprintf('error %d \n',error(k));
      if final_sign
          break;
      end       
    end
    if final_sign
         TrainingError=error(1)
         break;
    end     
    Rcv=sum(error)./M;
    fprintf('Rcv %d \n',Rcv)
    %C_error(loop)=Rcv;  
    error_save(loop,tau_loop)=Rcv;
end 
end
    K=sum(Xtest.*Xtest,2)*ones(1,TrainSize); % Computation of test error
    K=ones(size(Xtest,1),1)*sum(TrainSet.*TrainSet,2)'+K;
    K=-2.*Xtest*TrainSet'+K;
    K=exp(-(tau/2)*(K));
    y=K*(alpha.*Ytrainset)+b.*ones(size(Xtest,1),1);
    Testerror=sum(Ytest.*y<0)/size(Ytest,1)
%    figure
%    plot(C_test,error_save,'lineWidth',2, 'Marker','*')
%    figure
%    imagesc(error_save);
    figure
    plot(criterion)
    figure
    plot(convergence(1,:),'lineWidth',2,'Color','b');
    hold on 
    plot(convergence(2,:),'lineWidth',2,'Color','r');
    figure
    semilogy(convergence(2,:)-convergence(1,:));
    
    
        
        
        