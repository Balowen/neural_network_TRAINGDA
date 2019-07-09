clear;
format compact;
load dorothea2

%przygotowanie zmiennych
x = Pn;
t = T;

%Wyczyszczenie zbêdnych zmiennych
clearvars -except x t;

%Wektor neuronow wartswy 1 i 2
S1_vec = 10:10:70;
S2_vec = 10:10:70;

%Wektory inc/dec learning ratio i b³êdu
lr_inc_vec = 1.05

lr_dec_vec =  0.7

er_vec =  1.04



% Zapis nag³ówka dokumentu
header = 'S1\tS2\tlr_inc\tlr_dec\tmax_err\t  PK[%%]\t  lr\t\tMSE\t\tepoch\twork_progress[%%]\n';
file_var = fopen('nn_logs.txt', 'wt');
fprintf(file_var, header);
formating = '%2g \t %2g \t %1.6g \t %1.6g \t %1.6g \t %3.4g \t %1.6g \t %4.6g \t %3.6g \t %3.6g\n';
fclose(file_var);

%zmienne do zapisu
todo = length(S1_vec)*length(S2_vec)*length(lr_inc_vec)*...
        length(lr_dec_vec)*length(er_vec);    
PK_v=zeros(length(S1_vec),length(S2_vec),length(lr_inc_vec),...
 length(lr_dec_vec),length(er_vec));
MSE_v=PK_v;
LR_v=PK_v;
EPOCH_v=PK_v;
licznik = 0;


for ind_S1=1:length(S1_vec)
    for ind_S2=1:length(S2_vec)
        for ind_lr_inc=1:length(lr_inc_vec)
            for ind_lr_dec=1:length(lr_dec_vec)
                for ind_er=1:length(er_vec)

                    licznik = licznik +1;
                    work_progress = licznik/todo *100;
                    %Utworzenie sieci
                    net = feedforwardnet([S1_vec(ind_S1),S2_vec(ind_S2)], 'traingda');
                    %ustawienie wszelkich parametrów
                    
                    net.trainParam.epochs = 20000;
                    net.trainParam.max_fail = 500;
                    net.trainParam.lr = 0.01;
                    net.trainParam.lr_inc = lr_inc_vec(ind_lr_inc);
                    net.trainParam.lr_dec = lr_dec_vec(ind_lr_dec);
                    net.trainParam.max_perf_inc = er_vec(ind_er);
                    net.trainParam.show = 100;
                    net.trainParam.showWindow = false;
                    net.trainParam.showCommandLine = true;
                    
                    %Podzial danych
                    net.divideParam.trainRatio = 80/100;
                    net.divideParam.valRatio = 10/100;
                    net.divideParam.testRatio = 10/100;

                    %trenowanie sieci
                    [net,tr] = train(net,x,t);
                    
                    %test
                    y = net(x);
                    e = t-y;
                    mse_value = immse(y,t);                    
                    PK = (1-sum(abs(t-y)>=.3)/length(t))*100
                    
                    % Zapis danych do macierzy
                    PK_v(ind_S1,ind_S2,ind_lr_inc,ind_lr_dec,ind_er) = PK;
                    MSE_v(ind_S1,ind_S2,ind_lr_inc,ind_lr_dec,ind_er) = mse_value;
                    LR_v(ind_S1, ind_S2, ind_lr_inc, ind_lr_dec,ind_er) = tr.lr(tr.best_epoch);
                    EPOCH_v(ind_S1, ind_S2, ind_lr_inc, ind_lr_dec, ind_er) = tr.best_epoch;
                     
                    % Zapis danych do pliku
                    file_var = fopen('nn_logs.txt', 'at');
                    fprintf(file_var, formating, S1_vec(ind_S1),...
                            S2_vec(ind_S2), lr_inc_vec(ind_lr_inc),...
                            lr_dec_vec(ind_lr_dec), er_vec(ind_er),...
                            PK, tr.lr(tr.best_epoch), mse_value ,...
                            tr.best_epoch, work_progress);
                    fclose(file_var);
                end
            end
        end
    end
end

save("output");
%Wykresy
figure;
surf(S1_vec,S2_vec,PK_v')
xlabel('S1');
ylabel('S2');
zlabel('PK [%]');

figure;
surf(S1_vec,S2_vec,MSE_v')
xlabel('S1');
ylabel('S2');
zlabel('MSE');
% close all;
% figure('DefaultAxesFontSize',1);
% set(gcf,'position',[100,100,700,576]);
% curr_er_vec=1;
% subplot(2,2,1)
% surf(lr_inc_vec,lr_dec_vec,squeeze(PK_v(1,1,:,:,curr_er_vec))')
% xlabel('lr\_inc'); ylabel('lr\_dec'); zlabel('PK [%]');
% axis([-inf inf -inf inf -inf inf]); axis 'auto z';
% subplot(2,2,2)
% surf(lr_inc_vec,lr_dec_vec,squeeze(MSE_v(1,1,:,:,curr_er_vec))')
% xlabel('lr\_inc'); ylabel('lr\_dec'); zlabel('MSE');
% axis([-inf inf -inf inf -inf inf]); axis 'auto z';


