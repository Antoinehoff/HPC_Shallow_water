%% Performance measurement 
% Some manually measured performances
Nmax = [250 250 250 250 250];

m_time = [173 188 201 168 151 175];
m_flops = [0.52 0.47 0.44 0.53 0.59 0.51];
m_mean_flops = mean(m_flops);
m_std_flops  = std(m_flops);
m_mean_time = mean(m_time);
m_std_time  = std(m_time);

cpp_time = [67 67 68 69 66 66];
cpp_flops = [1.33 1.33 1.31 1.29 1.35 1.35];

cpp_mean_flops = mean(cpp_flops);
cpp_std_flops  = std(cpp_flops);
cpp_mean_time = mean(cpp_time);
cpp_std_time  = std(cpp_time);

figure
hold on
errorbar(1:6,m_time,ones(1,6)*m_std_time,'xb')
plot(0:7,ones(1,8)*m_mean_time,'--b')
errorbar(1:6,cpp_time,ones(1,6)*cpp_std_time,'xr')
plot(0:7,ones(1,8)*cpp_mean_time,'--r')
title("Computation time comparison")
legend("Matlab code","","C++ sequ. code","")
xlabel("Measurements")
ylabel("Time in seconds")
grid on