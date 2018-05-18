%% Load initial condition from disk

g           = 127267.200000000;	% Gravity, 9.82*(3.6)^2*1000 in [km / hr^2]
Size        = 500;              % Size of map, Size*Size [km]
nx          = 2001;             % Number of cells in each direction on the grid
Tend        = 0.2;             % Simulation time in hours [hr]
dx          = Size/nx;          % Grid spacening

fname   = "Solution_nx"+num2str(nx)+"_500km_T0.2_h.bin";
path    = "output/";
fmatlab = path + "Matlab_" + fname;
fcpp    = path + "Cpp_"    + fname;
fcuda   = path + "CUDA_"   + fname;

filename = ['data/Data_nx',num2str(nx),'_',num2str(Size),'km_T',num2str(Tend)];
Hinit    = fread(fopen([filename,'_h.bin'],'r'),[nx,nx],'double');
Hmatlab  = fread(fopen(fmatlab,'r'),[nx,nx],'double');
Hcpp     = fread(fopen(fcpp,'r'),   [nx,nx],'double');
Hcuda    = fread(fopen(fcuda,'r'),  [nx,nx],'double');

%% Plot error
Err_cuda_cpp    = Hcuda-Hcpp;
Err_matlab_cpp  = Hmatlab-Hcpp;
Err_cuda_matlab = Hcuda-Hmatlab;
figure
surf(Err_cuda_cpp,'EdgeColor','none');
title("Error between CUDA and C++")
xlabel('x')
ylabel('y')

figure
surf(Err_matlab_cpp,'EdgeColor','none')
title("Error between Matlab and C++")
xlabel('x')
ylabel('y')

figure
surf(Err_cuda_matlab,'EdgeColor','none')
title("Error between Matlab and CUDA")
xlabel('x')
ylabel('y')

%% Dt historic analysis
%Load dt_arrays
Nmax = 100;
fname   = "dt_nx"+num2str(nx)+"_500km_T0.2_h.bin";
path    = "output/";
fmatlab_dt = path + "Matlab_" + fname;
fcpp_dt    = path + "Cpp_" + fname;
fcuda_dt    = path + "CUDA_" + fname;
dt_matlab = fread(fopen(fmatlab_dt,'r'),Nmax,'double');
dt_cpp = fread(fopen(fcpp_dt,'r'),Nmax,'double');
dt_cuda = fread(fopen(fcuda_dt,'r'),Nmax,'double');
% 
figure
subplot(221)
hold on;
plot(1:5:Nmax,dt_matlab(1:5:end),'x')
plot(1:5:Nmax,dt_cpp(1:5:end),'o')
plot(1:5:Nmax,dt_cuda(1:5:end),'g+')

legend(["Matlab","cpp","CUDA"])
title("dt evolution")
grid on

subplot(222)
plot(1:5:Nmax, dt_cpp(1:5:end)-dt_matlab(1:5:end),'x--')
title("dt error C++ Matlab")
grid on

subplot(223)
plot(1:5:Nmax, dt_cpp(1:5:end)-dt_cuda(1:5:end),'x--')
title("dt error C++ CUDA")
grid on

subplot(224)
plot(1:5:Nmax, dt_cuda(1:5:end)-dt_matlab(1:5:end),'x--')
title("dt error CUDA Matlab")
grid on

