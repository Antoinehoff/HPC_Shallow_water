% Load initial condition from disk

g           = 127267.200000000;	% Gravity, 9.82*(3.6)^2*1000 in [km / hr^2]
Size        = 500;              % Size of map, Size*Size [km]
nx          = 2001;             % Number of cells in each direction on the grid
Tend        = 0.2;             % Simulation time in hours [hr]
dx          = Size/nx;          % Grid spacening

fname   = "Solution_nx"+num2str(nx)+"_500km_T0.2_h.bin";
path    = "output/";
fmatlab = path + "Matlab_" + fname
fcpp    = path + "Cpp_" + fname
%%
% Load initial state and Matlab - Cpp solutions
filename = ['data/Data_nx',num2str(nx),'_',num2str(Size),'km_T',num2str(Tend)]
Hinit = fread(fopen([filename,'_h.bin'],'r'),[nx,nx],'double');
Hmatlab = fread(fopen(fmatlab,'r'),[nx,nx],'double');
Hcpp = fread(fopen(fcpp,'r'),[nx,nx],'double');
max(max(Hmatlab))
max(max(Hcpp))
norm(Hmatlab-Hcpp)
%%
% Load dt_arrays
Nmax = 250;
fname   = "dt_nx"+num2str(nx)+"_500km_T0.2_h.bin";
path    = "output/";
fmatlab = path + "Matlab_" + fname
fcpp    = path + "Cpp_" + fname
dt_matlab = fread(fopen(fmatlab,'r'),Nmax,'double');
dt_cpp = fread(fopen(fcpp,'r'),Nmax,'double');

figure
hold on;
plot(1:Nmax,dt_matlab)
plot(1:Nmax,dt_cpp,'--')
legend(["Matlab","cpp"])
title("dt evolution")

figure
hold on;
plot(1:Nmax, dt_matlab-dt_cpp)
title("dt error")