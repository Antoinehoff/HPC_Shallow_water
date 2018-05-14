% Load initial condition from disk

g           = 127267.200000000;	% Gravity, 9.82*(3.6)^2*1000 in [km / hr^2]
Size        = 500;              % Size of map, Size*Size [km]
nx          = 2001;             % Number of cells in each direction on the grid
Tend        = 0.2;             % Simulation time in hours [hr]
dx          = Size/nx;          % Grid spacening

fname   = "Solution_nx"+num2str(nx)+"_500km_T0.2_h.bin";
path    = "output/";
fmatlab = path + "Matlab_" + fname;
fcpp    = path + "Cpp_" + fname;
%% Solution analysis
% Load initial state and Matlab - Cpp solutions
filename = ['data/Data_nx',num2str(nx),'_',num2str(Size),'km_T',num2str(Tend)]
Hinit = fread(fopen([filename,'_h.bin'],'r'),[nx,nx],'double');
Hmatlab = fread(fopen(fmatlab,'r'),[nx,nx],'double');
Hcpp = fread(fopen(fcpp,'r'),[nx,nx],'double');
H_err = Hmatlab-Hcpp;
% aggreg_err = zeros(1,nx);
% aggreg_sol = zeros(1,nx);
% for i=[1:nx]
%     aggreg_err(i) = sum(H_err(i,:));
%     aggreg_sol(i) = sum(Hmatlab(i,:));
% end
% 
% figure
% hold on;
% yyaxis left
% plot(0:nx-1,aggreg_err)
% yyaxis right
% plot(0:nx-1,aggreg_sol)
% title("1D Aggregated error")
% grid on
% legend("Aggregated error","Aggregated solution")
% xlim([0,nx-1])

figure
contourf(H_err)
title("Error on the last step (nt = 250)")
xlabel('x')
ylabel('y')

%% Dt historic analysis
% Load dt_arrays
Nmax = 250;
fname   = "dt_nx"+num2str(nx)+"_500km_T0.2_h.bin";
path    = "output/";
fmatlab_dt = path + "Matlab_" + fname
fcpp_dt    = path + "Cpp_" + fname
dt_matlab = fread(fopen(fmatlab_dt,'r'),Nmax,'double');
dt_cpp = fread(fopen(fcpp_dt,'r'),Nmax,'double');

figure
hold on;
plot(1:5:Nmax,dt_matlab(1:5:end),'x')
plot(1:5:Nmax,dt_cpp(1:5:end),'o')
legend(["Matlab","cpp"])
title("dt evolution")
grid on

figure
hold on;
plot(1:Nmax, dt_matlab-dt_cpp)
title("dt error")
grid on