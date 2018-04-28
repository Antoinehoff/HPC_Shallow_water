%% Load initial condition from disk

g           = 127267.200000000;	% Gravity, 9.82*(3.6)^2*1000 in [km / hr^2]
Size        = 500;              % Size of map, Size*Size [km]
nx          = 2001;             % Number of cells in each direction on the grid
Tend        = 0.2;             % Simulation time in hours [hr]
dx          = Size/nx;          % Grid spacening

% Set filename
filename = ['Data_nx',num2str(nx),'_',num2str(Size),'km_T',num2str(Tend)]

% Load initial condition from data files
H = fread(fopen([filename,'_h.bin'],'r'),[nx,nx],'double');

H(1:10,1:10)

%%

N = 36;
nx = 6;

B = reshape([0:N-1],nx,nx);
B
B(1,:) = B(2,:);
B
B(end,:) = B(end-1,:);
B
B(:,1) = B(:,2);
B
B(:,end) = B(:,end-1);
B