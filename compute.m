
%% Add data path and start timer
clc; addpath('data/'); tic;

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
HU = fread(fopen([filename,'_hu.bin'],'r'),[nx,nx],'double');
HV = fread(fopen([filename,'_hv.bin'],'r'),[nx,nx],'double');

% Load topography slopes from data files
Zdx = fread(fopen([filename,'_Zdx.bin'],'r'),[nx,nx],'double');
Zdy = fread(fopen([filename,'_Zdy.bin'],'r'),[nx,nx],'double');

% Allocate memory for computing
Ht = zeros(nx,nx);
HUt = zeros(nx,nx);
HVt = zeros(nx,nx);

%save dts
Nmax = 250;
dt_array = zeros(1,Nmax);

%% Compute all time-steps
T = 0; nt = 0;
while nt < Nmax;
    
    %% Compute the time-step length
    
    mu = sqrt( max(abs(HU./H-sqrt(H*g)),abs(HU./H+sqrt(H*g))).^2 + max(abs(HV./H-sqrt(H*g)),abs(HV./H+sqrt(H*g))).^2 );
    dt = dx/(sqrt(2)*max(mu(:)));
    if T+dt > Tend
        dt = Tend-T;
    end
    
    %% Print status
    
    disp(['Computing T: ',num2str(T+dt),'. ',num2str(100*(T+dt)/Tend),'%'])
    
    %% Copy solution to temp storage and enforce boundary condition
    
    Ht = H;
    HUt = HU;
    HVt = HV;
    
    Ht(1,:) = Ht(2,:);
    Ht(end,:) = Ht(end-1,:);
    Ht(:,1) = Ht(:,2);
    Ht(:,end) = Ht(:,end-1);
    
    HUt(1,:) = HUt(2,:);
    HUt(end,:) = HUt(end-1,:);
    HUt(:,1) = HUt(:,2);
    HUt(:,end) = HUt(:,end-1);
    
    HVt(1,:) = HVt(2,:);
    HVt(end,:) = HVt(end-1,:);
    HVt(:,1) = HVt(:,2);
    HVt(:,end) = HVt(:,end-1);
    
    %% Compute a time-step
    
    C = (0.5*dt/dx);
    
    H(2:nx-1,2:nx-1,1) = 0.25*(Ht(2:nx-1,1:nx-2)+Ht(2:nx-1,3:nx)+Ht(1:nx-2,2:nx-1)+Ht(3:nx,2:nx-1)) ...
        + C*( HUt(2:nx-1,1:nx-2) - HUt(2:nx-1,3:nx) + HVt(1:nx-2,2:nx-1) - HVt(3:nx,2:nx-1) );
    
    HU(2:nx-1,2:nx-1) = 0.25*(HUt(2:nx-1,1:nx-2)+HUt(2:nx-1,3:nx)+HUt(1:nx-2,2:nx-1)+HUt(3:nx,2:nx-1)) - dt*g*H(2:nx-1,2:nx-1).*Zdx(2:nx-1,2:nx-1) ...
        + C*( HUt(2:nx-1,1:nx-2).^2./Ht(2:nx-1,1:nx-2) + 0.5*g*Ht(2:nx-1,1:nx-2).^2 - HUt(2:nx-1,3:nx).^2./Ht(2:nx-1,3:nx) - 0.5*g*Ht(2:nx-1,3:nx).^2 ) ...
        + C*( HUt(1:nx-2,2:nx-1).*HVt(1:nx-2,2:nx-1)./Ht(1:nx-2,2:nx-1) - HUt(3:nx,2:nx-1).*HVt(3:nx,2:nx-1)./Ht(3:nx,2:nx-1) );
    
    HV(2:nx-1,2:nx-1) = 0.25*(HVt(2:nx-1,1:nx-2)+HVt(2:nx-1,3:nx)+HVt(1:nx-2,2:nx-1)+HVt(3:nx,2:nx-1)) - dt*g*H(2:nx-1,2:nx-1).*Zdy(2:nx-1,2:nx-1)  ...
        + C*( HUt(2:nx-1,1:nx-2).*HVt(2:nx-1,1:nx-2)./Ht(2:nx-1,1:nx-2) - HUt(2:nx-1,3:nx).*HVt(2:nx-1,3:nx)./Ht(2:nx-1,3:nx) ) ...
        + C*( HVt(1:nx-2,2:nx-1).^2./Ht(1:nx-2,2:nx-1) + 0.5*g*Ht(1:nx-2,2:nx-1).^2 - HVt(3:nx,2:nx-1).^2./Ht(3:nx,2:nx-1) - 0.5*g*Ht(3:nx,2:nx-1).^2  );
    
    %% Impose tolerances
    
    Ht(Ht<0) = 0.00001;
    HUt(Ht<=0.0001) = 0;
    HVt(Ht<=0.0001) = 0;
    
    %% Update time T
    
    T = T + dt;
    nt = nt + 1;
    dt
    dt_array(nt)=Ht(1);
end

%% Save solution to disk

filename = ['output/Matlab_Solution_nx',num2str(nx),'_',num2str(Size),'km','_T',num2str(Tend),'_h.bin'];
fileID = fopen(filename,'w');
fwrite(fileID,Ht,'double');
fclose(fileID);

filename = ['output/Matlab_dt_nx',num2str(nx),'_',num2str(Size),'km','_T',num2str(Tend),'_h.bin'];
fileID = fopen(filename,'w');
fwrite(fileID,dt_array,'double');
fclose(fileID);


% Stop timer
time = toc;

%% Communicate time-to-compute

ops = nt*( 15 + 2 + 11 + 30 + 30 + 1 )*nx^2;
flops = ops/time;
disp(['Time to compute solution : ',num2str(time),' seconds'])
disp(['Average performance      : ',num2str(flops/1.0e9),' gflops'])
