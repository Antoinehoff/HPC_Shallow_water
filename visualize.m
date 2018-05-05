
clear all, close all; clc; addpath('data/');

%% Set initial condition

g           = 127267.200000000;	% Gravity, 9.82*(3.6)^2*1000 in [km / hr^2]
Size        = 500;              % Size of map, Size*Size [km]
nx          = 2001;             % Number of cells in each direction on the grid
Tend        = 0.20;             % Simulation time in hours [hr]
dx          = Size/nx;          % Grid spacening
version     = 'Matlab_';

%% Load solution, colormap and Typography
ColorMap	= fread(fopen(['figures/Fig_nx',num2str(nx),'_',num2str(Size),'km_ColorMap.bin'],'r'),[nx,nx],'double');
Typography	= fread(fopen(['figures/Fig_nx',num2str(nx),'_',num2str(Size),'km_Typography.bin'],'r'),[nx,nx],'double');

%% Plot initial condition
filename = ['Data_nx',num2str(nx),'_',num2str(Size),'km_T',num2str(Tend),'_h.bin'];
if exist(filename, 'file') == 2
    H_Ini       = fread(fopen(filename,'r'),[nx,nx],'double');
    PlotH( Size, H_Ini, ColorMap, Typography);
else
    disp(['Did not find: ',filename,'. Check your data paths. ']);
end

%% Plot solution state at Tend
filename = ['output/',version,'Solution_nx',num2str(nx),'_',num2str(Size),'km_T',num2str(Tend),'_h.bin'];
filename = 'output/Cpp_Solution_nx2001_500km_T0.2_h.bin';
if exist(filename, 'file') == 2
    H_Sol       = fread(fopen(filename,'r'),[nx,nx],'double');
    PlotH( Size, H_Sol, ColorMap, Typography);
else
    disp(['Did not find: ',filename,'. Did you remember to run compute.m ?']);
end

%% Function to generate typography and surface plot
function [ EarthFigureHandle ] = PlotH( Size , H , ColorMap , Typography )

Zspan       = [0,1];            % Maximum and minimum vertical height to indicate on figures [km]
Zfac        = 10;               % Magnification of vertical axis for visualization purposes
Cfac        = 40;               % Highlight waves with red-black colors
nx          = size(H,1);        % Number of cells in each direction

% Create figure
EarthFigureHandle = figure();

% Set size and position of figure
set(EarthFigureHandle,'units', 'inches', 'pos', [0 0 16 9],'color','w');

% Get X and Y mesh grid for surface plots
[X,Y] = meshgrid(linspace(0,Size,nx),linspace(0,Size,nx));

% Plot typography using colormap
surf(X,flipud(Y),Zfac*Typography,ColorMap,...
    'FaceColor','interp','EdgeColor','none','FaceLighting','phong');

% Hold figures
hold on;

% Set surface material 
material dull;

% Set color axis
load('figures/cmap.mat');colormap(cmap);caxis([-10 10]);

% Modify axis light
axis equal;axis vis3d;camlight('right');

% Set axis limits
zLim = get(gca,'ZLim');
if zLim(1) > Zspan(1)*Zfac
    zLim(1) = Zspan(1)*Zfac;
end
if zLim(2) < Zspan(2)*Zfac
    zLim(2) = Zspan(2)*Zfac;
end
axis([ 0 Size 0 Size zLim ]);

% Set axis labels
xlabel('West-East [km]','fontWeight','bold');
ylabel('North-South [km]','fontWeight','bold');
zlabel('Elevation [km]','fontWeight','bold');

% Set axis fonts
set(gca,'fontWeight','bold','XTick',[0,Size])
set(gca,'fontWeight','bold','YTick',[0,Size])
set(gca,'fontWeight','bold','ZTick',[ Zspan(1) Zspan(2) ]*Zfac)
set(gca,'ZTickLabel',{num2str(Zspan(1)),num2str(Zspan(2))})

% Compute surface height
LiftedT         = zeros(size(H));
idx             = find(Typography>0);
LiftedT(idx)    = Typography(idx);
SurfHeight      = Typography+H;
idx             = find(H<0.00005);
SurfHeight(idx) = nan;

% Set color data to highlight waves using red and black colors 
ColorData = Typography - Cfac*abs(SurfHeight-LiftedT) - 1.5;

% Generate surface plot of surface data 
surf(X,flipud(Y),Zfac*SurfHeight,ColorData, ...
    'FaceColor','interp','EdgeColor','none','FaceLighting','phong');

% Set figure view
view(-120,30);

% Force draw figure
drawnow;

end
