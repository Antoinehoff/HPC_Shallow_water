nx      = 2001;
fname   = "Solution_nx"+num2str(nx)+"_500km_T0.2_h.bin";
path    = "output/";
fmatlab = path + "Matlab_" + fname
fcpp    = path + "Cpp_" + fname

Hmatlab = fread(fopen(fmatlab,'r'),[nx,nx],'double');
Hcpp = fread(fopen(fcpp,'r'),[nx,nx],'double');

diff = abs(Hmatlab-Hcpp);
max(max(diff))
max(max(Hmatlab))
max(max(Hcpp))