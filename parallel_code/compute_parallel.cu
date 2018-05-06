/*
Paralleel version in CUDA/C++ of compute.cpp code
*/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>

using namespace std;

// Functions predefinitions
void    load_initial_state  (string, double *, int);
double  update_dt           (const double *, const double *,
                             const double *,double, int);
void    cpy_to              (double *, const double *, int);
int     to_idx              (int, int, int);
void    enforce_BC          (double *, double *, double *, int);
void    time_step           (double *, double *, double *,
                             const double *,  const double *, const double *,
                             const double *,  const double *, double,
                             double,          int);
void    impose_tolerances   (double *, double *, double *, int);


double const g = 127267.2; //Gravity, 9.82*(3.6)^2*1000 in [km / hr^2]

int main(){
  // Basic Parameters of the simulation :
  clock_t timer = clock();
  int     Size        = 500;                          // Size of map, Size*Size [km]
  size_t  nx          = 2001;                         // Grid 1D size
  float   Tend        = 0.2;                          // Simulation time in hours [hr]
  double  dx          = ((float)Size)/((float)nx);    // Grid spacening
  int     numElements = nx*nx;                        // Total number of elements
  size_t  memsize     = numElements * sizeof(double); // Memory size of one array
  int     Ntmax       = 1;                            // Choose the maximum of iteration
  // Simulation variables HOST
  double  T           = 0.0;                          // Time
  int     nt          = 0;                            // Iteration counter
  double  dt          = 0.;                           // Time step
  double  C           = 0.0;                          // Coefficient 1/2*dt/dx
  double *H,    *HU,  *HV;                            // Water height and x,y speeds
  double *Ht,   *HUt, *HVt;                           // Temporary memory of H HU and HV
  double *Zdx,  *Zdy;                                 // Topology of the map
  // Simulation variables DEVICE
  double *d_H,    *d_HU,  *d_HV;                      // Water height and x,y speeds
  double *d_Ht,   *d_HUt, *d_HVt;                     // Temporary memory of H HU and HV
  double *d_Zdx,  *d_Zdy;                             // Topology of the map
  // Filestream variables
  string  datapath     = "../data/";                  // Path for the data
  string  identificator;                              // Identificator of the simulation
  ostringstream sfilename;                            // Identificator auxiliary
  // Tracking variables
  double *dt_array;                                   // Record the evolution time steps

  // Data filename :
  sfilename <<  datapath <<  "Data_nx" <<  to_string(nx) <<  "_"
            <<  to_string(Size) <<  "km_T"  <<  Tend  <<  setprecision(2);
  identificator = sfilename.str();
  cout << " Simulation identificator : "  <<  identificator  << endl;

  // Allocate host memory for loading the initial conditions
  cout << " Allocating host memory .."  <<  endl;
  H   = (double *)malloc(memsize);
  HU  = (double *)malloc(memsize);  HV  = (double *)malloc(memsize);
  Ht  = (double *)malloc(memsize);
  HUt = (double *)malloc(memsize);  HVt = (double *)malloc(memsize);
  Zdx = (double *)malloc(memsize);  Zdy = (double *)malloc(memsize);
  dt_array = (double *)malloc(Ntmax*sizeof(double));

  // Load initial condition from data files
  cout <<" Loading data on host memory.." << endl;
  load_initial_state(identificator + "_h.bin",   H,    numElements);
  load_initial_state(identificator + "_hu.bin",  HU,   numElements);
  load_initial_state(identificator + "_hv.bin",  HV,   numElements);
  // Load topography slopes from data files
  load_initial_state(identificator + "_Zdx.bin", Zdx,  numElements);
  load_initial_state(identificator + "_Zdy.bin", Zdy,  numElements);

  // Allocate device memory for computing
  cout  <<  " Allocating device memory on host.." <<  endl;
  cudaMalloc((void **)  &d_H,   memsize);
  cudaMalloc((void **)  &d_HU,  memsize); cudaMalloc((void **)  &d_HV, memsize);
  cudaMalloc((void **)  &d_Ht,  memsize);
  cudaMalloc((void **)  &d_HUt, memsize); cudaMalloc((void **)  &d_HVt,memsize);
  cudaMalloc((void **)  &d_Zdx, memsize); cudaMalloc((void **)  &d_Zdy,memsize);

  // Copy initial conditions from host to device
  cout << " Copying variables from host to device.."  <<  endl;
  cudaMemcpy(d_H,   H,    memsize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_HU,  HU,   memsize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_HV,  HV,   memsize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_Zdx, Zdx,  memsize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_Zdy, Zdy,  memsize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_Ht,  Ht,   memsize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_HUt, HUt,  memsize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_HVt, HVt,  memsize,  cudaMemcpyHostToDevice);

  // Evolution loop
  while (T < Tend and nt < Ntmax) {
        // Compute the time-step length
        dt = update_dt(H,HU,HV,dx,numElements);
        if(T+dt > Tend){
          dt = Tend-T;
        }
        //Print status
        cout  << "Computing for T=" << T+dt << " ("<< 100*(T+dt)/Tend << "%), "
              <<  "dt="             << dt   << endl;
        // Copy solution to temp storage and enforce boundary condition
        cpy_to(Ht,H,numElements);
        cpy_to(HUt,HU,numElements);
        cpy_to(HVt,HV,numElements);
        enforce_BC(Ht, HUt, HVt, nx);
        // Compute a time-step
        C = (.5*dt/dx);
        time_step(H,HU,HV,Zdx,Zdy,Ht,HUt,HVt,C,dt,nx);
        // Impose tolerances
        impose_tolerances(Ht,HUt,HVt,numElements);
        if(nt < Ntmax) dt_array[nt]=dt;
        T = T + dt;
        nt++;
  }

  // Save solution to disk
  ostringstream soutfilename;
  soutfilename <<"../output/CUDA_Solution_nx"<<to_string(nx)<<"_"<<to_string(Size)<<"km_T"<<Tend<<"_h.bin"<< setprecision(2);
  string outfilename = soutfilename.str();

  ofstream fout;
  fout.open(outfilename, std::ios::out | std::ios::binary);
  cout<<"Writing solution in "<<outfilename<<endl;
  fout.write(reinterpret_cast<char*>(&Ht[0]), numElements*sizeof(double));
  fout.close();

  //save dt historic
  ostringstream soutfilename2;
  soutfilename2 <<"../output/Cpp_dt_nx"<<to_string(nx)<<"_"<<to_string(Size)<<"km_T"<<Tend<<"_h.bin"<< setprecision(2);
  outfilename = soutfilename2.str();
  fout.open(outfilename, std::ios::out | std::ios::binary);
  cout<<"Writing solution in "<<outfilename<<endl;
  fout.write(reinterpret_cast<char*>(&dt_array[0]), Ntmax*sizeof(double));
  fout.close();

  // Free device global memory
  cout  <<  " Free device memory space.." <<  endl;
  cudaFree(d_H);    cudaFree(d_HU);   cudaFree(d_HV);   cudaFree(d_Zdx);
  cudaFree(d_Zdy);  cudaFree(d_Ht);   cudaFree(d_HUt);  cudaFree(d_HVt);

  // Free host memory
  cout  <<  " Free host memory space.." <<  endl;
  free(H);    free(HU);   free(HV);   free(Zdx); free(Zdy);
  free(Ht);   free(HUt);  free(HVt);  free(dt_array);

  // Timer end
  timer = clock()-timer;
  timer = (double)(timer)/CLOCKS_PER_SEC*1000;
  cout  <<  "Ellapsed time : "  <<  timer/60000  <<  "min "
        <<  timer/1000  <<  "s " << timer%1000 << "ms" << endl;
  std::cout << timer<< " " << CLOCKS_PER_SEC << '\n';
  return 0;
}


void load_initial_state(string identificator, double * H, int numElements){
  ifstream fin;
  fin.open(identificator, ios::in|ios::binary);
  if(!fin){
    cerr<<" Error, couldn't find file : "<<identificator<<endl;
    exit(EXIT_FAILURE);
  }
  fin.read(reinterpret_cast<char*>(&H[0]), numElements*sizeof(double));
  fin.close();
}

double update_dt(const double *H, const double *HU,
      const double *HV, double dx, int numElements){
  //Compute the max of mu and give dt back
  double mu = 0.0;
  double newmu = 0.0;
  for(int i=0; i<numElements; i++){
    newmu = sqrt(pow(max(abs(HU[i]/H[i]-sqrt(H[i]*g)),abs(HU[i]/H[i]+sqrt(H[i]*g))),2)
                +pow(max(abs(HV[i]/H[i]-sqrt(H[i]*g)),abs(HV[i]/H[i]+sqrt(H[i]*g))),2));
    if(newmu > mu){
      mu = newmu;
      }
    }
    return dx/(sqrt(2.0)*mu);
  }

void cpy_to(double *target, const double *source, int numElements){
  for(int i=0; i<numElements; i++){
      target[i] = source[i];
  }
}

int to_idx(int x, int y, int nx){
  return y * (nx) + x;
}

void enforce_BC(double *Ht, double *HUt, double *HVt, int nx){
  for(int i=0; i<nx; ++i){
    Ht[to_idx(0,i,nx)]      = Ht[to_idx(1,i,nx)];
    HUt[to_idx(0,i,nx)]     = HUt[to_idx(1,i,nx)];
    HVt[to_idx(0,i,nx)]     = HVt[to_idx(1,i,nx)];
  }
  for(int i=0; i<nx; ++i){
    Ht[to_idx(nx-1,i,nx)]   = Ht[to_idx(nx-2,i,nx)];
    HUt[to_idx(nx-1,i,nx)]  = HUt[to_idx(nx-2,i,nx)];
    HVt[to_idx(nx-1,i,nx)]  = HVt[to_idx(nx-2,i,nx)];
  }
  for(int i=0; i<nx; ++i){
    Ht[to_idx(i,0,nx)]      = Ht[to_idx(i,1,nx)];
    HUt[to_idx(i,0,nx)]     = HUt[to_idx(i,1,nx)];
    HVt[to_idx(i,0,nx)]     = HVt[to_idx(i,1,nx)];
  }
  for(int i=0; i<nx; ++i){
    Ht[to_idx(i,nx-1,nx)]   = Ht[to_idx(i,nx-2,nx)];
    HUt[to_idx(i,nx-1,nx)]  = HUt[to_idx(i,nx-2,nx)];
    HVt[to_idx(i,nx-1,nx)]  = HVt[to_idx(i,nx-2,nx)];
  }
}

void time_step( double *H,        double *HU,        double *HV,
               const double *Zdx, const double *Zdy,
               const double *Ht,  const double *HUt, const double *HVt,
               double C,          double dt,         int nx){
  for(int x=1; x<nx-1; x++){
    for(int y=1; y<nx-1; y++){
      H[to_idx(x,y,nx)]=
        0.25*( Ht[to_idx(x+1,y,nx)] + Ht[to_idx(x-1,y,nx)]
              +Ht[to_idx(x,y+1,nx)] + Ht[to_idx(x,y-1,nx)]  )+
        C   *( HUt[to_idx(x,y-1,nx)] - HUt[to_idx(x,y+1,nx)]
              +HVt[to_idx(x-1,y,nx)] - HVt[to_idx(x+1,y,nx)] );

      HU[to_idx(x,y,nx)]=
        0.25*( HUt[to_idx(x+1,y,nx)] + HUt[to_idx(x-1,y,nx)]
              +HUt[to_idx(x,y+1,nx)] + HUt[to_idx(x,y-1,nx)])
              -dt*g * H[to_idx(x,y,nx)] * Zdx[to_idx(x,y,nx)]
       +C   *( pow(HUt[to_idx(x,y-1,nx)],2) / Ht[to_idx(x,y-1,nx)]
              +0.5*g * pow(Ht[to_idx(x,y-1,nx)],2)
              -pow(HUt[to_idx(x,y+1,nx)],2) / Ht[to_idx(x,y+1,nx)]
              -0.5*g * pow(Ht[to_idx(x,y+1,nx)],2)                )
       +C   *( HUt[to_idx(x-1,y,nx)] * HVt[to_idx(x-1,y,nx)] / Ht[to_idx(x-1,y,nx)]
              -HUt[to_idx(x+1,y,nx)] * HVt[to_idx(x+1,y,nx)] / Ht[to_idx(x+1,y,nx)]);

      HV[to_idx(x,y,nx)]  =
        0.25*( HVt[to_idx(x+1,y,nx)] + HVt[to_idx(x-1,y,nx)]
              +HVt[to_idx(x,y+1,nx)] + HVt[to_idx(x,y-1,nx)])
              -dt*g * H[to_idx(x,y,nx)] * Zdy[to_idx(x,y,nx)]
       +C   *( HUt[to_idx(x,y-1,nx)] * HVt[to_idx(x,y-1,nx)] / Ht[to_idx(x,y-1,nx)]
              -HUt[to_idx(x,y+1,nx)] * HVt[to_idx(x,y+1,nx)] / Ht[to_idx(x,y+1,nx)])
       +C   *( pow(HVt[to_idx(x-1,y,nx)],2) / Ht[to_idx(x-1,y,nx)]
              +0.5*g * pow(Ht[to_idx(x-1,y,nx)],2)
              -pow(HVt[to_idx(x+1,y,nx)],2) / Ht[to_idx(x+1,y,nx)]
              -0.5*g * pow(Ht[to_idx(x+1,y,nx)],2));
    }
  }
}

void impose_tolerances(double *Ht, double *HUt, double *HVt, int numElements){
  for(int i=0; i<numElements; i++){
    if(Ht[i]<0){
      Ht[i] = 1e-5;
    }
    if(Ht[i]<= 1e-5){
      HUt[i] = 0;
      HVt[i] = 0;
    }
  }
}

void display(double *A, int nx){
  cout<<endl;
  for(int i=0; i<nx; i++){
    for(int j=0; j<nx; j++){
      cout<<A[to_idx(i,j,nx)]<<"\t";
    }
    cout<<endl;
  }
  cout<<endl;
}
