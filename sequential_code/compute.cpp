/*
Sequential version in C++ of compute.m code
*/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
#include "functions.h"

using namespace std;

int main(){
  // Basic Parameters of the simulation :
  clock_t  start    = clock();
  clock_t  end;
  int      Size    = 500;  // Size of map, Size*Size [km]
  size_t   nx      = 2001; //Number of cells in each direction on the grid
  float    Tend    = 0.2;  //Simulation time in hours [hr]
  double   dx      = ((float)Size)/((float)nx); //Grid spacening
  cout << "dx = " << dx << endl;
  int numElements = nx*nx;
  size_t memsize = numElements * sizeof(double);
  string  datapath     = "../data/";                  // Path for the data

  // Allocate memory for Computing
  cout <<" Allocating memory .."<<endl;
  double *H, *HU, *HV, *Zdx, *Zdy, *Ht, *HUt, *HVt;
  H   = (double *)malloc(memsize);
  HU  = (double *)malloc(memsize);
  HV  = (double *)malloc(memsize);
  Zdx = (double *)malloc(memsize);
  Zdy = (double *)malloc(memsize);
  Ht  = (double *)malloc(memsize);
  HUt   = (double *)malloc(memsize);
  HVt = (double *)malloc(memsize);

  // Load initial state on host memory
  load_initial_data(H, HU, HV, Zdx, Zdy, datapath, nx, Size, Tend, numElements);

  double  T     = 0.0;
  int     nt    = 0;
  double  dt    = 0.;
  double  C     = 0.0;
  int     Ntmax  = 250;
  double *dt_array;
  dt_array = (double *)malloc(Ntmax*sizeof(double));

  // Evolution loop
  cout  <<  " Computing.."  <<  endl;
  while (T<Tend and nt < Ntmax) {
        // Compute the time-step length
        dt = update_dt(H,HU,HV,dx,numElements);
        if(T+dt > Tend){
          dt = Tend-T;
        }
        //Print status
        cout  << "Computing for T = " << T+dt << " ("<< 100*(T+dt)/Tend << "%)"
              <<"\t dt = "<< dt<< "\t nt = "<< nt << endl;
        // Copy solution to temp storage and enforce boundary condition
        cpy_to(Ht,H,numElements);
        cpy_to(HUt,HU,numElements);
        cpy_to(HVt,HV,numElements);
        enforce_BC(Ht, HUt, HVt, nx);
        // Compute a time-step
        C = (.5*dt/dx);
        FV_time_step(H,HU,HV,Zdx,Zdy,Ht,HUt,HVt,C,dt,nx);
        // Impose tolerances
        impose_tolerances(H,HU,HV,numElements);
        if(nt < Ntmax) dt_array[nt]=dt;
        T = T + dt;
        nt++;
  }

  // Save solution to disk
  ostringstream soutfilename;
  soutfilename <<"../output/Cpp_Solution_nx"<<to_string(nx)<<"_"<<to_string(Size)<<"km_T"<<Tend<<"_h.bin"<< setprecision(2);
  string outfilename = soutfilename.str();

  ofstream fout;
  fout.open(outfilename, std::ios::out | std::ios::binary);
  cout<<"  Writing solution in "<<outfilename<<endl;
  fout.write(reinterpret_cast<char*>(&Ht[0]), numElements*sizeof(double));
  fout.close();

  //save dt historic
  ostringstream soutfilename2;
  soutfilename2 <<"../output/Cpp_dt_nx"<<to_string(nx)<<"_"<<to_string(Size)<<"km_T"<<Tend<<"_h.bin"<< setprecision(2);
  outfilename = soutfilename2.str();
  fout.open(outfilename, std::ios::out | std::ios::binary);
  cout<<"  Writing historic in "<<outfilename<<endl;
  fout.write(reinterpret_cast<char*>(&dt_array[0]), Ntmax*sizeof(double));
  fout.close();

  // Free memory space
  cout<<" Free memory space.."<<endl;
  free(H); free(HU); free(HV); free(Zdx); free(Zdy);
  free(Ht); free(HUt); free(HVt); free(dt_array);
  end = clock()-start;
  end /=CLOCKS_PER_SEC;
  cout  <<  "Ellapsed time \t\t : "  << end << " seconds (" << end/60  <<  "min "
        <<  end%60  <<  "sec)" <<  endl;
  unsigned long int ops = 0;
  ops = nt*( 15 + 2 + 11 + 30 + 30 + 1)*nx*nx;
  cout  <<  "Average performance \t : "  << ops/end/1.0e9 << " gflops" << endl;
  return 0;
}
