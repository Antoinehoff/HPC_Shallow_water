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
  clock_t timer = clock();
  int      Size    = 500;  // Size of map, Size*Size [km]
  size_t   nx      = 2001; //Number of cells in each direction on the grid
  float    Tend    = 0.2;  //Simulation time in hours [hr]
  double   dx      = ((float)Size)/((float)nx); //Grid spacening
  cout << "dx = " << dx << endl;
  int numElements = nx*nx;
  size_t memsize = numElements * sizeof(double);

  // Data filename :
  ostringstream sfilename;
  string addpath = "../data/";
  sfilename <<addpath<<"Data_nx"<<to_string(nx)<<"_"<<to_string(Size)<<"km_T"<<Tend<< setprecision(2);
  string filename = sfilename.str();

  // Allocate memory for Computing
  cout <<" Allocating memory .."<<endl;
  double *H, *HU, *HV, *Zdx, *Zdy, *Ht, *HUt, *HVt;
  H = (double *)malloc(memsize);
  HU = (double *)malloc(memsize);
  HV = (double *)malloc(memsize);
  Zdx = (double *)malloc(memsize);
  Zdy = (double *)malloc(memsize);
  Ht = (double *)malloc(memsize);
  HUt = (double *)malloc(memsize);
  HVt = (double *)malloc(memsize);

  // Load initial condition from data files
  cout <<" Loading data.." << endl;
  load_initial_state(filename+"_h.bin",H,numElements);
  load_initial_state(filename+"_hu.bin",HU,numElements);
  load_initial_state(filename+"_hv.bin",HV,numElements);
  // Load topography slopes from data files
  load_initial_state(filename+"_Zdx.bin",Zdx,numElements);
  load_initial_state(filename+"_Zdy.bin",Zdy,numElements);

  double T = 0.0;
  int nt = 0;
  double dt = 0.;
  double C = 0.0;
  int Nmax = 250;
  double *dt_array;
  dt_array = (double *)malloc(Nmax*sizeof(double));

  // Evolution loop
  while (T < Tend) {
        // Compute the time-step length
        dt = update_dt(H,HU,HV,dx,numElements);
        if(T+dt > Tend){
          dt = Tend-T;
        }
        //Print status
        cout << "Computing for T = " << T+dt << " ("<< 100*(T+dt)/Tend << "%)"<<endl;
        cout <<"dt = "<< dt<<endl;
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
        if(nt < Nmax) dt_array[nt]=dt;
        T = T + dt;
        nt++;
  }

  // Save solution to disk
  ostringstream soutfilename;
  soutfilename <<"../output/Cpp_Solution_nx"<<to_string(nx)<<"_"<<to_string(Size)<<"km_T"<<Tend<<"_h.bin"<< setprecision(2);
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
  fout.write(reinterpret_cast<char*>(&dt_array[0]), Nmax*sizeof(double));
  fout.close();

  // Free memory space
  cout<<" Free memory space.."<<endl;
  free(H); free(HU); free(HV); free(Zdx); free(Zdy);
  free(Ht); free(HUt); free(HVt); free(dt_array);
  timer = clock()-timer;
  timer = (double)(timer)/CLOCKS_PER_SEC;
  cout<<"Ellapsed time : "<<timer/60<<"min "<<timer%60<<"sec"<<endl;
  return 0;
}
