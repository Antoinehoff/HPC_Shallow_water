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
  for(int i = 0; i<10; ++i){
  // Profiling variables :
  clock_t  start;
  clock_t  end;
  clock_t *time_array = (clock_t *)malloc(9*sizeof(clock_t));
  for(int i=0; i<9; ++i){time_array[i]=0;}
  // Basic Parameters of the simulation :
//START TOTAL TIME MEASUREMENT
start = clock();
//START PROFILING 0
time_array[0] -= clock();
  int      Size    = 500;  // Size of map, Size*Size [km]
  size_t   nx      = 2001; //Number of cells in each direction on the grid
  float    Tend    = 0.2;  //Simulation time in hours [hr]
  double   dx      = ((float)Size)/((float)nx); //Grid spacening
//  cout << "dx = " << dx << endl;
  int numElements = nx*nx;
  size_t memsize = numElements * sizeof(double);

  // Loop parameters :
  double  T     = 0.0;
  int     nt    = 0;
  double  dt    = 0.;
  double  C     = 0.0;
  int     Nmax  = 250;
  double *dt_array;
  dt_array = (double *)malloc(Nmax*sizeof(double));

  // Data filename :
  ostringstream sfilename;
  string addpath = "../data/";
  sfilename <<addpath<<"Data_nx"<<to_string(nx)<<"_"<<to_string(Size)<<"km_T"<<Tend<< setprecision(2);
  string filename = sfilename.str();

  // Allocate memory for Computing
//  cout <<" Allocating memory .."<<endl;
  double *H, *HU, *HV, *Zdx, *Zdy, *Ht, *HUt, *HVt;
  H   = (double *)malloc(memsize);
  HU  = (double *)malloc(memsize);
  HV  = (double *)malloc(memsize);
  Zdx = (double *)malloc(memsize);
  Zdy = (double *)malloc(memsize);
  Ht  = (double *)malloc(memsize);
  HUt = (double *)malloc(memsize);
  HVt = (double *)malloc(memsize);
time_array[0] += clock();
//END PROFILING 0

  // Load initial condition from data files
//START PROFILING 1
time_array[1] -= clock();
//  cout <<" Loading data.." << endl;
  load_initial_state(filename+"_h.bin",H,numElements);
  load_initial_state(filename+"_hu.bin",HU,numElements);
  load_initial_state(filename+"_hv.bin",HV,numElements);
  // Load topography slopes from data files
  load_initial_state(filename+"_Zdx.bin",Zdx,numElements);
  load_initial_state(filename+"_Zdy.bin",Zdy,numElements);
time_array[1] += clock();
//END PROFILING 1

  // Evolution loop
//  cout  <<  " Computing.."  <<  endl;
  while (nt < Nmax) {
        // Compute the time-step length
//START PROFILING 2
time_array[2] -= clock();
        dt = update_dt(H,HU,HV,dx,numElements);
        if(T+dt > Tend){
          dt = Tend-T;
        }
time_array[2] += clock();
//END PROFILING 2
        //Print status
//        cout  << "Computing for T = " << T+dt << " ("<< 100*(T+dt)/Tend << "%)"
//              <<"\t dt = "<< dt<< "\t nt = "<< nt << endl;
        // Copy solution to temp storage and enforce boundary condition
//START PROFILING 3
time_array[3] -= clock();
        cpy_to(Ht,H,numElements);
        cpy_to(HUt,HU,numElements);
        cpy_to(HVt,HV,numElements);
time_array[3] += clock();
//END PROFILING 3
//START PROFILING 4
time_array[4] -= clock();
        enforce_BC(Ht, HUt, HVt, nx);
time_array[4] += clock();
//END PROFILING 4

        // Compute a time-step
//START PROFILING 5
time_array[5] -= clock();
        C = (.5*dt/dx);
        FV_time_step(H,HU,HV,Zdx,Zdy,Ht,HUt,HVt,C,dt,nx);
time_array[5] += clock();
//END PROFILING 5


        // Impose tolerances

//START PROFILING 6
time_array[6] -= clock();
        impose_tolerances(H,HU,HV,numElements);
time_array[6] += clock();
//END PROFILING 6

        if(nt < Nmax) dt_array[nt]=dt;
        T = T + dt;
        nt++;
  }

  // Save solution to disk
//START PROFILING 7
time_array[7] -= clock();
  ostringstream soutfilename;
  soutfilename <<"../output/Cpp_Solution_nx"<<to_string(nx)<<"_"<<to_string(Size)<<"km_T"<<Tend<<"_h.bin"<< setprecision(2);
  string outfilename = soutfilename.str();

  ofstream fout;
  fout.open(outfilename, std::ios::out | std::ios::binary);
//  cout<<"  Writing solution in "<<outfilename<<endl;
  fout.write(reinterpret_cast<char*>(&Ht[0]), numElements*sizeof(double));
  fout.close();

  //save dt historic
  ostringstream soutfilename2;
  soutfilename2 <<"../output/Cpp_dt_nx"<<to_string(nx)<<"_"<<to_string(Size)<<"km_T"<<Tend<<"_h.bin"<< setprecision(2);
  outfilename = soutfilename2.str();
  fout.open(outfilename, std::ios::out | std::ios::binary);
//  cout<<"  Writing historic in "<<outfilename<<endl;
  fout.write(reinterpret_cast<char*>(&dt_array[0]), Nmax*sizeof(double));
  fout.close();
time_array[7] += clock();
//END PROFILING 7

  // Free memory space
//START PROFILING 8
time_array[8] -= clock();
//  cout<<" Free memory space.."<<endl;
  free(H); free(HU); free(HV); free(Zdx); free(Zdy);
  free(Ht); free(HUt); free(HVt); free(dt_array);
time_array[8] += clock();
//END PROFILING 9
end = clock()-start;
//END TOTAL TIME PROFILING

  double time_ms = ((double)end)/CLOCKS_PER_SEC*1000.0;
  clock_t ttp = 0;
  clock_t while_time = 0;
  for(int i=0; i<9; ++i){ ttp += time_array[i];}
  for(int i=2; i<7; ++i){while_time += time_array[i];}
/*  cout  <<  "Total ticks \t\t\t: "          <<  end             << endl;
  cout  <<  "Variables initialization \t: " <<  time_array[0]   << "\t\t" << ((double)time_array[0])/ttp*100.0 << "%"<< endl;
  cout  <<  "Initial state loading \t\t: "  <<  time_array[1]   << "\t\t" << ((double)time_array[1])/ttp*100.0 << "%"<< endl;
  cout  <<  "Dt updating \t\t\t: "          <<  time_array[2]   << "\t"   << ((double)time_array[2])/ttp*100.0 << "%"<< endl;
  cout  <<  "Copy temporary variables \t: " <<  time_array[3]   << "\t"   << ((double)time_array[3])/ttp*100.0 << "%"<< endl;
  cout  <<  "Enforce BC \t\t\t: "           <<  time_array[4]   << "\t\t" << ((double)time_array[4])/ttp*100.0 << "%"<< endl;
  cout  <<  "Time step performing \t\t: "   <<  time_array[5]   << "\t"   << ((double)time_array[5])/ttp*100.0 << "%"<< endl;
  cout  <<  "Impose tolerances \t\t: "      <<  time_array[6]   << "\t"   << ((double)time_array[6])/ttp*100.0 << "%"<< endl;
  cout  <<  "Save output \t\t\t: "          <<  time_array[7]   << "\t\t" << ((double)time_array[7])/ttp*100.0 << "%"<< endl;
  cout  <<  "Free memory space \t\t: "      <<  time_array[8]   << "\t\t" << ((double)time_array[8])/ttp*100.0 << "%"<< endl;
  cout  <<  "Total profiling \t\t: "        <<  ttp             << endl;
*/
  cout  <<  (double)time_array[0]/CLOCKS_PER_SEC*1000.0   << ",";
  cout  <<  (double)time_array[1]/CLOCKS_PER_SEC*1000.0   << ",";
  cout  <<  (double)time_array[2]/CLOCKS_PER_SEC*1000.0   << ",";
  cout  <<  (double)time_array[3]/CLOCKS_PER_SEC*1000.0   << ",";
  cout  <<  (double)time_array[4]/CLOCKS_PER_SEC*1000.0   << ",";
  cout  <<  (double)time_array[5]/CLOCKS_PER_SEC*1000.0   << ",";
  cout  <<  (double)time_array[6]/CLOCKS_PER_SEC*1000.0   << ",";
  cout  <<  (double)time_array[7]/CLOCKS_PER_SEC*1000.0   << ",";
  cout  <<  (double)time_array[8]/CLOCKS_PER_SEC*1000.0   << "," << endl;
//  cout  <<  ttp             << endl;

  unsigned long int ops = 0;
  ops = nt*( 15 + 2 + 11 + 30 + 30 + 1)*nx*nx;
  free(time_array);
}
  return 0;
}
