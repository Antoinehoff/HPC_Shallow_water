/*
Parallel version in CUDA/C++ of compute.cpp code
*/
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <cstring>
#include <sstream>
#include "kernels.cuh" //parallel kernels
#include "functions.h" //Sequential functions

using namespace std;

int main(){
  // Basic Parameters of the simulation :
  clock_t timer = clock();
  int     Size        = 500;                          // Size of map, Size*Size [km]
  size_t  nx          = 2001;                         // Grid 1D size
  float   Tend        = 0.2;                          // Simulation time in hours [hr]
  double  dx          = ((float)Size)/((float)nx);    // Grid spacening
  int     numElements = nx*nx;                        // Total number of elements
  size_t  memsize     = numElements * sizeof(double); // Memory size of one array
  int     Ntmax       = 250;                            // Choose the maximum of iteration
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
  // Tracking variables
  double *dt_array;                                   // Record the evolution time steps
  string  datapath     = "../data/";                  // Path for the data

  // Allocate host memory for loading the initial conditions
  cout << " Allocating host memory .."  <<  endl;
  H   = (double *)malloc(memsize);
  HU  = (double *)malloc(memsize);  HV  = (double *)malloc(memsize);
  Ht  = (double *)malloc(memsize);
  HUt = (double *)malloc(memsize);  HVt = (double *)malloc(memsize);
  Zdx = (double *)malloc(memsize);  Zdy = (double *)malloc(memsize);
  dt_array = (double *)malloc(Ntmax*sizeof(double));

  // Load initial state on host memory
  load_initial_data(H, HU, HV, Zdx, Zdy, datapath, nx, Size, Tend, numElements);

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

  // One dimensional grid block threads version :
  int Nthreadx = 128;
  dim3 threadsPerBlock(Nthreadx);
  int Nblockx = ceil(nx*nx*1.0/Nthreadx);
  dim3 numBlocks(Nblockx);
  cout << "1D parallel model description :"<<endl;
  cout <<"\t Number of elements \t\t:" << nx*nx << endl;
  cout <<"\t Number of blocks needed \t:" << Nblockx << "x" << 1 << endl;
  cout <<"\t Nthreads \t\t\t:" << Nblockx*Nthreadx <<endl;

  // Evolution loop
  while (T < Tend and nt < Ntmax) {
        // Compute the time-step length
        dt = update_dt(H,HU,HV,dx,numElements);
        if(T+dt > Tend){
          dt = Tend-T;
        }
        //Print status
        cout  << " Computing for T=" << T+dt << " ("<< 100*(T+dt)/Tend << "%), "
              <<  "dt="             << dt   << endl;
        // Copy solution to temp storage and enforce boundary condition
        //cpy_to(Ht,H,numElements);
        //cpy_to(HUt,HU,numElements);
        //cpy_to(HVt,HV,numElements);
        //enforce_BC(Ht, HUt, HVt, nx);
        // Compute a time-step
        C = (.5*dt/dx);
        //FV_time_step(H,HU,HV,Zdx,Zdy,Ht,HUt,HVt,C,dt,nx);
        FV_time_step_kernel<<<Nblockx,Nthreadx>>>(d_H,d_HU,d_HV,d_Zdx,d_Zdy,d_Ht,d_HUt,d_HVt,C,dt,nx);
        // Impose tolerances
        //impose_tolerances(Ht,HUt,HVt,numElements);
        if(nt < Ntmax) dt_array[nt]=dt;
        T = T + dt;
        nt++;
  }

  // Copy device result to the host memory
  cout << " Copy the output data from the CUDA device to the host memory" << endl;
  cudaMemcpy(Ht, d_Ht, memsize, cudaMemcpyDeviceToHost);
  // Save solution to disk
  ostringstream soutfilename;
  soutfilename <<"../output/CUDA_Solution_nx"<<to_string(nx)<<"_"<<to_string(Size)<<"km_T"<<Tend<<"_h.bin"<< setprecision(2);
  string outfilename = soutfilename.str();

  ofstream fout;
  fout.open(outfilename, std::ios::out | std::ios::binary);
  cout<<" Writing solution in "<<outfilename<<endl;
  fout.write(reinterpret_cast<char*>(&Ht[0]), numElements*sizeof(double));
  fout.close();

  //save dt historic
  ostringstream soutfilename2;
  soutfilename2 <<"../output/CUDA_dt_nx"<<to_string(nx)<<"_"<<to_string(Size)<<"km_T"<<Tend<<"_h.bin"<< setprecision(2);
  outfilename = soutfilename2.str();
  fout.open(outfilename, std::ios::out | std::ios::binary);
  cout<<" Writing solution in "<<outfilename<<endl;
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
  return 0;
}
