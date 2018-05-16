#include "tests.cuh"
#include "kernels.cuh"
#include "functions.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>


void test_max(){
  cout << "\t -----\tTesting maximum CUDA function\t-----\t" << '\n';
  int numElements   = 2001*2001;
  size_t memsize    = numElements * sizeof(double);
  double CPU_answer = -1;

  double *h_A       = (double *)malloc(memsize);
  double *GPU_answer  = (double *)malloc(sizeof(double));

  load_initial_state("../data/Data_nx2001_500km_T0.2_h.bin",   h_A,    numElements);

  clock_t timer_h;
  timer_h=clock();
  for(int i=0; i<numElements; i++){
    CPU_answer = fmaxf(CPU_answer, h_A[i]);
  }
  timer_h = clock()-timer_h;
  timer_h = (double) timer_h/CLOCKS_PER_SEC*1000.0;

  double *d_A, *d_answer; int *d_mutex;
  cudaMalloc((void **)&d_A,memsize);
  cudaMalloc((void **)&d_answer,sizeof(double));
  cudaMemset(d_answer, 0, sizeof(float));
  cudaMalloc((void **)&d_mutex, sizeof(int));
	cudaMemset(d_mutex, 0, sizeof(float));

  int threadsPerBlock = 128;
  int blocksPerGrid   = 256;//(numElements + threadsPerBlock - 1)/threadsPerBlock;
  cout << "blocksPerGrid x threadsPerBlock :" << threadsPerBlock << " x "
       << blocksPerGrid << " (= " << threadsPerBlock * blocksPerGrid  << ")\n";
  float gpu_elapsed_time;
  cudaEvent_t gpu_start, gpu_stop;
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);

  cudaMemcpy(d_A, h_A, memsize, cudaMemcpyHostToDevice);

  cudaEventRecord(gpu_start, 0);
  find_maximum_kernel<<<threadsPerBlock,blocksPerGrid>>>(d_A, d_answer, d_mutex, numElements);
  cudaEventRecord(gpu_stop, 0);

  cudaMemcpy(GPU_answer, d_answer, sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

  std::cout  << "The gpu took: "        <<  gpu_elapsed_time      <<  " milli-seconds"  <<  std::endl;
  std::cout  << "The cpu took: "        <<  timer_h               <<  " milli-seconds"  <<  std::endl;
  if(abs(GPU_answer[0]-CPU_answer) < 1e-5){
    std::cout <<  "Test PASSED "        << '\n';
  }
  else{
    std::cout <<  "Test FAILED "        << '\n';
    std::cout  << " CPU Answer : "      <<  CPU_answer            << '\n';
    std::cout  << " GPU Answer : "      <<  GPU_answer[0]           << '\n';
  }


  cudaFree(d_A); cudaFree(d_answer); cudaFree(d_mutex);
  free(h_A); free(GPU_answer);

}

void test_update_dt(){
  cout << "\t -----\tTesting update_dt CUDA function\t-----\t" << '\n';
  int nx            = 2001;
  int Size          = 500;
  double dx         = Size/nx;
  int numElements   = 2001*2001;
  size_t memsize    = numElements * sizeof(double);
  double CPU_dt     = -1;

  double *H         = (double *)malloc(memsize);
  double *HU        = (double *)malloc(memsize);
  double *HV        = (double *)malloc(memsize);
  double *GPU_dt      = (double *)malloc(sizeof(double));

  load_initial_state("../data/Data_nx2001_500km_T0.2_h.bin",  H,  numElements);
  load_initial_state("../data/Data_nx2001_500km_T0.2_hu.bin", HU, numElements);
  load_initial_state("../data/Data_nx2001_500km_T0.2_hv.bin", HV, numElements);

  clock_t timer_h;
  timer_h   =clock();
  CPU_dt    = update_dt(H,HU,HV,dx,numElements);
  timer_h   = clock()-timer_h;
  timer_h   = (double) timer_h/CLOCKS_PER_SEC*1000.0;

  double *d_H, *d_HU, *d_HV, *d_dt;
  cudaMalloc((void **)&d_H,memsize);
  cudaMalloc((void **)&d_HU,memsize);
  cudaMalloc((void **)&d_HV,memsize);
  cudaMalloc((void **)&d_dt,sizeof(double));

  int threadsPerBlock = 256;
  int blocksPerGrid   = ceil(numElements*1.0/256);//(numElements + threadsPerBlock - 1)/threadsPerBlock;
  cout << "blocksPerGrid x threadsPerBlock :" << threadsPerBlock << " x "
       << blocksPerGrid << " (= " << threadsPerBlock * blocksPerGrid  << ")\n";
  float gpu_elapsed_time;
  cudaEvent_t gpu_start, gpu_stop;
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);

  cudaMemcpy(d_H,  H,  memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_HU, HU, memsize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_HV, HV, memsize, cudaMemcpyHostToDevice);

  cudaEventRecord(gpu_start, 0);
  //update_dt_kernel<<<threadsPerBlock,blocksPerGrid>>>(d_H, d_HU, d_HV, d_dt, dx, numElements);
  cudaEventRecord(gpu_stop, 0);

  cudaMemcpy(GPU_dt, d_dt, sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

  std::cout  << "The gpu took: "        <<  gpu_elapsed_time      <<  " milli-seconds"  <<  std::endl;
  std::cout  << "The cpu took: "        <<  timer_h               <<  " milli-seconds"  <<  std::endl;
  if(abs(GPU_dt[0] - CPU_dt) < 1e-5){
    std::cout <<  "Test PASSED "        << '\n';
  }
  else{
    std::cout <<  "Test FAILED "        << '\n';
    std::cout  << " CPU Answer : "      <<  CPU_dt              << '\n';
    std::cout  << " GPU Answer : "      <<  GPU_dt[0]           << '\n';
  }


  cudaFree(d_H); cudaFree(d_HU); cudaFree(d_HV); cudaFree(d_dt);
  free(H); free(HU); free(HV); free(GPU_dt);
}
