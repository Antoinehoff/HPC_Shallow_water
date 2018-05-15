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
  double *h_answer  = (double *)malloc(sizeof(double));

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

  int threadsPerBlock = 256;
  int blocksPerGrid   = ceil(numElements*1.0/256);//(numElements + threadsPerBlock - 1)/threadsPerBlock;
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

  cudaMemcpy(h_answer, d_answer, sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

  std::cout  << "The gpu took: "        <<  gpu_elapsed_time      <<  " milli-seconds"  <<  std::endl;
  std::cout  << "The cpu took: "        <<  timer_h               <<  " milli-seconds"  <<  std::endl;
  if(h_answer[0]-CPU_answer < 1e-5){
    std::cout <<  "Test PASSED "        << '\n';
  }
  else{
    std::cout <<  "Test FAILED "        << '\n';
    std::cout  << " CPU Answer : "      <<  CPU_answer            << '\n';
    std::cout  << " GPU Answer : "      <<  h_answer[0]           << '\n';
  }


  cudaFree(d_A); cudaFree(d_answer); cudaFree(d_mutex);
  free(h_A); free(h_answer);

}
