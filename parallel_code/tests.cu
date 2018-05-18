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
  //cout << "\t -----\tTesting maximum CUDA function\t-----\t" << '\n';
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

  cudaMemcpy(d_A, h_A, memsize, cudaMemcpyHostToDevice);

  float gpu_elapsed_time;
  cudaEvent_t gpu_start, gpu_stop;
  int i = 7;
  //for( i = 0; i<15; ++i){
  cudaEventCreate(&gpu_start);
  cudaEventCreate(&gpu_stop);

  int Nthreadx = 256;
  dim3 threadsPerBlock(Nthreadx);
  int Nblockx = pow(2,i);//ceil(nx*nx*1.0/Nthreadx)/2;
  dim3 numBlocks(Nblockx);

  cudaEventRecord(gpu_start, 0);
  find_maximum_kernel<<<numBlocks,threadsPerBlock>>>(d_A, d_answer, d_mutex, numElements);
  cudaEventRecord(gpu_stop, 0);

  cudaMemcpy(GPU_answer, d_answer, sizeof(double), cudaMemcpyDeviceToHost);

	cudaEventSynchronize(gpu_stop);
	cudaEventElapsedTime(&gpu_elapsed_time, gpu_start, gpu_stop);
	cudaEventDestroy(gpu_start);
	cudaEventDestroy(gpu_stop);

//  std::cout << Nblockx << "\t" << gpu_elapsed_time << std::endl;
if(true){
  cout << "blocksPerGrid x threadsPerBlock :" << Nthreadx << " x "
      << Nblockx << " (= " << Nthreadx * Nblockx  << ")\n";
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
//}
}
  cudaFree(d_A); cudaFree(d_answer); cudaFree(d_mutex);
  free(h_A); free(GPU_answer);

}

void test_enforce_bc(int Nblocks, int Nthreads){
  int nx = 10;
  int numElements = nx*nx;
  double *original_array = (double *)malloc(numElements*sizeof(double));
  double *GPU_result_array   = (double *)malloc(numElements*sizeof(double));
  double *CPU_result_array   = (double *)malloc(numElements*sizeof(double));
  int x,y;
  for(int i=0; i<numElements; i++){
    x = i%nx;
    y = i/nx;
    if(x == 0 or x == nx-1 or y == 0 or y == nx-1){
      original_array[y*nx+x] = 0.0;
    }
    else{original_array[y*nx+x] = 1;}
  }

  int offset = 0;
  int numthread = 50;
  for(int i=0;i<numthread; i++){
        y = (i+offset)/nx+1;
        x = (i+offset)%nx+1;
    while(x < nx-1 and y < nx-1){
      original_array[y*nx+x] = y*nx+x;
      offset += numthread;
      y = (i+offset)/nx+1;
      x = (i+offset)%nx+1;
    }
    offset = 0;
  }

  cpy_to(CPU_result_array,original_array,nx*nx);
  enforce_BC(CPU_result_array,CPU_result_array,CPU_result_array,nx);

  double *d_array;
  cudaMalloc((void **) &d_array, numElements*sizeof(double));
  cudaMemcpy(d_array, original_array, numElements*sizeof(double), cudaMemcpyHostToDevice);

  call_enforce_BC_kernel<<<Nblocks,Nthreads>>>(d_array,d_array,d_array,nx);

  cudaMemcpy(GPU_result_array, d_array, numElements*sizeof(double), cudaMemcpyDeviceToHost);
  cudaFree(d_array);

  std::cout << "\t\t\t----- Original test array -----\t" << '\n';
  for(int iy=0; iy<nx; iy++){
    for(int ix=0; ix<nx; ix++){
      cout << original_array[iy*nx+ix] << "\t";
    }
    cout<<endl;
  }
  std::cout << "\t\t\t----- CPU Result array -----\t" << '\n';
  for(int iy=0; iy<nx; iy++){
    for(int ix=0; ix<nx; ix++){
      cout << CPU_result_array[iy*nx+ix] << "\t";
    }
    cout<<endl;
  }
  std::cout << "\t\t\t----- GPU Result array -----\t" << '\n';
  for(int iy=0; iy<nx; iy++){
    for(int ix=0; ix<nx; ix++){
      cout << GPU_result_array[iy*nx+ix] << "\t";
    }
    cout<<endl;
  }
    std::cout << "\t\t\t----- Error array -----\t" << '\n';
    for(int iy=0; iy<nx; iy++){
      for(int ix=0; ix<nx; ix++){
        cout << CPU_result_array[iy*nx+ix]-GPU_result_array[iy*nx+ix] << "\t";
      }
      cout<<endl;
    }
  free(original_array);
  free(GPU_result_array);
}
