#include<iostream>
#include "kernels.cuh"

using namespace std;


int main(){
  int numElements   = 1<<22;
  size_t memsize    = numElements * sizeof(double);
  double CPU_answer = -1;

  double *h_A       = (double *)malloc(memsize);
  double *h_answer  = (double *)malloc(sizeof(double));

  for (int i = 0; i < numElements; ++i){
    h_A[i] = numElements*double(rand())/RAND_MAX;
  }

  clock_t timer_h;
  timer_h=clock();
  CPU_answer = find_maximum_CPU(h_A, numElements);
  timer_h = clock()-timer_h;
  timer_h = (double) timer_h/CLOCKS_PER_SEC*1000.0;

  double *d_A, *d_answer; int *d_mutex;
  cudaMalloc((void **)&d_A,memsize);
  cudaMalloc((void **)&d_answer,sizeof(double));
  cudaMemset(d_answer, 0, sizeof(float));
  cudaMalloc((void **)&d_mutex, sizeof(int));
	cudaMemset(d_mutex, 0, sizeof(float));

  dim3 threadsPerBlock = 256;
  dim3 blocksPerGrid   = 256; //(numElements + threadsPerBlock - 1)/threadsPerBlock;

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

  std::cout  << "The gpu took: "  <<  gpu_elapsed_time  <<  " milli-seconds"  <<  std::endl;
  std::cout  << " GPU Answer : "  <<  h_answer[0]       << '\n';
  std::cout  << "The cpu took: "  <<  timer_h           <<  " milli-seconds"  <<  std::endl;
  std::cout  << " CPU Answer : "  <<  CPU_answer        << '\n';



  cudaFree(d_A); cudaFree(d_answer); cudaFree(d_mutex);
  free(h_A); free(h_answer);

  return 0;
}
