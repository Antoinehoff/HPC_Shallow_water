#include<iostream>

using namespace std;

__global__ void max_parallel(const double *A, double *max, int numElements){
  // Find the maximum :   -Each thread find the max of subarray
  //
  extern __shared__ double tmax_array[];
  int gdim         = gridDim.x;
  int bdim         = blockDim.x;
  int bidx         = blockIdx.x;
  int tidx         = threadIdx.x;
  int idx          = bidx * bdim + tidx;
  int Nthreads     = gridDim.x  * blockDim.x;
  int Elperthreads = numElements/Nthreads;
  float tmax       = 0;
  float bmax       = 0;

  for(int i = 0; i < Elperthreads; i++){
    if(tmax < A[Elperthreads*tidx+i]){
       tmax = A[Elperthreads*tidx+i];
    }
  }
  tmax_array[idx] = tmax;
  __syncthreads();

}

int main(){
  int numElements   = 1<<20;
  size_t memsize    = numElements * sizeof(double);
  double solution   = 10.0;

  double *h_A       = (double *)malloc(memsize);
  double *h_answer  = (double *)malloc(sizeof(double));

  for (int i = 0; i < numElements; ++i){
    h_A[i] = rand()/(double)RAND_MAX;
  }
  h_A[10] = solution;

  double *d_A, *d_answer;
  cudaMalloc((void **)&d_A,memsize);
  cudaMalloc((void **)&d_answer,sizeof(double));
  cudaMemcpy(d_A, h_A, memsize, cudaMemcpyHostToDevice);

  int threadsPerBlock = 256;
  int blocksPerGrid = (numElements + threadsPerBlock - 1)/threadsPerBlock;
  clock_t start;
  clock_t end;
  start = clock();
  max_parallel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock*sizeof(float)>>>
                                                      (d_A, d_answer, numElements);
  cudaMemcpy(d_answer, h_answer, sizeof(double), cudaMemcpyDeviceToHost);
  end = clock();

  std::cout << "Answer : " << h_answer[0] << '\n';
  std::cout << "Ellapsed time : " << (double)(end-start)/CLOCKS_PER_SEC*1000 << '\n';
  return 0;
}
