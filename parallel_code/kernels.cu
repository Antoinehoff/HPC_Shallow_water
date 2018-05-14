#include "kernels.cuh"

__global__ void find_maximum_kernel(double *array, double *max, int *mutex, unsigned int numElements)
{
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride = gridDim.x*blockDim.x;
	unsigned int offset = 0;
	__shared__ double cache[256];
	double temp = -1.0;
	while(index + offset < numElements){
		temp = fmaxf(temp, array[index + offset]);
		offset += stride;
	}
	cache[threadIdx.x] = temp;
	__syncthreads();
	// reduction
	unsigned int i = blockDim.x/2;
	while(i != 0){
		if(threadIdx.x < i){
			cache[threadIdx.x] = fmaxf(cache[threadIdx.x], cache[threadIdx.x + i]);
		}
		__syncthreads();
		i /= 2;
	}
	if(threadIdx.x == 0){
		while(atomicCAS(mutex,0,1) != 0);  //lock
		*max = fmaxf(*max, cache[0]);
		atomicExch(mutex, 0);  //unlock
	}
}

/*
__global__ void update_dt_kernel(const double *H, const double *HU, const double *HV,
                                 double* dt,      double dx,        int numElements){
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
     *dt = dx/(sqrt(2.0)*mu);
}
*/
//----------------------------------------

double find_maximum_CPU(double *array, unsigned int numElements){
  double max = -1.0;
  int i = 0;
  while(i<numElements){
    max = fmaxf(max, array[i]);
    ++i;
  }
  return max;
}
