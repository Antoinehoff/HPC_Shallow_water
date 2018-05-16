#include "kernels.cuh"

__device__ void enforce_BC_device(double *d_Ht, double *d_HUt, double *d_HVt,
	int nx, int x, int y){
	if(x == 0		){
		d_Ht [y * (nx) + x]  = d_Ht [y		* (nx) + x+1];
		d_HUt[y * (nx) + x]  = d_HUt[y 		* (nx) + x+1];
		d_HVt[y * (nx) + x]  = d_HVt[y 		* (nx) + x+1];
	}
	__syncthreads();
	if(x == nx-1){
		d_Ht [y * (nx) + x]  = d_Ht [y 		* (nx) + x-1];
		d_HUt[y * (nx) + x]  = d_HUt[y 		* (nx) + x-1];
		d_HVt[y * (nx) + x]  = d_HVt[y 		* (nx) + x-1];
	}
	__syncthreads();
	if(y == 0		){
		d_Ht [y * (nx) + x]  = d_Ht [(y+1)* (nx) + x	];
		d_HUt[y * (nx) + x]  = d_HUt[(y+1)* (nx) + x	];
		d_HVt[y * (nx) + x]  = d_HVt[(y+1)* (nx) + x	];
	}
	__syncthreads();
	if(y == nx-1){
		d_Ht [y * (nx) + x]  = d_Ht [(y-1)* (nx) + x	];
		d_HUt[y * (nx) + x]  = d_HUt[(y-1)* (nx) + x	];
		d_HVt[y * (nx) + x]  = d_HVt[(y-1)* (nx) + x	];
	}
	__syncthreads();
}

__global__ void FV_time_step_kernel(double *d_H, double *d_HU, double *d_HV,
const double *d_Zdx, const double *d_Zdy, double *d_Ht, double *d_HUt,
double *d_HVt, double C, double dt, int nx){

	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int y = idx/nx;
	unsigned int x = idx%nx;

	//Check if the thread is inside the domain
	//if(x < nx and y < nx){
		//COPY LAST STATE IN TEMPORARY VARIABLES
		/*
			d_Ht[idx] 	= d_H[idx];
			d_HUt[idx]	= d_HU[idx];
			d_HVt[idx]	= d_HV[idx];
		*/
		__syncthreads();

		//ENFORCE BC
		enforce_BC_device(d_Ht,d_HUt,d_HVt,nx,x,y);


		__syncthreads();

		//FINITE VOLUME STEP
		if(x > 0 and y > 0 and x < nx-1 and y < nx-1){
			d_H[y * (nx) + x]=
				0.25*( d_Ht[y * (nx) + (x+1)]+d_Ht[y * (nx) + (x-1)]
							+d_Ht[(y+1) * (nx) + x]+d_Ht[(y-1) * (nx) + x])+
				C   *( d_HUt[(y-1) * (nx) + x]-d_HUt[(y+1) * (nx) + x]
							+d_HVt[y * (nx) + (x-1)]-d_HVt[y * (nx) + (x+1)]);

			d_HU[y * (nx) + x]=
				0.25*( d_HUt[y * (nx) + (x+1)]+d_HUt[y * (nx) + (x-1)]
							+d_HUt[(y+1) * (nx) + x]+d_HUt[(y-1) * (nx) + x])
							-dt*g*d_H[y * (nx) + x]*d_Zdx[y * (nx) + x]
			 +C   *( pow(d_HUt[(y-1) * (nx) + x],2)/d_Ht[(y-1) * (nx) + x]
							+0.5*g*pow(d_Ht[(y-1) * (nx) + x],2)
							-pow(d_HUt[(y+1) * (nx) + x],2)/d_Ht[(y+1) * (nx) + x]
							-0.5*g*pow(d_Ht[(y+1) * (nx) + x],2))
			 +C   *( d_HUt[y * (nx) + (x-1)]*d_HVt[y * (nx) + (x-1)]/d_Ht[y * (nx) + (x-1)]
							-d_HUt[y * (nx) + (x+1)]*d_HVt[y * (nx) + (x+1)]/d_Ht[y * (nx) + (x+1)]);

			d_HV[y * (nx) + x]  =
				0.25*( d_HVt[y * (nx) + (x+1)]+d_HVt[y * (nx) + (x-1)]
							+d_HVt[(y+1) * (nx) + x]+d_HVt[(y-1) * (nx) + x])
							-dt*g*d_H[y * (nx) + x]*d_Zdy[y * (nx) + x]
			 +C   *( d_HUt[(y-1) * (nx) + x]*d_HVt[(y-1) * (nx) + x]/d_Ht[(y-1) * (nx) + x]
							-d_HUt[(y+1) * (nx) + x]*d_HVt[(y+1) * (nx) + x]/d_Ht[(y+1) * (nx) + x])
			 +C   *( pow(d_HVt[y * (nx) + (x-1)],2)/d_Ht[y * (nx) + (x-1)]
							+0.5*g*pow(d_Ht[y * (nx) + (x-1)],2)
							-pow(d_HVt[y * (nx) + (x+1)],2)/d_Ht[y * (nx) + (x+1)]
							-0.5*g*pow(d_Ht[y * (nx) + (x+1)],2));
		}
		__syncthreads();

		//IMPOSING TOLERANCES
/*
		if(d_Ht[idx]<0){
			d_Ht[idx] = 1e-5;
		}
		if(d_Ht[idx] <= 1e-5){
			d_HUt[idx] = 0;
			d_HVt[idx] = 0;
		}
*/
	}
}

__device__ void find_maximum_device(double *array, double *max, int *mutex,
unsigned int numElements)
{
	unsigned int modifiedgridDim = gridDim.x;
	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride 					= modifiedgridDim*blockDim.x;

	__shared__ double cache[256];
	double temp 									= -1.0;

	for(int i = index; i<numElements; i += stride){
		temp = fmaxf(temp, array[i]);
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

__global__ void find_maximum_kernel(double *array, double *max, int *mutex, unsigned int numElements)
{
//	unsigned int index = threadIdx.x + blockIdx.x*blockDim.x;
//	if(index % 2 == 0 and index < numElements){
		find_maximum_device(array,max,mutex,numElements);
//	}
}


/*
__device__ void update_dt_kernel(const double *H, const double *HU, const double *HV,
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

void copy_host2device(double * d_Ht,double * d_HUt,double * d_HVt,double * Ht,
double * HUt,double * HVt,size_t memsize){
  cudaMemcpy(d_Ht,   Ht,    memsize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_HUt,  HUt,   memsize,  cudaMemcpyHostToDevice);
  cudaMemcpy(d_HVt,  HVt,   memsize,  cudaMemcpyHostToDevice);
}

void copy_device2host(double * H,double * HU,double * HV,double * d_H,
double * d_HU,double * d_HV,size_t memsize){
  cudaMemcpy(H,     d_H,  memsize,  cudaMemcpyDeviceToHost);
  cudaMemcpy(HU,    d_HU, memsize,  cudaMemcpyDeviceToHost);
  cudaMemcpy(HV,    d_HV, memsize,  cudaMemcpyDeviceToHost);
}
