#include "kernels.cuh"
#include <stdio.h>
#include <assert.h>

__device__ void copy_temp_variables(double *d_H, double *d_HU, double *d_HV,
	double *d_Ht, double *d_HUt, double *d_HVt, int nx){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int offset = 0;
	int stride = gridDim.x * blockDim.x;

	while(idx + offset < nx*nx){
		d_Ht[idx] 	= d_H[idx];
		d_HUt[idx]	= d_HU[idx];
		d_HVt[idx]	= d_HV[idx];
		offset += stride;
	}
}

__global__ void call_enforce_BC_kernel(double *d_Ht, double *d_HUt, double *d_HVt,
	int nx){
  enforce_BC_device(d_Ht,d_HUt,d_HVt,nx);
}

__device__ void enforce_BC_device(double *d_Ht, double *d_HUt, double *d_HVt,
	int nx){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int y = (idx)/nx;
	unsigned int x = (idx)%nx;
	int offset = 0;
	int stride = gridDim.x * blockDim.x;

	while(idx + offset < nx*nx){
		if(x == 0		){
			d_Ht [y * (nx) + x]  = d_Ht [y		* (nx) + x+1];
			d_HUt[y * (nx) + x]  = d_HUt[y 		* (nx) + x+1];
			d_HVt[y * (nx) + x]  = d_HVt[y 		* (nx) + x+1];
		}
		if(x == nx-1){
			d_Ht [y * (nx) + x]  = d_Ht [y 		* (nx) + x-1];
			d_HUt[y * (nx) + x]  = d_HUt[y 		* (nx) + x-1];
			d_HVt[y * (nx) + x]  = d_HVt[y 		* (nx) + x-1];
		}
		if(y == 0		){
			d_Ht [y * (nx) + x]  = d_Ht [(y+1)* (nx) + x	];
			d_HUt[y * (nx) + x]  = d_HUt[(y+1)* (nx) + x	];
			d_HVt[y * (nx) + x]  = d_HVt[(y+1)* (nx) + x	];
		}
		if(y == nx-1){
			d_Ht [y * (nx) + x]  = d_Ht [(y-1)* (nx) + x	];
			d_HUt[y * (nx) + x]  = d_HUt[(y-1)* (nx) + x	];
			d_HVt[y * (nx) + x]  = d_HVt[(y-1)* (nx) + x	];
		}
		offset += stride;
		y = (idx+offset)/nx;
		x = (idx+offset)%nx;
	}
	__syncthreads();
	//Ensuring Corners are done right (Small bottleneck)
	if(idx == 0){ //down left (0,0)
		d_Ht [0]  = d_Ht [1];
		d_HUt[0]  = d_HUt[1];
		d_HVt[0]  = d_HVt[1];
	}
	if(idx == 1){ //down right (nx-1,0)
		d_Ht [nx-1] = d_Ht [nx-2];
		d_HUt[nx-1] = d_HUt[nx-2];
		d_HVt[nx-1] = d_HVt[nx-2];
	}
	if(idx == 2){ //up left (0,nx-1)
		d_Ht [(nx-1)*nx]  = d_Ht [(nx-2)*nx];
		d_HUt[(nx-1)*nx]  = d_HUt[(nx-2)*nx];
		d_HVt[(nx-1)*nx]  = d_HVt[(nx-2)*nx];
	}
	if(idx == 3){ //up right (nx-1,nx-1)
		d_Ht [(nx-1)*nx + (nx-1)]  = d_Ht [(nx-2)*nx + (nx-2)];
		d_HUt[(nx-1)*nx + (nx-1)]  = d_HUt[(nx-2)*nx + (nx-2)];
		d_HVt[(nx-1)*nx + (nx-1)]  = d_HVt[(nx-2)*nx + (nx-2)];
	}

}

__device__ void FV_iterator_device(double *d_H, double *d_HU, double *d_HV,
const double *d_Zdx, const double *d_Zdy, double *d_Ht, double *d_HUt,
double *d_HVt, double C, double dt, int nx){

	unsigned int idx		= threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int y;
	unsigned int x;
	unsigned int offset = 0;
	unsigned int stride = gridDim.x * blockDim.x;
	while(idx + offset < nx * nx){
		y = (idx+offset)/nx;
		x = (idx+offset)%nx;
		if(x > 0 && y > 0 && x < nx-1 && y < nx-1){
			//if(idx == 0 or idx == 1){printf("Idx : %d\t offset : %d\t coord : (%d,%d)\n",idx,offset,x,y);}
				d_H[y * (nx) + x] =
					0.25*( d_Ht [y * (nx) + (x+1)] + d_Ht [y 		 * (nx) + (x-1)]
								+d_Ht [(y+1) * (nx) + x] + d_Ht [(y-1) * (nx) + x])
				 +C   *( d_HUt[(y-1) * (nx) + x] - d_HUt[(y+1) * (nx) + x]
								+d_HVt[y * (nx) + (x-1)] - d_HVt[y 		 * (nx) + (x+1)]);

				d_HU[y * (nx) + x] =
					0.25*( d_HUt [y * (nx) + (x+1)] + d_HUt[y * (nx) + (x-1)]
								+d_HUt [(y+1) * (nx) + x] + d_HUt[(y-1) * (nx) + x])
				 -dt * g * d_H[y * (nx) + x]*d_Zdx[y * (nx) + x]
				 +C   *( d_HUt[y * (nx) + (x-1)]*d_HVt[y * (nx) + (x-1)]/d_Ht[y * (nx) + (x-1)]
								-d_HUt[y * (nx) + (x+1)]*d_HVt[y * (nx) + (x+1)]/d_Ht[y * (nx) + (x+1)])
				 +C   *( pow(d_HUt[(y-1) * (nx) + x],2)/d_Ht[(y-1) * (nx) + x]
								+0.5 * g * pow(d_Ht[(y-1) * (nx) + x],2)
								-pow(d_HUt[(y+1) * (nx) + x],2)/d_Ht[(y+1) * (nx) + x]
								-0.5 * g * pow(d_Ht[(y+1) * (nx) + x],2));

				d_HV[y * (nx) + x]  =
					0.25*( d_HVt[y * (nx) + (x+1)] + d_HVt[y * (nx) + (x-1)]
								+d_HVt[(y+1) * (nx) + x] + d_HVt[(y-1) * (nx) + x])
				 -dt * g * d_H[y * (nx) + x]*d_Zdy[y * (nx) + x]
				 +C   *( d_HUt[(y-1) * (nx) + x]*d_HVt[(y-1) * (nx) + x]/d_Ht[(y-1) * (nx) + x]
								-d_HUt[(y+1) * (nx) + x]*d_HVt[(y+1) * (nx) + x]/d_Ht[(y+1) * (nx) + x])
				 +C   *( pow(d_HVt[y * (nx) + (x-1)],2)/d_Ht[y * (nx) + (x-1)]
								+0.5 * g * pow(d_Ht[y * (nx) + (x-1)],2)
								-pow(d_HVt[y * (nx) + (x+1)],2)/d_Ht[y * (nx) + (x+1)]
								-0.5 * g * pow(d_Ht[y * (nx) + (x+1)],2));
				offset += stride;
		}
		offset += stride;
	}

}

__device__ void impose_tolerances_device(double *d_Ht, double *d_HUt,
double *d_HVt, int nx){
	unsigned int idx = threadIdx.x + blockIdx.x*blockDim.x;
	int offset = 0;
	int stride = gridDim.x * blockDim.x;

	while(idx + offset < nx*nx){
		if(d_Ht[idx]<0){
			d_Ht[idx] = 1e-5;
		}
		if(d_Ht[idx] <= 1e-5){
			d_HUt[idx] = 0;
			d_HVt[idx] = 0;
		}
		offset += stride;
	}
}

__global__ void FV_time_step_kernel(double *d_H, double *d_HU, double *d_HV,
const double *d_Zdx, const double *d_Zdy, double *d_Ht, double *d_HUt,
double *d_HVt, double C, double dt, int nx){

		//COPY LAST STATE IN TEMPORARY VARIABLES
		//copy_temp_variables(d_H,d_HU,d_HV,d_Ht,d_HUt,d_HVt,nx);

		//__syncthreads();

		//ENFORCE BC
		//enforce_BC_device(d_Ht,d_HUt,d_HVt,nx);

		//__syncthreads();

		//FINITE VOLUME STEP
		FV_iterator_device(d_H,d_HU,d_HV,d_Zdx,d_Zdy,d_Ht,d_HUt,d_HVt,C,dt,nx);


		//IMPOSING TOLERANCES
		impose_tolerances_device(d_Ht,d_HUt,d_HVt,nx);

}


__device__ void find_maximum_device(double *array, double *max, int *mutex,
unsigned int numElements)
{
	unsigned int index 						= threadIdx.x + blockIdx.x*blockDim.x;
	unsigned int stride 					= gridDim.x*blockDim.x;
	unsigned int offset						= 0;

	__shared__ double cache[256];
	double temp 									= -1.0;
	while((index+offset) <numElements){
		temp = fmaxf(temp, array[index+offset]);
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

__global__ void find_maximum_kernel(double *array, double *max, int *mutex, unsigned int numElements)
{
		find_maximum_device(array,max,mutex,numElements);
}



__global__ void update_dt_kernel(const double *H, const double *HU, const double *HV, int *mutex,
  double* dt,double dx,int numElements){

 	 	unsigned int index 						= threadIdx.x + blockIdx.x*blockDim.x;
 	 	unsigned int stride 					= gridDim.x*blockDim.x;
 	 	unsigned int offset						= 0;
		double 			 mu 							= 0.0;
		double 			 temp 						= -1;
		__shared__ double cache[256];

		while((index+offset)<numElements){
	    mu = sqrt(pow(fmaxf(abs(HU[index+offset]/H[index+offset]-sqrt(H[index+offset]*g)),
													abs(HU[index+offset]/H[index+offset]+sqrt(H[index+offset]*g))),2)
	             +pow(fmaxf(abs(HV[index+offset]/H[index+offset]-sqrt(H[index+offset]*g)),
							 						abs(HV[index+offset]/H[index+offset]+sqrt(H[index+offset]*g))),2));
			temp = fmaxf(mu,temp);
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
			*dt = dx/(sqrt(2.0)*cache[0]);
	 		atomicExch(mutex, 0);  //unlock
	 		}
}


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
