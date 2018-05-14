#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

__global__ void find_maximum_kernel(double *array, double *max, int *mutex, unsigned int numElements);

double find_maximum_CPU(double *array, unsigned int numElements);

#endif
