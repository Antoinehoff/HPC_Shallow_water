#ifndef g
#define g 127267.2 //Gravity, 9.82*(3.6)^2*1000 in [km / hr^2]
#endif
#ifndef __KERNELS_CUH__
#define __KERNELS_CUH__

__device__ void enforce_BC_device(double *d_Ht, double *d_HUt, double *d_HVt,
   int nx, int x, int y);

__global__ void FV_time_step_kernel(double *d_H, double *d_HU, double *d_HV,
const double *d_Zdx, const double *d_Zdy, double *d_Ht, double *d_HUt,
double *d_HVt, double C, double dt, int nx);

__device__ void find_maximum_device(double *array, double *max, int *mutex,
  unsigned int numElements);

__global__ void find_maximum_kernel(double *array, double *max, int *mutex,
  unsigned int numElements);

__device__ void update_dt_kernel(const double *H, const double *HU,
  const double *HV,double* dt,double dx,int numElements);

 void copy_host2device(double * d_Ht,double * d_HUt,double * d_HVt,double * Ht,
   double * HUt,double * HVt,size_t memsize);
 void copy_device2host(double * H,double * HU,double * HV,double * d_H,
   double * d_HU,double * d_HV,size_t memsize);

#endif
