#ifndef g
#define g 127267.2 //Gravity, 9.82*(3.6)^2*1000 in [km / hr^2]
#endif
#ifndef __FUNCTIONSH__
#define __FUNCTIONSH__
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>
using namespace std;

void load_initial_data(double *H,double *HU,double *HV,double *Zdx,double *Zdy,
string datapath,int nx,int Size,float Tend,int numElements);

void load_initial_state(string filename, double * H, int numElements);

double update_dt(const double *H, const double *HU,
                 const double *HV,double *dt, double dx, int numElements);

void cpy_to(double *target, const double *source, int numElements);

int to_idx(int x, int y, int nx);

void enforce_BC(double *Ht, double *HUt, double *HVt, int nx);

void FV_time_step( double *H,        double *HU,        double *HV,
               const double *Zdx, const double *Zdy,
               const double *Ht,  const double *HUt, const double *HVt,
               double C,          double dt,         int nx);

void impose_tolerances(double *Ht, double *HUt, double *HVt, int numElements);

void display(double *A, int nx);

#endif
