#include "functions.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>
#include <string>
#include <sstream>

using namespace std;

void load_initial_data(double *H,double *HU,double *HV,double *Zdx,double *Zdy,
string datapath,int nx,int Size,float Tend,int numElements){
  // Filestream variables
  string  identificator;                              // Identificator of the simulation
  ostringstream sfilename;                            // Identificator auxiliary
  // Data filename :
  sfilename <<  datapath <<  "Data_nx" <<  to_string(nx) <<  "_"
            <<  to_string(Size) <<  "km_T"  <<  Tend  <<  setprecision(2);
  identificator = sfilename.str();
  cout << " Simulation identificator : "  <<  identificator  << endl;
  // Load initial condition from data files
  cout <<" Loading data on host memory.." << endl;
  load_initial_state(identificator + "_h.bin",   H,    numElements);
  load_initial_state(identificator + "_hu.bin",  HU,   numElements);
  load_initial_state(identificator + "_hv.bin",  HV,   numElements);
  // Load topography slopes from data files
  load_initial_state(identificator + "_Zdx.bin", Zdx,  numElements);
  load_initial_state(identificator + "_Zdy.bin", Zdy,  numElements);
}

void load_initial_state(string filename, double * H, int numElements){
  ifstream fin;
  fin.open(filename, ios::in|ios::binary);
  if(!fin){
    cerr<<" Error, couldn't find file : "<<filename<<endl;
    exit(EXIT_FAILURE);
  }
  fin.read(reinterpret_cast<char*>(&H[0]), numElements*sizeof(double));
  fin.close();
}

double update_dt(const double *H, const double *HU,
                 const double *HV, double *dt, double dx, int numElements){
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
    dt[0] = dx/(sqrt(2.0)*mu);
  }

void cpy_to(double *target, const double *source, int numElements){
  for(int i=0; i<numElements; i++){
      target[i] = source[i];
  }
}

int to_idx(int x, int y, int nx){
  return y * (nx) + x;
}

void enforce_BC(double *Ht, double *HUt, double *HVt, int nx){

  for(int i=0; i<nx; ++i){
    Ht  [to_idx(0,    i,    nx)] = Ht  [to_idx(1,     i,    nx)];
    HUt [to_idx(0,    i,    nx)] = HUt [to_idx(1,     i,    nx)];
    HVt [to_idx(0,    i,    nx)] = HVt [to_idx(1,     i,    nx)];
  }
  for(int i=0; i<nx; ++i){
    Ht  [to_idx(nx-1, i,    nx)]  = Ht  [to_idx(nx-2, i,    nx)];
    HUt [to_idx(nx-1, i,    nx)]  = HUt [to_idx(nx-2, i,    nx)];
    HVt [to_idx(nx-1, i,    nx)]  = HVt [to_idx(nx-2, i,    nx)];
  }
  for(int i=0; i<nx; ++i){
    Ht  [to_idx(i,    0,    nx)]  = Ht  [to_idx(i,    1,    nx)];
    HUt [to_idx(i,    0,    nx)]  = HUt [to_idx(i,    1,    nx)];
    HVt [to_idx(i,    0,    nx)]  = HVt [to_idx(i,    1,    nx)];
  }
  for(int i=0; i<nx; ++i){
    Ht  [to_idx(i,    nx-1, nx)]  = Ht  [to_idx(i,    nx-2, nx)];
    HUt [to_idx(i,    nx-1, nx)]  = HUt [to_idx(i,    nx-2, nx)];
    HVt [to_idx(i,    nx-1, nx)]  = HVt [to_idx(i,    nx-2, nx)];
  }
}

void FV_time_step( double *H, double *HU,double *HV,const double *Zdx,
                  const double *Zdy,const double *Ht,  const double *HUt,
                  const double *HVt,double C,double dt,int nx){
  for(int x=1; x<nx-1; x++){
    for(int y=1; y<nx-1; y++){
      H[to_idx(x,y,nx)]=
        0.25*( Ht[to_idx(x+1,y,nx)]+Ht[to_idx(x-1,y,nx)]
              +Ht[to_idx(x,y+1,nx)]+Ht[to_idx(x,y-1,nx)])+
        C   *( HUt[to_idx(x,y-1,nx)]-HUt[to_idx(x,y+1,nx)]
              +HVt[to_idx(x-1,y,nx)]-HVt[to_idx(x+1,y,nx)]);

      HU[to_idx(x,y,nx)]=
        0.25*( HUt[to_idx(x+1,y,nx)]+HUt[to_idx(x-1,y,nx)]
              +HUt[to_idx(x,y+1,nx)]+HUt[to_idx(x,y-1,nx)])
              -dt*g*H[to_idx(x,y,nx)]*Zdx[to_idx(x,y,nx)]
       +C   *( pow(HUt[to_idx(x,y-1,nx)],2)/Ht[to_idx(x,y-1,nx)]
              +0.5*g*pow(Ht[to_idx(x,y-1,nx)],2)
              -pow(HUt[to_idx(x,y+1,nx)],2)/Ht[to_idx(x,y+1,nx)]
              -0.5*g*pow(Ht[to_idx(x,y+1,nx)],2))
       +C   *( HUt[to_idx(x-1,y,nx)]*HVt[to_idx(x-1,y,nx)]/Ht[to_idx(x-1,y,nx)]
              -HUt[to_idx(x+1,y,nx)]*HVt[to_idx(x+1,y,nx)]/Ht[to_idx(x+1,y,nx)]);

      HV[to_idx(x,y,nx)]  =
        0.25*( HVt[to_idx(x+1,y,nx)]+HVt[to_idx(x-1,y,nx)]
              +HVt[to_idx(x,y+1,nx)]+HVt[to_idx(x,y-1,nx)])
              -dt*g*H[to_idx(x,y,nx)]*Zdy[to_idx(x,y,nx)]
       +C   *( HUt[to_idx(x,y-1,nx)]*HVt[to_idx(x,y-1,nx)]/Ht[to_idx(x,y-1,nx)]
              -HUt[to_idx(x,y+1,nx)]*HVt[to_idx(x,y+1,nx)]/Ht[to_idx(x,y+1,nx)])
       +C   *( pow(HVt[to_idx(x-1,y,nx)],2)/Ht[to_idx(x-1,y,nx)]
              +0.5*g*pow(Ht[to_idx(x-1,y,nx)],2)
              -pow(HVt[to_idx(x+1,y,nx)],2)/Ht[to_idx(x+1,y,nx)]
              -0.5*g*pow(Ht[to_idx(x+1,y,nx)],2));
    }
  }
}

void impose_tolerances(double *Ht, double *HUt, double *HVt, int numElements){
  for(int i=0; i<numElements; i++){
    if(Ht[i]<0){
      Ht[i] = 1e-5;
    }
    if(Ht[i]<= 1e-5){
      HUt[i] = 0;
      HVt[i] = 0;
    }
  }
}

void display(double *A, int nx){
  cout<<endl;
  for(int i=0; i<nx; i++){
    for(int j=0; j<nx; j++){
      cout<<A[to_idx(i,j,nx)]<<"\t";
    }
    cout<<endl;
  }
  cout<<endl;
}
