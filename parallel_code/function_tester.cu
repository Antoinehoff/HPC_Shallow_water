#include<iostream>
#include "kernels.cuh"
#include "tests.cuh"
#include "functions.h"

using namespace std;


int main(){
//  int nBlocks   = 256;
//  int nThreads  = 256;
  test_max();
//  test_update_dt();
//  test_enforce_bc(nBlocks,nThreads);
  return 0;
}
