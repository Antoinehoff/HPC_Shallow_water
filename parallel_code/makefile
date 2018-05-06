objects= compute_parallel.o kernels.o
NVCC=nvcc
opt= -O2 -g -G -std=c++11
ARCH= -arch=sm_30
LIBS=
EXE=compute_parallel.exe

compute_parallel : $(objects)
	$(NVCC) $(opt)  -o ${EXE} $(objects) $(LIBS)

kernels.o : kernels.cu
	$(NVCC) $(opt) $(ARCH) -c kernels.cu

compute_parallel.o : compute_parallel.cu
	$(NVCC) $(opt) $(ARCH) -c compute_parallel.cu

function_tester : clean_2
	$(LD) function_tester.cu -o function_tester.exe ${CXXFLAGS} $(LDFLAGS)

clean :
	rm -f *.exe *.o *~
