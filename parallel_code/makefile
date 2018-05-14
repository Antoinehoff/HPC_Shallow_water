objects= compute_parallel.o kernels.o functions.o
NVCC=nvcc
opt= -O2 -g -G -std=c++11
ARCH= -arch=sm_30
LIBS=
EXE=compute_parallel.exe
NVCCFLAGS += -D_FORCE_INLINES -ccbin=$(CXX) -Xcompiler -fPIC $(COMMON_FLAGS)

compute_parallel : $(objects)
	$(NVCC) $(opt)  -o ${EXE} $(objects) $(LIBS) ${NVCCFLAGS}

kernels.o : kernels.cu
	$(NVCC) $(opt) $(ARCH) -c kernels.cu ${NVCCFLAGS}

compute_parallel.o : compute_parallel.cu
	$(NVCC) $(opt) $(ARCH) -c compute_parallel.cu ${NVCCFLAGS}

functions.o : functions.cpp
	$(NVCC) $(opt) $(ARCH) -c functions.cpp ${NVCCFLAGS}

function_tester : function_tester kernels.o
	$(NVCC) function_tester.cu -o function_tester.exe kernels.o ${NVCCFLAGS}

clean :
	rm -f *.exe *.o *~
