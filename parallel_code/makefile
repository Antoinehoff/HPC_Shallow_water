CXX=nvcc
CC=nvcc
LD=${CXX}
CXXFLAGS+=-std=c++11
LDFLAGS+=-lm
EXE=compute_parallel.exe

all: clean compute_parallel


compute_parallel :
	$(LD) compute_parallel.cu -o ${EXE} ${CXXFLAGS} $(LDFLAGS)

clean:
	rm -f $(EXE) *.o *~
