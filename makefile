OPTIM+=-O3 -march=native
CXX=g++
CC=g++
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -std=c++11 $(OPTIM)
LDFLAGS+=-lm
EXE=compute.exe

all: clean compute

compute : compute.o
	$(LD) -o $(EXE) compute.o $(LDFLAGS)


clean:
	rm -f $(EXE) *.o *~
