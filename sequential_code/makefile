OPTIM+=-O3 -march=native
CXX=g++
CC=g++
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -std=c++11 $(OPTIM)
LDFLAGS+=-lm
EXE=compute.exe

all: clean compute

compute : clean compute.o
	$(LD) -o $(EXE) compute.o $(LDFLAGS)

function_tester : clean function_tester.o
		$(LD) -o function_tester.exe function_tester.o $(LDFLAGS)


clean:
	rm -f $(EXE) *.o *~
