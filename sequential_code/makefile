OPTIM+=-Ofast -march=native
CXX=g++
CC=g++
LD=${CXX}
CXXFLAGS+=-Wall -Wextra -std=c++11 $(OPTIM)
LDFLAGS+=-lm
EXE=compute.exe
objects=compute.o functions.o

all: clean compute

compute : $(objects)
	$(LD) -o $(EXE) $(objects) $(LDFLAGS) $(OPTIM)

compute_profiling : compute_profiling.o functions.o
	$(LD) -o compute_profiling.exe compute_profiling.o functions.o $(LDFLAGS) $(OPTIM)

functions.o : functions.cpp
	$(LD) -c functions.cpp -std=c++11 $(LDFLAGS) $(OPTIM)

compute.o : compute.cpp
	$(LD) -c compute.cpp -std=c++11 $(LDFLAGS) $(OPTIM)

compute_profiling.o : compute_profiling.cpp
		$(LD) -c compute_profiling.cpp -std=c++11 $(LDFLAGS) $(OPTIM)

function_tester : clean function_tester.o
		$(LD) -o function_tester.exe function_tester.o $(LDFLAGS) $(OPTIM)


clean:
	rm -f $(EXE) *.o *~
