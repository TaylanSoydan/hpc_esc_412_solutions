CXX = CC
CXXFLAGS = -O3 -std=c++17 -fopenmp
LDFLAGS = -lfftw3f -lfftw3f_omp -lm
INC_DIR=/users/tsoydan/usr/local/include/

assign: assign.o tipsy.o
	$(CXX) $(CXXFLAGS) -o assign assign.o tipsy.o $(LDFLAGS)

assign.o: assign.cxx tipsy.h
	$(CXX) $(CXXFLAGS) -I$(INC_DIR)  -c -o assign.o assign.cxx

tipsy.o: tipsy.cxx tipsy.h
	$(CXX) $(CXXFLAGS) -I$(INC_DIR)  -c -o tipsy.o tipsy.cxx

clean:
	rm -f assign assign.o tipsy.o
