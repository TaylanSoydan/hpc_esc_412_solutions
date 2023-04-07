// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <cstdint>
#include <stdlib.h>
#include "blitz/array.h"
#include "tipsy.h"
#include <chrono>
#include <omp.h>
#include <new>
#include <complex>
#include <fftw3.h>
#include "weights.h"
using namespace blitz;


int main(int argc, char *argv[]) {
    auto start_read = std::chrono::high_resolution_clock::now();
    if (argc<=1) {
        std::cerr << "Usage: " << argv[0] << " tipsyfile.std [grid-size]"
                  << std::endl;
        return 1;
    }

    int nGrid = 100;
    if (argc>2) nGrid = atoi(argv[2]);

    TipsyIO io;
    io.open(argv[1]);
    if (io.fail()) {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
        return errno;
    }
    std::uint64_t N = io.count();
    // int N = io.count(); 
    // Load particle positions
    std::cerr << "Loading " << N << " particles" << std::endl;
    Array<float,2> r(N,3);
    io.load(r);
    auto end_read = std::chrono::high_resolution_clock::now();
    // Create Mass Assignment Grid

    Array<float,3> grid_pcs(nGrid,nGrid,nGrid);
    // Allocate memory externally 
    float* data = new (std::align_val_t(64)) float[nGrid*nGrid*(nGrid+2)]; // 512-bit alignment
    blitz::TinyVector<int, 3> shape(nGrid, nGrid, nGrid+2); // Shape of the array
    blitz::Array<float, 3> myArray(data, shape, blitz::neverDeleteData); 
    // Create a subarray view with dimensions N × N × N
    blitz::Range all = blitz::Range::all();
    blitz::Array<float, 3> subArray = myArray(all, all, blitz::Range(0, nGrid));
    // Create complex array
    blitz::Array<std::complex<float>,3> kdata(reinterpret_cast<std::complex<float>*>(data), blitz::shape(nGrid,nGrid,nGrid/2+1));
    
    auto start_assign = std::chrono::high_resolution_clock::now();

    //PCS
    #pragma omp parallel for
    for(int pn=0; pn<N; pn++) { 
        float rx = r(pn, 0) + 0.5;
        float ry = r(pn, 1) + 0.5;
        float rz = r(pn, 2) + 0.5;
        rx *= nGrid;
        ry *= nGrid;
        rz *= nGrid;
        float Wx[4];
        float Wy[4];
        float Wz[4];
        int ix = pcs_weights(rx, Wx);
        int iy = pcs_weights(ry, Wy);
        int iz = pcs_weights(rz, Wz);
        for(int i = 0 ; i< 4; i++) {
            for(int j= 0 ; j < 4; j++) {
                for (int k= 0 ; k < 4; k++) {
                    int I = (ix + i + nGrid) % nGrid;
                    int J = (iy + j + nGrid) % nGrid;
                    int K = (iz + k + nGrid) % nGrid;
                    float W = Wx[i] * Wy[j] * Wz[k];
                    #pragma omp atomic
                    grid_pcs(I,J,K)+=W; 

        }}}}
    auto sum_pcs = blitz::sum(grid_pcs);
    std::cout << "pcs total = " << sum_pcs << "\n";
    
   auto end_assign = std::chrono::high_resolution_clock::now();
   auto start_project = std::chrono::high_resolution_clock::now();
   Array<float,2>projected(nGrid,nGrid);
   thirdIndex k;
   projected = blitz::max(grid_pcs,k);
   auto end_project = std::chrono::high_resolution_clock::now();
   const int BATCH = 10; 
    fftwf_plan plan = fftwf_plan_dft_r2c_3d(nGrid,nGrid,nGrid,data,         // rank
                                              reinterpret_cast<fftwf_complex*>(kdata.data()), // output parameters
                                              FFTW_MEASURE); // flags
    fftwf_execute(plan);
    // Destroy plan
    fftwf_destroy_plan(plan); 

   auto elapsed_read = std::chrono::duration_cast<std::chrono::nanoseconds>(end_read - start_read).count();
   std::cout << "Reading file took: " << elapsed_read/1e9 << " seconds\n";
   auto elapsed_assign = std::chrono::duration_cast<std::chrono::nanoseconds>(end_assign - start_assign).count();
   std::cout << "Mass assignment took: " << elapsed_assign/1e9 << " seconds\n";
   auto elapsed_project = std::chrono::duration_cast<std::chrono::nanoseconds>(end_project - start_project).count();
   std::cout << "Projection took: " << elapsed_project/1e9 << " seconds\n";
   std::ofstream fout("fout.txt");
   if (fout.is_open()) {
	   for (int i = 0; i<nGrid; i++) {
		   for (int j = 0; j<nGrid; j++) {
			   fout << projected(i,j) << ",";
		   }
	   }
   }
   fout.close();

   delete[] data; 

   return 0;
}
