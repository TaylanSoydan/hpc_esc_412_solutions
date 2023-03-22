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

using namespace blitz;


float w_cic(float s) {
    float abs_s = abs(s);
    if (abs_s < 1) {
        return 1 - abs_s;
    } else {
        return 0.0;
    }
}

float w_tsc(float s) {
    float abs_s = abs(s);
    if (abs_s < 0.5) {
        return 0.75 - pow(abs_s, 2);
    } else if (abs_s < 1.5) {
        return 0.5 * pow(1.5 - abs_s, 2);
    } else {
        return 0.0;
    }
}

float w_pcs(float s) {
    float abs_s = abs(s);
    if (abs_s < 1.0) {
        return (0.1666666667 * (4 - 6 * abs_s * abs_s + 3 * abs_s * abs_s * abs_s));
    } else if (abs_s < 2.0) {
        return (0.1666666667 * (2 - abs_s * abs_s * abs_s));
    } else {
        return 0.0;
    }
}

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
    //int N = io.count(); //////////////////////////////////////////////////////////

    // Load particle positions
    std::cerr << "Loading " << N << " particles" << std::endl;
    Array<float,2> r(N,3);
    io.load(r);
    auto end_read = std::chrono::high_resolution_clock::now();
    // Create Mass Assignment Grid
    auto start_assign = std::chrono::high_resolution_clock::now();
    
    //Array<float,3> grid(nGrid,nGrid,nGrid);
    // Allocate memory externally //////////////////////////////////////////////////////////
    float* data = new (std::align_val_t(64)) float[nGrid*nGrid*(nGrid+2)]; // 512-bit alignment
    blitz::TinyVector<int, 3> shape(nGrid, nGrid, nGrid+2); // Shape of the array
    blitz::Array<float, 3> myArray(data, shape, blitz::neverDeleteData); 
    // Create a subarray view with dimensions N × N × N
    blitz::Range all = blitz::Range::all();
    blitz::Array<float, 3> subArray = myArray(all, all, blitz::Range(0, nGrid));
    // Create complex array
    blitz::Array<std::complex<float>,3> kdata(reinterpret_cast<std::complex<float>*>(data), blitz::shape(nGrid,nGrid,nGrid/2+1));
    //////////////////////////////////////////////////////////


    float sx, sy, sz;
    float icenter,jcenter,kcenter;
    float Wx, Wy, Wz,W;
    int i_new, j_new, k_new;
    #pragma omp parallel for
    for(int pn=0; pn<N; ++pn) { 
        float rx = r(pn, 0) + 0.5;
        float ry = r(pn, 1) + 0.5;
        float rz = r(pn, 2) + 0.5;

        int istart = (int) floor(rx*nGrid - 0.5);
        int jstart = (int) floor(ry*nGrid - 0.5);
        int kstart = (int) floor(rz*nGrid - 0.5);
       
        for(int i = istart ; i< istart +2; i++) {
            for(int j= jstart ; j< jstart +2; j++) {
                for (int k= kstart ; k< kstart +2; k++) {
                    
                    icenter = i + 0.5; // cell center
                    sx = rx*nGrid - icenter; // distance from cell center to the particle
                    Wx = w_cic(sx); // Weight in x dimension , then do the same for y and z
                    
                    jcenter = j + 0.5; // cell center
                    sy = ry*nGrid - jcenter;
                    Wy = w_cic(sy); // Weight in x dimension , then do the same for y and z
                    
                    kcenter = k + 0.5; // cell center
                    sz = rz*nGrid - kcenter;
                    Wz = w_cic(sz); // Weight in x dimension , then do the same for y and z
                    W = Wx * Wy * Wz; // total weight
                    i_new = i;
                    j_new = j;
                    k_new = k;
                    if (i < 0)  {
                        i_new = i + nGrid;
                    }
                    if (i > nGrid - 1)  {
                        i_new = i - nGrid;
                    }
                    if (j < 0)  {
                        j_new = j + nGrid;
                    }
                    if (j > nGrid - 1)  {
                        j_new = j - nGrid;
                    }                    
                    if (k < 0)  {
                        k_new = k + nGrid;
                    }
                    if (k > nGrid - 1)  {
                        k_new = k - nGrid;
                    }
                    #pragma omp atomic
                    subArray(i_new,j_new,k_new)+=W; //////////////////////////////////////////////////////////
        }}
        
        }

        }

	
    
   auto end_assign = std::chrono::high_resolution_clock::now();
   auto start_project = std::chrono::high_resolution_clock::now();
   Array<float,2>projected(nGrid,nGrid);
    for (int i=0; i<100; i++) {
        for (int j=0; j<100; j++) {
            float max = 0;
            for (int k=0; k<100; k++){
                if (subArray(i,j,k)>max) { //////////////////////////////////////////////////////////
                    max = subArray(i,j,k); //////////////////////////////////////////////////////////
                }
            projected(i,j)=max;
	}
	}
	}
    int dims[3] = {nGrid, nGrid, nGrid+2};
   //thirdIndex k;
   //projected = blitz::max(grid,k);
   const int BATCH = 10; //////////////////////////////////////////////////////////
    fftwf_plan plan = fftwf_plan_dft_r2c_3d(nGrid,nGrid,nGrid,data,         // rank
                                             // dims,        // dimensions
                                             // BATCH,     // number of transforms
                                              //data, //, NULL, 1, nGrid * nGrid, // input parameters
                                              reinterpret_cast<fftwf_complex*>(kdata.data()), // output parameters
                                              FFTW_MEASURE); // flags
    fftwf_execute(plan);

    // Destroy plan
    fftwf_destroy_plan(plan); //////////////////////////////////////////////////////////


   auto end_project = std::chrono::high_resolution_clock::now();
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

   delete[] data; //////////////////////////////////////////////////////////

   return 0;
}
