// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <cstdint>
#include <stdlib.h>
#include "blitz/array.h"
#include "tipsy.h"
#include <chrono>
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
        return (1.5 * pow(abs_s, 3) - 2.5 * pow(abs_s, 2) + 1.0);
    } else if (abs_s < 2.0) {
        return (-0.5 * pow(abs_s, 3) + 2.5 * pow(abs_s, 2) - 4.0 * abs_s + 2.0);
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

    // Load particle positions
    std::cerr << "Loading " << N << " particles" << std::endl;
    Array<float,2> r(N,3);
    io.load(r);
    auto end_read = std::chrono::high_resolution_clock::now();
    // Create Mass Assignment Grid
    auto start_assign = std::chrono::high_resolution_clock::now();
    Array<float,3> grid(nGrid,nGrid,nGrid);

    float sx, sy, sz;
    float icenter,jcenter,kcenter;
    float Wx, Wy, Wz,W;
    for(int pn=0; pn<N; ++pn) { 
        float rx = r(pn, 0) + 0.5;
        float ry = r(pn, 1) + 0.5;
        float rz = r(pn, 2) + 0.5;

        int istart = (int) floor(rx*nGrid - 1.0);
        int jstart = (int) floor(ry*nGrid - 1.0);
        int kstart = (int) floor(rz*nGrid - 1.0);

        for(int i = istart ; i< istart +3; i++) {
            for(int j= jstart ; j< jstart +3; j++) {
                for (int k= kstart ; k< kstart +3; k++) {
                    icenter = i + 0.5; // cell center
                    sx = icenter/nGrid - rx; // distance from cell center to the particle
                    Wx = w_tsc(sx); // Weight in x dimension , then do the same for y and z
                    jcenter = j + 0.5; // cell center
                    sy = jcenter/nGrid - ry; // distance from cell center to the particle
                    Wy = w_tsc(sy); // Weight in x dimension , then do the same for y and z
                    kcenter = k + 0.5; // cell center
                    sz = kcenter/nGrid - rz; // distance from cell center to the particle
                    Wz = w_tsc(sz); // Weight in x dimension , then do the same for y and z
                    W = Wx * Wy * Wz; // total weight
                    //i = (i + nGrid ) % nGrid; //periodic boundary 
                    //j = (j + nGrid ) % nGrid; //periodic boundary
                    //k = (k + nGrid ) % nGrid; //periodic boundary
                    std::cout << "    pn = " << pn << " rx: " << rx << " ry: " << ry << " rz: " << rz << " istart: " << istart << " jstart" << jstart << " kstart" << kstart;
                    std::cout << " i " << i << " j" << j << " k" << k;
            		std::cout << " icenter: " << icenter << " jcenter: " << jcenter << " kcenter: " << kcenter << " sx "  << sx << " sy " << sy << " sz " << sz << " Wx:   " << Wx << " Wy" << Wy << " Wz " << Wz << "\n";
                    if (i < 0) {
                        i += nGrid;
                    }
                    if (j < 0) {
                        j += nGrid;
                    }
                    if (k < 0) {
                        k += nGrid;
                    }
                    grid(i,j,k)+=W;
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
			if (grid(i,j,k)>max) {
				max = grid(i,j,k);
			}
		projected(i,j)=max;
	}
	}
	}

   //thirdIndex k;
   //projected = blitz::max(grid,k);
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

   return 0;
}
