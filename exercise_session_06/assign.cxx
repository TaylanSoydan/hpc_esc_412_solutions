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
void overdensity(Array<float, 2> &r, int N, int nGrid, Array<float, 3> &grid) {
    auto start_assign = std::chrono::high_resolution_clock::now();
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
                    //std::cout << "W = " << W;
                    #pragma omp atomic
                    grid(I,J,K)+=W; 
        }}}}
    std::cout << "mass assigned.\n";
    auto sum_pcs = blitz::sum(grid);
    std::cout << "pcs total = " << sum_pcs << "\n";
    float mean_pcs = sum_pcs / (nGrid * nGrid * nGrid);
    cout << "mean_pcs = " << mean_pcs; 
    grid -= mean_pcs;
    grid /= mean_pcs;
    std::cout << "over density calculated. \n";
    auto end_assign = std::chrono::high_resolution_clock::now();
    auto elapsed_assign = std::chrono::duration_cast<std::chrono::nanoseconds>(end_assign - start_assign).count();
    std::cout << "Mass assignment took: " << elapsed_assign/1e9 << " seconds\n";
    auto sum_overdensity = blitz::sum(grid);
    std::cout << "overdensity total = " << sum_overdensity << "\n";
}

int main(int argc, char *argv[]) {
    // Read
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

    float *data = new (std::align_val_t(64)) float[nGrid * nGrid * (nGrid + 2)];
    // Create Mass Assignment Grid
    Array<float, 3> grid_data(data, shape(nGrid, nGrid, nGrid), deleteDataWhenDone);
    Array<float, 3> grid = grid_data(Range::all(), Range::all(), Range(0, nGrid - 1));
    std::complex<float> *complex_data = reinterpret_cast<std::complex<float> *>(data);
    blitz::Array<std::complex<float>, 3> kdata(complex_data, shape(nGrid, nGrid, nGrid / 2 + 1));
    overdensity(r, N, nGrid, grid);
    //std:cout << grid << endl;
    //std:cout << kdata << endl;
    //std::cout << "Shape of array: (" << grid.shape()[0] << ", " << grid.shape()[1] << ", " << grid.shape()[2] << ")" << std::endl;

    fftwf_plan plan = fftwf_plan_dft_r2c_3d(nGrid, nGrid, nGrid, data, (fftwf_complex *)complex_data, FFTW_ESTIMATE);
    cout << "Plan created" << endl;
    fftwf_execute(plan);
    cout << "Plan executed" << endl;
    fftwf_destroy_plan(plan);
    cout << "Plan destroyed" << endl;

    std::cout << std::norm(kdata(0,0,0)) << endl;
    std::cout << "Shape of kdata: (" << kdata.shape()[0] << ", " << kdata.shape()[1] << ", " << kdata.shape()[2] << ")" << std::endl;
    //std::cout << "norm = " << std::norm(kdata(5,5,5));
    std::cout << " kdata SUM = " << blitz::sum(kdata) << endl;
    // Initialize K
    int nBins = nGrid;
    Array<float,1> fPower(nBins);
    fPower = 0.0;
    Array<int,1> nPower(nBins);
    nPower = 0;
    Array<float,1> avgPower(nBins);
    avgPower = 0.0;
    auto maxK = 0.0;
    auto maxbin = 0;
    // Calculating K and Binning with ibin = k
    for (int i = 0; i < nGrid; i++) {
        for (int j = 0; j < nGrid; j++) {
            for (int k = 0; k < nGrid/2; k++) {
            
            // Calculate kx, ky, kz
            double kx = i < nGrid/2 ? i : i - nGrid;
            double ky = j < nGrid/2 ? j : j - nGrid;
            double kz = k;

            // Calculate k
            double K = std::sqrt(kx*kx + ky*ky + kz*kz);
            double Pk = std::norm(kdata(i,j,k));

            int bin = int(K);
            if (bin >= nBins) bin = nBins - 1;
            
            // Add P(k) to the bin and increment the bin count
            fPower(bin) += Pk;
            nPower(bin) += 1;
            //std::cout << " bin = " << bin;
            if (K > maxK) maxK = K;
            if (bin > maxbin) maxbin = bin;
            
            //std::cout << " maxK = " << maxK;
            //std::cout << " i = " << i << "j = " << j << " k = " << k  << "K = " << K << " Pk = " << Pk << " bin = " << bin << "\n";
            }
        }
    }
    std::cout << " maxbin = " << maxbin;
    cout << "fPower, nPower calculated with 100 bins" << endl;
    //for (int i = maxK; i < 100; i++){
    //    std::cout << " fPower = " << fPower(i) << " nPower = " << nPower(i);
    //}
    avgPower = fPower / nPower;
    for (int i = maxbin - 1; i < nBins; i++){
        avgPower(i) = 0.0;
    }
    std::cout << " fPower SUM = " << blitz::sum(fPower) << " nPower SUM = " <<blitz::sum(nPower);
    std::ofstream outFile("bin100.txt"); // Open the output file
    // Write k and Pk arrays to file
    for (int i = 0; i < nBins; i++) {
        outFile << i << " " << avgPower(i) << std::endl;
    }
    outFile.close(); // Close the output file


    cout << "starting 80 bins" << endl;
   
    // Calculating K and Binning with ibin = (k - kmax) / nbins
    double Kmax = floor(nGrid/2 * sqrt(3));
    nBins = 80;
    Array<float,1> fPower80(nBins);
    fPower80 = 0.0;
    Array<int,1> nPower80(nBins);
    nPower80 = 0;
    Array<float,1> avgPower80(nBins);
    avgPower80 = 0.0;
    for (int i = 0; i < nGrid; i++) {
        for (int j = 0; j < nGrid; j++) {
            for (int k = 0; k < nGrid/2; k++) {
            
            // Calculate kx, ky, kz
            double kx = i < nGrid/2 ? i : i - nGrid;
            double ky = j < nGrid/2 ? j : j - nGrid;
            double kz = k;

            // Calculate k
            double K = std::sqrt(kx*kx + ky*ky + kz*kz);
            double Pk = std::norm(kdata(i,j,k));

            int bin = int( (K / Kmax) * nBins);
            if (bin >= nBins) bin = nBins - 1;
            
            // Add P(k) to the bin and increment the bin count
            fPower80(bin) += Pk;
            nPower80(bin) += 1;
            // std::cout << "80" << kx << ky << kz << K << Pk << bin;
            }
        }
    }
    cout << "fPower80, nPower80 calculated with 80 bins" << endl;
    for (int i; i < nBins; i++){
        if (nPower80(i)) avgPower80(i) = fPower80(i) / nPower80(i);;
    }
    std::ofstream outFile80("bin80.txt"); // Open the output file

    // Write k and Pk arrays to file
    for (int i = 0; i < nBins; i++) {
        outFile80 << i << " " << avgPower80(i) << std::endl;
    }

    outFile80.close(); // Close the output file

    // Log binning
    nBins = 80;
    Array<float,1> fPowerlog(nBins);
    fPowerlog = 0.0;
    Array<int,1> nPowerlog(nBins);
    nPowerlog = 0;
    Array<float,1> avgPowerlog(nBins);
    avgPowerlog = 0.0;
    for (int i = 0; i < nGrid; i++) {
        for (int j = 0; j < nGrid; j++) {
            for (int k = 0; k < nGrid/2; k++) {
            
            // Calculate kx, ky, kz
            double kx = i < nGrid/2 ? i : i - nGrid;
            double ky = j < nGrid/2 ? j : j - nGrid;
            double kz = k;

            // Calculate k
            double K = std::sqrt(kx*kx + ky*ky + kz*kz);
            double Pk = std::norm(kdata(i,j,k));
            
            int bin = int( (std::log(K) / std::log(Kmax)) * nBins);
            if (K < 0.00001) bin = 0;
            if (bin >= nBins) bin = nBins - 1;
            // Add P(k) to the bin and increment the bin count
            fPowerlog(bin) += Pk;
            nPowerlog(bin) += 1;
            // std::cout << "80" << kx << ky << kz << K << Pk << bin;
            }
        }
    }
    for (int i; i < nBins; i++){
        if (nPowerlog(i)) avgPowerlog(i) = fPowerlog(i) / nPowerlog(i);;
    }

    cout << "fPowerlog, nPowerlog calculated with log bins" << endl;
    
    std::ofstream outFilelog("binlog.txt"); // Open the output file

    // Write k and Pk arrays to file
    for (int i = 0; i < nBins; i++) {
        outFilelog << i << " " << avgPowerlog(i) << std::endl;
    }

    auto elapsed_read = std::chrono::duration_cast<std::chrono::nanoseconds>(end_read - start_read).count();
    std::cout << "Reading file took: " << elapsed_read/1e9 << " seconds\n";

    //delete[] fft_data; 

    return 0;
}
