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
    float *fft_data = new (std::align_val_t(64)) float[nGrid*nGrid*(nGrid+2)]; // 512-bit alignment
    blitz::TinyVector<int, 3> fft_shape(nGrid, nGrid, nGrid+2); // Shape of the array
    blitz::Array<float, 3> fft_array(fft_data, fft_shape, blitz::neverDeleteData); //padded
    fft_array = 0.0;
    blitz::Array<float, 3> fft_grid = fft_array(blitz::Range::all(), blitz::Range::all(), blitz::Range(0, nGrid)); //ngrid^3 subgrid of fft_array
    blitz::Array<std::complex<float>,3> fft_kdata(reinterpret_cast<std::complex<float>*>(fft_data), blitz::shape(nGrid,nGrid,nGrid/2+1));


    //Array<float,3> grid_pcs(nGrid,nGrid,nGrid);
    //Array<float,3> fft_grid(nGrid,nGrid,nGrid);
    
        // Allocate memory externally 
    //float* data = new (std::align_val_t(64)) float[nGrid*nGrid*(nGrid+2)]; // 512-bit alignment
    //blitz::TinyVector<int, 3> shape(nGrid, nGrid, nGrid+2); // Shape of the array
    //blitz::Array<float, 3> myArray(data, shape, blitz::neverDeleteData); 
    //// Create a subarray view with dimensions N × N × N
    //blitz::Range all = blitz::Range::all();
    //blitz::Array<float, 3> subArray = myArray(all, all, blitz::Range(0, nGrid));
    //// Create complex array
    //blitz::Array<std::complex<float>,3> kdata(reinterpret_cast<std::complex<float>*>(data), blitz::shape(nGrid,nGrid,nGrid/2+1));

    auto start_assign = std::chrono::high_resolution_clock::now();

    //PCS
    //#pragma omp parallel for
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
                    //#pragma omp atomic
                    fft_grid(I,J,K)+=W; 

        }}}}
    auto sum_pcs = blitz::sum(fft_grid);
    std::cout << "pcs total = " << sum_pcs << "\n";
    //std::cout << "Shape of array: (" << fft_grid.shape()[0] << ", " << fft_grid.shape()[1] << ", " << fft_grid.shape()[2] << ")" << std::endl;


    // Calculate overdensity
    float mean_pcs = sum_pcs / (nGrid * nGrid * nGrid);
    for(int i = 0 ; i < nGrid; i++) {
        for(int j= 0 ; j < nGrid; j++) {
            for (int k = 0 ; k < nGrid; k++) {
                float density = fft_grid(i,j,k); 
                float overdensity = (density - mean_pcs) / mean_pcs;
                fft_grid(i,j,k) = overdensity;
                //std::cout << " overdensity = " << overdensity;
                }}}

    auto end_assign = std::chrono::high_resolution_clock::now();
    auto start_project = std::chrono::high_resolution_clock::now();
    // Projection
    Array<float,2>projected(nGrid,nGrid);
    thirdIndex k;
    projected = blitz::max(fft_grid,k);
    
    auto end_project = std::chrono::high_resolution_clock::now();
//// FFT
    
    fftwf_plan plan = fftwf_plan_dft_r2c_3d(nGrid,nGrid,nGrid, fft_array.data(),//fft_array.data(), 
                                                        (fftwf_complex *)fft_kdata.data(), // output parameters
                                                        FFTW_ESTIMATE); // flags
    fftwf_execute(plan);
    //Destroy plan
    fftwf_destroy_plan(plan); 
//float *data = new (std::align_val_t(64)) float[nGrid * nGrid * (nGrid + 2)];
//Array<float, 3> grid_data(data, shape(nGrid, nGrid, nGrid), deleteDataWhenDone);
//Array<float, 3> grid = grid_data(Range::all(), Range::all(), Range(0, nGrid - 1));
//std::complex<float> *complex_data = reinterpret_cast<std::complex<float> *>(data);
//blitz::Array<std::complex<float>, 3> kdata(complex_data, shape(nGrid, nGrid, nGrid / 2 + 1));
//fftwf_plan plan = fftwf_plan_dft_r2c_3d(nGrid, nGrid, nGrid, data, (fftwf_complex *)complex_data, FFTW_ESTIMATE);

    //for (int t = 0; t < nGrid; nGrid++) {
        //std::cout << "fft_data = " << fft_kdata.data()[t];
        //std::cout << "fft_grid = " << fft_grid(t);
        //std::cout << "grid_pcs = " << grid_pcs(0,t,0);
    //}
    //for (int i = 0; i < nGrid * nGrid * (nGrid/2+1); i++) {
    //    std::cout << "Element " << i << ": " << fft_kdata[i][0] << " + " << fft_kdata[i][1] << "i" << std::endl;
    //}

    // Initialize K
    int nBins = nGrid;
    Array<double,1> fPower(nBins);
    fPower = 0.0;
    Array<int,1> nPower(nBins);
    nPower = 0;
    Array<double,1> avgPower(nBins);
    avgPower = 0.0;
    // Calculating K and Binning with ibin = k
    for (int i = 0; i < nGrid; i++) {
        for (int j = 0; j < nGrid; j++) {
            for (int k = 0; k < nGrid/2+1; k++) {
            
            // Calculate kx, ky, kz
            double kx = i < nGrid/2 ? i : i - nGrid;
            double ky = j < nGrid/2 ? j : j - nGrid;
            double kz = k;

            // Calculate k
            double K = std::sqrt(kx*kx + ky*ky + kz*kz);
            double Pk = std::norm(fft_kdata(i,j,k));

            int bin = int(K);
            if (bin >= nBins) bin = nBins - 1;
            
            // Add P(k) to the bin and increment the bin count
            fPower[bin] += Pk;
            nPower[bin] += 1;
            //std::cout << " K = " << K << " Pk = " << Pk << " bin = " << bin;
            }
        }
    }
    avgPower = fPower / nPower;
    std::ofstream fout2("bin100.txt");
    if (fout2.is_open()) {
        for (int i = 0; i<nBins; i++) {
                fout2 << avgPower(i) << ",";
        }
    }
    fout2.close();

    // Calculating K and Binning with ibin = (k - kmax) / nbins
    double Kmax = nGrid * nGrid * sqrt(3);
    nBins = 80;
    fPower = 0.0;
    nPower = 0;
    avgPower = 0.0;
    for (int i = 0; i < nGrid; i++) {
        for (int j = 0; j < nGrid; j++) {
            for (int k = 0; k < nGrid/2+1; k++) {
            
            // Calculate kx, ky, kz
            double kx = i < nGrid/2 ? i : i - nGrid;
            double ky = j < nGrid/2 ? j : j - nGrid;
            double kz = k;

            // Calculate k
            double K = std::sqrt(kx*kx + ky*ky + kz*kz);
            double Pk = std::norm(fft_kdata(i,j,k));

            int bin = int( (K / Kmax) * nBins);
            if (bin >= nBins) bin = nBins - 1;
            
            // Add P(k) to the bin and increment the bin count
            fPower[bin] += Pk;
            nPower[bin] += 1;
            // std::cout << "80" << kx << ky << kz << K << Pk << bin;
            }
        }
    }
    avgPower = fPower / nPower;
    std::ofstream fout3("bin80.txt");
    if (fout3.is_open()) {
        for (int i = 0; i<nBins; i++) {
                fout3 << fPower(i) << ",";
        }
    }
    fout3.close();

    // Log binning
    nBins = 80;
    fPower = 0.0;
    nPower = 0;
    avgPower = 0.0;
    for (int i = 0; i < nGrid; i++) {
        for (int j = 0; j < nGrid; j++) {
            for (int k = 0; k < nGrid/2+1; k++) {
            
            // Calculate kx, ky, kz
            double kx = i < nGrid/2 ? i : i - nGrid;
            double ky = j < nGrid/2 ? j : j - nGrid;
            double kz = k;

            // Calculate k
            double K = std::sqrt(kx*kx + ky*ky + kz*kz);
            double Pk = std::norm(fft_kdata(i,j,k));

            
            int bin = int( (std::log(K) / std::log(Kmax)) * nBins);
            if (K < 0.00001) bin = 0;
            if (bin >= nBins) bin = nBins - 1;
            
            // Add P(k) to the bin and increment the bin count
            fPower[bin] += Pk;
            nPower[bin] += 1;
            // std::cout << "80" << kx << ky << kz << K << Pk << bin;
            }
        }
    }
    avgPower = fPower / nPower;
    std::ofstream fout4("binlog.txt");
    if (fout4.is_open()) {
        for (int i = 0; i<nBins; i++) {
                fout4 << fPower(i) << ",";
        }
    }
    fout4.close();

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

    delete[] fft_data; 

    return 0;
}
