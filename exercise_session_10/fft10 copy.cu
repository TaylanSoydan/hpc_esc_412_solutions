#include <blitz/array.h>
#include <fftw3.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cufft.h>

using namespace blitz;
using std::complex;

void fill_array(Array<float, 2> &data) {
    // Set the grid to the sum of two sine functions
    for (int i = 0; i < data.rows(); i++) {
        for (int j = 0; j < data.cols(); j++) {
            float x = (float)i / 25.0; // Period of 1/4 of the box in x
            float y = (float)j / 10.0; // Period of 1/10 of the box in y
            data(i, j) = sin(2.0 * M_PI * x) + sin(2.0 * M_PI * y);
        }
    }
}

// Verify the FFT (kdata) of data by performing a reverse transform and comparing
bool validate(Array<float, 2> &data, Array<std::complex<float>, 2> kdata) {
    Array<float, 2> rdata(data.extent());
    fftwf_plan plan = fftwf_plan_dft_c2r_2d(data.rows(), data.cols(),
                                           reinterpret_cast<fftwf_complex *>(kdata.data()), rdata.data(), FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    rdata /= data.size(); // Normalize for the FFT
    return all(abs(data - rdata) < 1e-5);
}

int main() {
    int n = 10000;

    // Out of place
    Array<float, 2> rdata1(n, n);
    Array<std::complex<float>, 2> kdata1(n, n / 2 + 1);
    fftwf_plan plan1 = fftwf_plan_dft_r2c_2d(n, n,
                                             rdata1.data(), reinterpret_cast<fftwf_complex *>(kdata1.data()), FFTW_ESTIMATE);
    fill_array(rdata1);
    fftwf_execute(plan1);
    fftwf_destroy_plan(plan1);
    std::cout << ">>> Out of place FFT " << (validate(rdata1, kdata1) ? "match" : "MISMATCH") << std::endl;

    // In-place
    Array<float, 2> raw_data2(n, n + 2);
    Array<float, 2> rdata2 = raw_data2(Range(0, n - 1), Range(0, n - 1));
    fftwf_plan plan2 = fftwf_plan_dft_r2c_2d(n, n,
                                             rdata2.data(), reinterpret_cast<fftwf_complex *>(rdata2.data()), FFTW_ESTIMATE);
    fill_array(rdata2);
    fftwf_execute(plan2);
    fftwf_destroy_plan(plan2);
    Array<std::complex<float>, 2> kdata2(reinterpret_cast<std::complex<float> *>(rdata2.data()),
                                         shape(n, n / 2 + 1), neverDeleteData);
    std::cout << ">>> In-place FFT " << (validate(rdata1, kdata2) ? "match" : "MISMATCH") << std::endl;

    // Transfer data3 to GPU and perform in-place FFT
    Array<float, 2> raw_data3(n, n + 2);
    Array<float, 2> data3 = raw_data3(Range(0, n - 1), Range(0, n - 1));
    fill_array(data3);

    size_t size_in_bytes = sizeof(float) * n * (n + 2);

    // Allocate memory on the GPU
    void *device_data;
    cudaMalloc(&device_data, size_in_bytes);

    // Copy data from CPU to GPU
    cudaMemcpy(device_data, data3.data(), size_in_bytes, cudaMemcpyHostToDevice);

    // Perform in-place FFT on GPU
    int dims[2] = {n, n};
    int inembed[2] = {n, n + 2};
    int onembed[2] = {n, n / 2 + 1};
    int batch = 1;
    int odist = n * (n / 2 + 1);
    int idist = 2 * odist;
    int istride = 1;
    int ostride = 1;

    cufftHandle plan;
    cufftPlanMany(&plan, 2, dims, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);
    cufftExecR2C(plan, reinterpret_cast<cufftReal *>(device_data),
                 reinterpret_cast<cufftComplex *>(device_data));
    cufftDestroy(plan);

    // Copy data from GPU back to CPU
    cudaMemcpy(data3.data(), device_data, size_in_bytes, cudaMemcpyDeviceToHost);

    // Synchronize GPU
    cudaDeviceSynchronize();

    // Compare data3 and data4
    std::cout << ">>> Data Comparison " << (validate(data3, rdata2) ? "match" : "MISMATCH") << std::endl;

    // Free GPU memory
    cudaFree(device_data);

    return 0;
}
