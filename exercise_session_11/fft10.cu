#include <blitz/array.h>
#include <fftw3.h>
#include <complex>
#include <cmath>
#include <iostream>
#include <cuda.h>
#include <cufft.h>
#include <cufftw.h>

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
    //               reinterpret_cast<fftwf_complex *>(kdata.data()), rdata.data(), FFTW_ESTIMATE);
    fftwf_execute(plan);
    fftwf_destroy_plan(plan);
    rdata /= data.size(); // Normalize for the FFT
    return all(abs(data - rdata) < 1e-5);
}
    // Ex1
    //int main() {
    //    int n = 10000;

    //    // Out of place
    //    Array<float, 2> rdata1(n, n);
    //    Array<std::complex<float>, 2> kdata1(n, n / 2 + 1);

    //    fftwf_plan plan1 = fftwf_plan_dft_r2c_2d(n, n,
    //        rdata1.data(), reinterpret_cast<fftwf_complex *>(kdata1.data()), FFTW_ESTIMATE);

    //    fill_array(rdata1);
    //    fftwf_execute(plan1);
    //    fftwf_destroy_plan(plan1);

    //    std::cout << ">>> Out of place FFT " << (validate(rdata1, kdata1) ? "match" : "MISMATCH") << std::endl;

    //    // In-place
    //    Array<float, 2> raw_data2(n, n + 2);
    //    Array<float, 2> rdata2 = raw_data2(Range(0, n - 1), Range(0, n - 1));

    //    fftwf_plan plan2 = fftwf_plan_dft_r2c_2d(n, n,
    //        rdata2.data(), reinterpret_cast<fftwf_complex *>(rdata2.data()), FFTW_ESTIMATE);

    //    fill_array(rdata2);
    //    fftwf_execute(plan2);
    //    fftwf_destroy_plan(plan2);

    //    Array<std::complex<float>, 2> kdata2(reinterpret_cast<std::complex<float> *>(rdata2.data()),
    //        shape(n, n / 2 + 1), neverDeleteData);

    //    std::cout << ">>> In-place FFT " << (validate(rdata1, kdata2) ? "match" : "MISMATCH") << std::endl;

    //    // Transfer data3 to GPU and perform in-place FFT
    //    Array<float, 2> raw_data3(n, n + 2);
    //    Array<float, 2> data3 = raw_data3(blitz::Range::all(), blitz::Range(0, n - 1));

    //    fill_array(data3);

    //    size_t size_in_bytes = sizeof(float) * n * (n + 2);

    //    // Allocate memory on the GPU
    //    void *device_data;
    //    cudaMalloc(&device_data, size_in_bytes);

    //    // Copy data from CPU to GPU
    //    cudaMemcpy(device_data, data3.data(), size_in_bytes, cudaMemcpyHostToDevice);
    //    cudaDeviceSynchronize();

    //    cudaMemcpy(raw_data4.data(), device_data, size_in_bytes, cudaMemcpyDeviceToHost);
    //    std::cout << (validate2d(raw_data3, raw_data4) ? "passed" : "failed") << std::endl;

    //    cudaFree(device_data);

    //    // Perform in-place FFT on GPU
    //    Array<float, 2> raw_data5(n, n + 2);
    //    Array<float, 2> data5 = raw_data5(blitz::Range::all(), blitz::Range(0, n - 1));
    //    Array<std::complex<float>, 2> data6(n, n / 2 + 1);

    //    float *device_data;
    //    size_t size_in_byte = sizeof(float) * n * (n + 2);
    //    cudaMalloc(&device_data, size_in_byte);
    //    cudaMemcpy(device_data, raw_data5.data(), size_in_byte, cudaMemcpyHostToDevice);
    //    cudaDeviceSynchronize();

    //    int dims[2] = { n, n };
    //    int inembed[2] = { n, n + 2 };
    //    int onembed[2] = { n, n / 2 + 1 };
    //    int batch = 1;
    //    int odist = 1;
    //    int idist = 2;
    //    int istride = 1;
    //    int ostride = 1;

    //    cufftHandle plan;
    //    cufftCreate(&plan);
    //    cufftSetAutoAllocation(plan, CUFFT_NO_WORKSPACE);
    //    cufftMakePlanMany(plan, 2, dims, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);

    //    size_t workSize;
    //    cufftGetSize(plan, &workSize);

    //    void *workArea;
    //    cudaMalloc(&workArea, workSize);

    //    cufftSetWorkArea(plan, workArea);

    //    cufftExecR2C(plan, reinterpret_cast<cufftReal *>(device_data),
    //        reinterpret_cast<cufftComplex *>(device_data));

    //    cudaDeviceSynchronize();
    //    cufftDestroy(plan);

    //    // Copy data from GPU back to CPU
    //    cudaMemcpy(data6.data(), device_data, size_in_bytes, cudaMemcpyDeviceToHost);

    //    // Synchronize GPU
    //    cudaDeviceSynchronize();

    //    // Compare data3 and data4
    //    std::cout << ">>> Data Comparison " << (validate(data5, data6) ? "match" : "MISMATCH") << std::endl;

    //    // Free GPU memory
    //    cudaFree(device_data);
    //    cudaFree(workArea);

    //    return 0;
    //}
    // EX 2
    //int main() {
    //    int n = 10000;

    //    // Create a new stream
    //    cudaStream_t stream;
    //    cudaStreamCreate(&stream);

    //    // Out of place
    //    Array<float, 2> rdata1(n, n);
    //    Array<std::complex<float>, 2> kdata1(n, n / 2 + 1);

    //    fftwf_plan plan1 = fftwf_plan_dft_r2c_2d(n, n,
    //        rdata1.data(), reinterpret_cast<fftwf_complex *>(kdata1.data()), FFTW_ESTIMATE);

    //    fill_array(rdata1);
    //    fftwf_execute(plan1);
    //    fftwf_destroy_plan(plan1);

    //    std::cout << ">>> Out of place FFT " << (validate(rdata1, kdata1) ? "match" : "MISMATCH") << std::endl;

    //    // In-place
    //    Array<float, 2> raw_data2(n, n + 2);
    //    Array<float, 2> rdata2 = raw_data2(Range(0, n - 1), Range(0, n - 1));

    //    fftwf_plan plan2 = fftwf_plan_dft_r2c_2d(n, n,
    //        rdata2.data(), reinterpret_cast<fftwf_complex *>(rdata2.data()), FFTW_ESTIMATE);

    //    fill_array(rdata2);
    //    fftwf_execute(plan2);
    //    fftwf_destroy_plan(plan2);

    //    Array<std::complex<float>, 2> kdata2(reinterpret_cast<std::complex<float> *>(rdata2.data()),
    //        shape(n, n / 2 + 1), neverDeleteData);

    //    std::cout << ">>> In-place FFT " << (validate(rdata1, kdata2) ? "match" : "MISMATCH") << std::endl;

    //    // Transfer data3 to GPU and perform in-place FFT
    //    Array<float, 2> raw_data3(n, n + 2);
    //    Array<float, 2> data3 = raw_data3(blitz::Range::all(), blitz::Range(0, n - 1));

    //    fill_array(data3);

    //    size_t size_in_bytes = sizeof(float) * n * (n + 2);

    //    // Allocate memory on the GPU with the new stream
    //    void *device_data;
    //    cudaMallocAsync(&device_data, size_in_bytes, stream);

    //    // Copy data from CPU to GPU with the new stream
    //    cudaMemcpyAsync(device_data, data3.data(), size_in_bytes, cudaMemcpyHostToDevice, stream);

    //    // Synchronize the new stream
    //    cudaStreamSynchronize(stream);

    //    cudaMemcpy(raw_data4.data(), device_data, size_in_bytes, cudaMemcpyDeviceToHost);
    //    std::cout << (validate2d(raw_data3, raw_data4) ? "passed" : "failed") << std::endl;

    //    cudaFree(device_data);

    //    // Perform in-place FFT on GPU
    //    Array<float, 2> raw_data5(n, n + 2);
    //    Array<float, 2> data5 = raw_data5(blitz::Range::all(), blitz::Range(0, n - 1));
    //    Array<std::complex<float>, 2> data6(n, n / 2 + 1);

    //    float *device_data;
    //    size_t size_in_byte = sizeof(float) * n * (n + 2);
    //    cudaMallocAsync(&device_data, size_in_byte, stream);
    //    cudaMemcpyAsync(device_data, raw_data5.data(), size_in_byte, cudaMemcpyHostToDevice, stream);

    //    // Create cuFFT plan
    //    int dims[2] = { n, n };
    //    int inembed[2] = { n, n + 2 };
    //    int onembed[2] = { n, n / 2 + 1 };
    //    int batch = 1;
    //    int odist = 1;
    //    int idist = 2;
    //    int istride = 1;
    //    int ostride = 1;

    //    cufftHandle plan;
    //    cufftCreate(&plan);
    //    cufftSetAutoAllocation(plan, CUFFT_NO_WORKSPACE);
    //    cufftMakePlanMany(plan, 2, dims, inembed, istride, idist, onembed, ostride, odist, CUFFT_R2C, batch);

    //    // Set the stream and work area
    //    cufftSetStream(plan, stream);
    //    cufftSetWorkArea(plan, device_data);

    //    // Execute the cuFFT plan
    //    cufftExecR2C(plan, reinterpret_cast<cufftReal *>(device_data),
    //        reinterpret_cast<cufftComplex *>(device_data));

    //    // Synchronize the stream
    //    cudaStreamSynchronize(stream);

    //    cufftDestroy(plan);

    //    // Copy data from GPU back to CPU with the new stream
    //    cudaMemcpyAsync(data6.data(), device_data, size_in_bytes, cudaMemcpyDeviceToHost, stream);

    //    // Synchronize the new stream
    //    cudaStreamSynchronize(stream);

    //    // Compare data3 and data4
    //    std::cout << ">>> Data Comparison " << (validate(data5, data6) ? "match" : "MISMATCH") << std::endl;

    //    // Free GPU memory
    //    cudaFree(device_data);

    //    // Destroy the stream
    //    cudaStreamDestroy(stream);

    //    return 0;
    //}

 // Ex3
int main() {
    int n = 10000;
    int n_streams = 3;

    // Create an array of streams
    cudaStream_t streams[n_streams];
    for (int i = 0; i < n_streams; ++i) {
        cudaStreamCreate(&streams[i]);
    }

    // Create arrays for GPU slabs and work areas for each stream
    void *gpu_slabs[n_streams];
    void *gpu_works[n_streams];
    size_t slab_size = sizeof(float) * n * (n + 2);
    size_t work_size = calculate_work_size(n);

    for (int i = 0; i < n_streams; ++i) {
        cudaMallocAsync(&gpu_slabs[i], slab_size, streams[i]);
        cudaMallocAsync(&gpu_works[i], work_size, streams[i]);
    }

    // Out of place
    Array<float, 2> rdata1(n, n);
    Array<std::complex<float>, 2> kdata1(n, n / 2 + 1);
    fftwf_plan plan1 = fftwf_plan_dft_r2c_2d(n, n,
        rdata1.data(), reinterpret_cast<fftwf_complex *>(kdata1.data()), FFTW_ESTIMATE);
    fill_array(rdata1);

    // Process each slab
    int stream_index = 0;
    for (int i = 0; i < n; ++i) {
        cudaMemcpyAsync(gpu_slabs[stream_index], rdata1.data(), slab_size, cudaMemcpyHostToDevice, streams[stream_index]);
        cufftSetStream(plan1, streams[stream_index]);
        cufftSetWorkArea(plan1, gpu_works[stream_index]);
        cufftExecR2C(plan1, reinterpret_cast<cufftReal *>(gpu_slabs[stream_index]),
            reinterpret_cast<cufftComplex *>(gpu_slabs[stream_index]));
        cudaMemcpyAsync(rdata1.data(), gpu_slabs[stream_index], slab_size, cudaMemcpyDeviceToHost, streams[stream_index]);
        stream_index = (stream_index + 1) % n_streams;
    }

    fftwf_destroy_plan(plan1);

    std::cout << ">>> Out of place FFT " << (validate(rdata1, kdata1) ? "match" : "MISMATCH") << std::endl;

    // Free GPU memory
    for (int i = 0; i < n_streams; ++i) {
        cudaFree(gpu_slabs[i]);
        cudaFree(gpu_works[i]);
    }

    // Destroy streams
    for (int i = 0; i < n_streams; ++i) {
        cudaStreamDestroy(streams[i]);
    }

    return 0;
}
