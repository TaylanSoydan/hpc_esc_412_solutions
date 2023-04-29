// This uses features from C++17, so you may have to turn this on to compile
// g++ -std=c++17 -O3 -o assign assign.cxx tipsy.cxx
#include <iostream>
#include <cstdint>
#include <stdlib.h>
#include <chrono>
#include "blitz/array.h"
#include "tipsy.h"
#include <math.h>
#include <new>
#include "weights.h"
#include <vector>
#include <complex>
#include <fftw3.h>
#include <mpi.h>
#include <fftw3-mpi.h>
#include <numeric>
int compare(const void* a, const void* b) {
    float x1 = *((float*)a);
    float x2 = *((float*)b);
    if (x1 < x2) return -1;
    else if (x1 > x2) return 1;
    else return 0;
}
int get_slab(float x)
{
    float slab_width = 1;
    int slab = int(x / slab_width);
    if (slab >= 100) slab = 100 - 1;
    return slab;
}

int wrap_edge(int coordinate, int N)
{
    if (coordinate < 0)
    {
        coordinate += N;
    }
    else if (coordinate >= N)
    {
        coordinate -= N;
    }
    return coordinate;
}

int precalculate_W(float W[], int order, float r, float cell_half = 0.5)
{
    switch (order)
    {
    case 1:
        return ngp_weights(r, W);
    case 2:
        return cic_weights(r, W);
    case 3:
        return tsc_weights(r, W);
    case 4:
        return pcs_weights(r, W);
    default:
        throw std::invalid_argument("[precalculate_W] Order out of bound");
    }
}

int k_indx(int i, int nGrid)
{
    int res = i;
    if (i > nGrid / 2)
    {
        res = i - nGrid;
    }
    return res;
}

int get_i_bin(double k, int n_bins, int nGrid, int task = 1)
{
    double k_max = sqrt((nGrid / 2.0) * (nGrid / 2.0) * 3.0);
    switch (task)
    {
    case 1:
        return int(k);
    case 2:
        return int(k / k_max * n_bins);
    case 3:
        if (k != 0)
        {
            return int(log10(k) / log10(k_max) * n_bins);
        }
        else
        {
            return 0;
        }
    default:
        std::cout << "[get_i_bin] Selected invalid task number. Deafult to 1" << std::endl;
        return int(k);
    }
}

void save_binning(const int binning, std::vector<float> &fPower, std::vector<int> &nPower)
{
    const char *binning_filename;
    switch (binning)
    {
    case 1:
        binning_filename = "linear_binning.csv";
        break;
    case 2:
        binning_filename = "variable_binning.csv";
        break;
    case 3:
        binning_filename = "log_binning.csv";
        break;
    }

    std::ofstream f1(binning_filename);
    f1.precision(6);
    f1 << "P_k,k" << std::endl;
    for (int i = 0; i < fPower.size(); ++i)
    {
        f1 << (nPower[i] != 0 ? (fPower[i] / nPower[i]) : 0) << "," << i + 1 << std::endl;
    }
    f1.close();
}

void assign_mass(blitz::Array<float, 2> &r, int part_i_start, int part_i_end, int nGrid, blitz::Array<float, 3> &grid, int order = 4)
{
    // Loop over all cells for this assignment
    float cell_half = 0.5;
    std::cout << "Assigning mass to the grid using order " << order << std::endl;
#pragma omp parallel for
    for (int pn = part_i_start; pn < part_i_end; ++pn)
    {
        float x = r(pn, 0);
        float y = r(pn, 1);
        float z = r(pn, 2);

        float rx = (x + 0.5) * nGrid;
        float ry = (y + 0.5) * nGrid;
        float rz = (z + 0.5) * nGrid;

        // precalculate Wx, Wy, Wz and return start index
        float Wx[order], Wy[order], Wz[order];
        int i_start = precalculate_W(Wx, order, rx);
        int j_start = precalculate_W(Wy, order, ry);
        int k_start = precalculate_W(Wz, order, rz);

        for (int i = i_start; i < i_start + order; i++)
        {
            for (int j = j_start; j < j_start + order; j++)
            {
                for (int k = k_start; k < k_start + order; k++)
                {
                    float W_res = Wx[i - i_start] * Wy[j - j_start] * Wz[k - k_start];

// Deposit the mass onto grid(i,j,k)
#pragma omp atomic
                    grid(wrap_edge(i, nGrid), wrap_edge(j, nGrid), wrap_edge(k, nGrid)) += W_res;
                }
            }
        }
    }
}

void project_grid(blitz::Array<float, 3> &grid, int nGrid, const char *out_filename)
{
    auto start_time = std::chrono::high_resolution_clock::now();
    blitz::Array<float, 2> projected(nGrid, nGrid);
    for (int i = 0; i < nGrid; ++i)
    {
        for (int j = 0; j < nGrid; ++j)
        {
            projected(i, j) = max(grid(i, j, blitz::Range::all()));
        }
    }
    std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Projection took " << std::setw(9) << diff.count() << " s\n";

    std::ofstream f(out_filename);
    for (int i = 0; i < nGrid; ++i)
    {
        for (int j = 0; j < nGrid; ++j)
        {
            if (j != nGrid - 1)
            {
                f << projected(i, j) << ",";
            }
            else
            {
                f << projected(i, j);
            }
        }
        f << std::endl;
    }
    f.close();
}

int main(int argc, char *argv[])
{
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int i_rank, N_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &N_rank);

    if (argc <= 1)
    {
        std::cerr << "Usage: " << argv[0] << " tipsyfile.std grid-size [order, projection-filename]"
                  << std::endl;
        return 1;
    }

    int nGrid = 100;
    if (argc > 2)
    {
        nGrid = atoi(argv[2]);
    }

    int order = 1; 
    if (argc > 3)
    {
        order = atoi(argv[3]);
    }

    const char *out_filename = (argc > 4) ? argv[4] : "projected.csv";

    auto start_time = std::chrono::high_resolution_clock::now();

    TipsyIO io;
    io.open(argv[1]);
    if (io.fail())
    {
        std::cerr << "Unable to open tipsy file " << argv[1] << std::endl;
        return errno;
    }
    int N = io.count();

    // Load particle positions
    int N_per = (N + N_rank - 1) / N_rank;
    int i_start = N_per * i_rank;
    int i_end = std::min(N_per * (i_rank + 1), N);

    std::cerr << "Loading " << N << " particles" << std::endl;
    blitz::Array<float, 2> r(blitz::Range(i_start, i_end - 1), blitz::Range(0, 2));
    io.load(r);
    std::chrono::duration<double> diff_load = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Reading file took " << std::setw(9) << diff_load.count() << " s\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //int start0, local0; 
    //ptrdiff_t *local0, *start0;
    ptrdiff_t local0, start0;

    fftw_mpi_local_size_3d(nGrid, nGrid, nGrid, MPI_COMM_WORLD, &local0, &start0);
    float *data = new (std::align_val_t(64)) float[local0 * nGrid * nGrid]; //float[nGrid * nGrid * (nGrid + 2)];
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    blitz::Array<float, 3> grid_data(data, blitz::shape(local0, nGrid, nGrid), blitz::deleteDataWhenDone);
    grid_data = 0.0;
    blitz::Range new_range(start0, local0 + start0);
    blitz::Array<float, 3> grid = grid_data(new_range, blitz::Range::all(), blitz::Range(0, nGrid - 1));
    //blitz::Array<float, 3> grid = grid_data(blitz::Range::all(), blitz::Range::all(), blitz::Range(0, nGrid - 1));
    std::complex<float> *complex_data = reinterpret_cast<std::complex<float> *>(data);
    blitz::Array<std::complex<float>, 3> kdata(complex_data, blitz::shape(nGrid, nGrid, nGrid / 2 + 1));

    start_time = std::chrono::high_resolution_clock::now();
    assign_mass(r, i_start, i_end, nGrid, grid, order);
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int all_start0[N_rank];
    int all_local0[N_rank];

    MPI_Allgather(&start0, 1, MPI_INT, all_start0, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&local0, 1, MPI_INT, all_local0, 1, MPI_INT, MPI_COMM_WORLD);
    int slab2rank[nGrid];
    int slab = start0;
    for (int i = 0; i < N_rank; i++) {
        for (int j = 0; j < all_local0[i]; j++) {
        slab2rank[slab] = i;
        slab++;
        }
    }
    qsort(r.data(), r.rows(), 3*sizeof(float),compare);
    // First, create the sendcounts array
    // Get the number of ranks

    // Compute the displacement arrays
    blitz::Array<int, 1> sendcounts(N_rank);
    sendcounts = 0;
    for (int i = 0; i < N; i++) {
        int slab = get_slab(r(i, 0)); // get slab for particle i
        sendcounts(slab)++; // increment send count for that slab
    }

    blitz::Array<int, 1> recvcounts(N_rank);
    MPI_Alltoall(sendcounts.data(), 1, MPI_INT, recvcounts.data(), 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = blitz::sum(recvcounts);

    blitz::Array<float, 2> new_particles(total_recv, 3);

    blitz::Array<int, 1> senddispls(N_rank);
    blitz::Array<int, 1> recvdispls(N_rank);

    senddispls(0) = 0;
    recvdispls(0) = 0;

    for (int i = 1; i < N_rank; i++) {
        senddispls(i) = senddispls(i-1) + sendcounts(i-1);
        recvdispls(i) = recvdispls(i-1) + recvcounts(i-1);
    }

    MPI_Alltoallv(r.data(), sendcounts.data(), senddispls.data(), MPI_FLOAT,
                    new_particles.data(), recvcounts.data(), recvdispls.data(), MPI_FLOAT,
                    MPI_COMM_WORLD);

    // Copy the received particles to the sortedParticles array
    //std::memcpy(sortedParticles, new_particles.data(), total_recv * 3 * sizeof(float));






    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    std::chrono::duration<double> diff_assignment = std::chrono::high_resolution_clock::now() - start_time;
    std::cout << "Mass assignment took " << std::setw(9) << diff_assignment.count() << " s\n";

    if (i_rank == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, grid.data(), grid.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Reduce(grid.data(), nullptr, grid.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (i_rank == 0)
    {
        // Simple test
        std::cout << "Sum of all grid mass = " << blitz::sum(grid) << std::endl;

        project_grid(grid, nGrid, out_filename);

        // Convert to overdensity
        float grid_sum = sum(grid);
        float mean_density = grid_sum / (nGrid * nGrid * nGrid);
        grid = (grid - mean_density) / mean_density;

        //fftwf_plan plan = fftwf_plan_dft_r2c_3d(nGrid, nGrid, nGrid, data, (fftwf_complex *)complex_data, FFTW_ESTIMATE);
        fftwf_plan plan = fftwf_mpi_plan_dft_r2c_3d(nGrid, nGrid, nGrid, data, (fftwf_complex *)complex_data, MPI_COMM_WORLD,FFTW_ESTIMATE);
        
        std::cout << "Plan created" << std::endl;
        fftwf_execute(plan);
        std::cout << "Plan executed" << std::endl;
        fftwf_destroy_plan(plan);
        std::cout << "Plan destroyed" << std::endl;
        // Linear binning is 1
        // Variable binning is 2
        // Log binning is 3
        //const int binning = 2;
        //int n_bins = 80;
        //if (binning == 1)
        //{
        //    n_bins = nGrid;
        //}
        //std::vector<float> fPower(n_bins, 0.0);
        //std::vector<int> nPower(n_bins, 0);
        //float k_max = sqrt((nGrid / 2.0) * (nGrid / 2.0) * 3.0);

        //// loop over δ(k) and compute k from kx, ky and kz
        //for (int i = 0; i < nGrid; i++)
        //{
        //    int kx = k_indx(i, nGrid);
        //    for (int j = 0; j < nGrid; j++)
        //    {
        //        int ky = k_indx(j, nGrid);
        //        for (int l = 0; l < nGrid / 2 + 1; l++)
        //        {
        //            int kz = l;

        //            float k = sqrt(kx * kx + ky * ky + kz * kz);
        //            int i_bin = get_i_bin(k, n_bins, nGrid, binning);
        //            if (i_bin == fPower.size())
        //                i_bin--;
        //            fPower[i_bin] += std::norm(kdata(i, j, l));
        //            nPower[i_bin] += 1;
        //        }
        //    }
        //}

        //save_binning(binning, fPower, nPower);
        const int binning = 2;

        int n_bins = 80;
        if (binning == 1)
        {
            n_bins = nGrid;
        }
        std::vector<float> fPower_local(n_bins, 0.0);
        std::vector<int> nPower_local(n_bins, 0);
        float k_max = sqrt((nGrid / 2.0) * (nGrid / 2.0) * 3.0);

        // Determine the start and end index of kx values for each rank
        int kx_start = i_rank * nGrid / N_rank;
        int kx_end = (i_rank + 1) * nGrid / N_rank;

        // loop over δ(k) and compute k from kx, ky and kz for the current rank
        for (int i = kx_start; i < kx_end; i++)
        {
            int kx = k_indx(i, nGrid);
            for (int j = 0; j < nGrid; j++)
            {
                int ky = k_indx(j, nGrid);
                for (int l = 0; l < nGrid / 2 + 1; l++)
                {
                    int kz = l;

                    float k = sqrt(kx * kx + ky * ky + kz * kz);
                    int i_bin = get_i_bin(k, n_bins, nGrid, binning);
                    if (i_bin == fPower_local.size())
                        i_bin--;
                    fPower_local[i_bin] += std::norm(kdata(i, j, l));
                    nPower_local[i_bin] += 1;
                }
            }
        }

        // Reduce the binned values from all ranks to compute the final P(k) values
        std::vector<float> fPower(n_bins, 0.0);
        std::vector<int> nPower(n_bins, 0);
        MPI_Reduce(fPower_local.data(), fPower.data(), n_bins, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        MPI_Reduce(nPower_local.data(), nPower.data(), n_bins, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

        if (i_rank == 0)
        {
            save_binning(binning, fPower, nPower);
        }


    }
    MPI_Finalize();
}