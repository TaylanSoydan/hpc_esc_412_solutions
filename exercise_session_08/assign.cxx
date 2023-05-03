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

int main(int argc, char *argv[]){
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

    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);

    int i_rank, N_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &i_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &N_rank);

    ptrdiff_t alloc_local, local0, start0;
    alloc_local = fftw_mpi_local_size_3d(nGrid, nGrid, nGrid, MPI_COMM_WORLD, &local0, &start0);
    assert (local0 > 0);


    int* COMM_SLAB_SIZE = new int [N_rank];
    int* COMM_SLAB_START = new int [N_rank];
    int* SLAB2RANK = new int[nGrid];
    MPI_Allgather(&start0, 1, MPI_INT, COMM_SLAB_START, 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&local0, 1, MPI_INT, COMM_SLAB_SIZE, 1, MPI_INT, MPI_COMM_WORLD);

    int current = 0;
    for (int i = 0; i < nGrid; i++) {
        if (current < N_rank - 1 && COMM_SLAB_START[current + 1] <= i) current++;
        SLAB2RANK[i] = current;
    }

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

    std::cout << "i am rank = " << i_rank << " i_start, i_end - 1 = " << i_start << "," << i_end - 1 << " \n";
    io.load(r);
    std::chrono::duration<double> diff_load = std::chrono::high_resolution_clock::now() - start_time;
    //std::cout << "Reading file took " << std::setw(9) << diff_load.count() << " s\n";
    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    
    qsort(r.data(), r.rows(), 3*sizeof(float),compare);

    for (int i = i_start; i < i_end-1; i++){
        if (r(i, 0) > r(i+1,0)){
            std::cout << "r not sorted properly" << "\n";
        }
    }
    int * slab_cut_indexes = new int [N_rank];
    slab_cut_indexes[N_rank - 1] = i_end - i_start;
    int counter = 0;
    printf("starting to find slab cut indexes \n");
    for (int i = i_start; i < i_end; i++){
        int particle_slab = int((r(i, 0) + 0.5)*nGrid);
        int next_particle_slab = int((r(i+1, 0) + 0.5)*nGrid);
        int particle_rank = SLAB2RANK[particle_slab];
        int next_particle_rank = SLAB2RANK[next_particle_slab];
        if (next_particle_rank > particle_rank) {
            printf("r(i, 0) + 0.5 = %f r(i+1, 0) + 0.5 = %f\n", r(i, 0) + 0.5, r(i+1, 0) + 0.5);
            printf("particle_rank = %d next_particle_rank = %d\n", particle_rank, next_particle_rank);
            printf("i = %d i_start = %d i_end = %d\n", i, i_start, i_end);
            printf("will be cut at %d\n", i+1-i_start);
            slab_cut_indexes[counter] = i+1-i_start; 
            counter++;
        }}
    for (int i = 0; i < N_rank; i++){printf("slab_cut_index for rank = %d is = %d\n", i_rank, slab_cut_indexes[i]);}
    int* num_particles_to_send = new int[N_rank];  
    num_particles_to_send[0] = slab_cut_indexes[0];
    for (int i = 1; i < N_rank; i++) {
    num_particles_to_send[i] = slab_cut_indexes[i] - slab_cut_indexes[i-1];}
    
    for (int i = 0; i < N_rank; i++) {printf("num_particles_to_send for rank = %d is = %d\n", i_rank, num_particles_to_send[i]);}
    int total_num_particles_to_send = 0;
    for (int i = 0; i < N_rank; i++) total_num_particles_to_send += num_particles_to_send[i];
    printf("total_num_particles_to_send for rank = %d is = %d\n", i_rank, total_num_particles_to_send);
    
    int* num_particles_to_recv = new int[N_rank];
    MPI_Alltoall(num_particles_to_send, 1, MPI_INT, num_particles_to_recv, 1, MPI_INT, MPI_COMM_WORLD);
    int total_num_particles_to_recv = 0;
    for (int i = 0; i < N_rank; i++) {
        printf("num_particles_to_recv for rank = %d is = %d\n", i_rank, num_particles_to_recv[i]);
        total_num_particles_to_recv += num_particles_to_recv[i];}
    printf("total_num_particles_to_recv for rank = %d is = %d\n", i_rank, total_num_particles_to_recv);

    int* MPISendCount = new int [N_rank];
    int* MPIRecvCount = new int [N_rank];
    for (int i = 0; i < N_rank; i++) {
        MPISendCount[i] = (num_particles_to_send[i] * 3);
        MPIRecvCount[i] = (num_particles_to_recv[i] * 3);
        assert (MPISendCount[i] >= 0);
        assert (MPIRecvCount[i] >= 0);
    }

    int* MPISendOffset = new int [N_rank];
    int* MPIRecvOffset = new int [N_rank];
    MPISendOffset[0] = 0;
    MPIRecvOffset[0] = 0;
    for (int i = 1; i < N_rank; i++)  {
        MPISendOffset[i] = MPISendOffset[i-1] + MPISendCount[i-1];
        MPIRecvOffset[i] = MPIRecvOffset[i-1] + MPIRecvCount[i-1];
        printf("MPISendOffset for rank = %d is = %d\n", i_rank, MPISendOffset[i]);
        printf("MPIRecvOffset for rank = %d is = %d\n", i_rank, MPIRecvOffset[i]);
    }
    float* r_sorted = new float [total_num_particles_to_recv * 3];
    MPI_Alltoallv(r.data(), MPISendCount, MPISendOffset, MPI_FLOAT, r_sorted, MPIRecvCount, MPIRecvOffset, MPI_FLOAT, MPI_COMM_WORLD);
    delete [] MPISendCount;
    delete [] MPISendOffset;
    delete [] MPIRecvCount;
    delete [] MPIRecvOffset;
    delete [] num_particles_to_send;
    delete [] num_particles_to_recv;
    blitz::Array<float, 2> rsorted(r_sorted, blitz::shape(total_num_particles_to_recv,3), blitz::deleteDataWhenDone);

    int new_dim = local0 + order - 1;
    float *data = new (std::align_val_t(64)) float[new_dim * nGrid * (nGrid+2)]; //float[nGrid * nGrid * (nGrid + 2)];
    blitz::Array<float, 3> grid_data(data, blitz::shape(new_dim, nGrid, (nGrid+2)), blitz::deleteDataWhenDone);
    grid_data = 0.0;
    blitz::Array<float, 3> grid = grid_data(blitz::Range::all(), blitz::Range::all(), blitz::Range(0, nGrid - 1));
    grid = 0.0;
    std::complex<float> *complex_data = reinterpret_cast<std::complex<float> *>(data);
    blitz::Array<std::complex<float>, 3> kdata(complex_data, blitz::shape(new_dim, nGrid, nGrid / 2 + 1));

    start_time = std::chrono::high_resolution_clock::now();
    //assign_mass(rsorted, i_start, i_end, nGrid, grid, order);
    //int upperbound;
    //    float upperboundary;
    //    if (i_rank == N_rank - 1) {
    //        upperbound = nGrid;
    //    } else {
    //        upperbound = COMM_SLAB_START[i_rank + 1];
    //    }
    //    upperboundary = (float) upperbound / nGrid;
    //    printf("i_rank = %d upperboundary = %f\n", i_rank, upperboundary);    
    //    printf("i_start = %d i_start + total_num_particles_to_recv = %d\n", i_start, i_start + total_num_particles_to_recv);    
    #pragma omp parallel for
    for (int pn = 0; pn < 100000 ; pn++)
    {
        float x = rsorted(pn, 0);
        float y = rsorted(pn, 1);
        float z = rsorted(pn, 2);

        float rx = (x + 0.5) * nGrid; //std::ceil(upperboundary)
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
                    grid(wrap_edge(i, nGrid), wrap_edge(j, nGrid), wrap_edge(k, nGrid)) += W_res; //std::ceil(upperboundary)
                }
            }
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    //std::chrono::duration<double> diff_assignment = std::chrono::high_resolution_clock::now() - start_time;
    //std::cout << "Mass assignment took " << std::setw(9) << diff_assignment.count() << " s\n";


    //    // Simple test
    printf("For rank = %d sum of mass = %f\n", i_rank, blitz::sum(grid));

    if (i_rank == 0)
    {MPI_Reduce(MPI_IN_PLACE, grid.data(), grid.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);}
    else
    {MPI_Reduce(grid.data(), nullptr, grid.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);}

    if (i_rank == 0)
    {printf("Sum of all grid mass = %f \n", blitz::sum(grid));}
                                                                                                //        project_grid(grid, nGrid, out_filename);

                                                                                                //        // Convert to overdensity
    //float grid_sum = sum(grid);
    //float mean_density = grid_sum / (nGrid * nGrid * nGrid);
    //grid = (grid - mean_density) / mean_density;

                                                                                                //        //fftwf_plan plan = fftwf_plan_dft_r2c_3d(nGrid, nGrid, nGrid, data, (fftwf_complex *)complex_data, FFTW_ESTIMATE);
    //fftwf_plan plan = fftwf_mpi_plan_dft_r2c_3d(nGrid, nGrid, nGrid, data, (fftwf_complex *)complex_data, MPI_COMM_WORLD,FFTW_ESTIMATE);
                                                                                                //        
    //std::cout << "Plan created" << std::endl;
    //fftwf_execute(plan);
    //std::cout << "Plan executed" << std::endl;
    //fftwf_destroy_plan(plan);
    //std::cout << "Plan destroyed" << std::endl;
                                                                                                //        // Linear binning is 1
                                                                                                //        // Variable binning is 2
                                                                                                //        // Log binning is 3
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
    //    //int kx = k_indx(i, nGrid);
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

    ////MPI_Reduce(MPI_IN_PLACE, fPower.data(), fPower.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    ////MPI_Reduce(MPI_IN_PLACE, nPower.data(), nPower.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    //if (i_rank == 0)
    //{
    //    MPI_Reduce(MPI_IN_PLACE, fPower.data(), fPower.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    //}
    //else
    //{
    //    MPI_Reduce(fPower.data(), nullptr, fPower.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    //}
    //    if (i_rank == 0)
    //{
    //    MPI_Reduce(MPI_IN_PLACE, nPower.data(), nPower.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    //}
    //else
    //{
    //    MPI_Reduce(nPower.data(), nullptr, nPower.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    //}

    //if (i_rank == 0)
    //{
    //    save_binning(binning, fPower, nPower);
    //}
    //

MPI_Finalize();
}