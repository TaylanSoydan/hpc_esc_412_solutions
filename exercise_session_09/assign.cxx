#include "tipsy.h"
#include <math.h>
#include <new>
#include "weights.h"
#include <vector>
#include <complex>
#include <fftw3.h>
#include <mpi.h>
#include <fftw3-mpi.h>
#include <algorithm>

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

int add_order(int coordinate, int order)
{
    return coordinate + order;
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

void assign_mass(blitz::Array<float, 2> &r, int part_i_start, int part_i_end, int nGrid, blitz::Array<float, 3> &grid, int order, int grid_start, int grid_end)
{
    // Loop over all cells for this assignment
    float cell_half = 0.5;
    printf("Assigning mass to the grid using order %d\n", order);
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
        if (i_start < 0)
        {
            i_start += nGrid;
        }

        for (int i = i_start; i < i_start + order; i++)
        {
            for (int j = j_start; j < j_start + order; j++)
            {
                for (int k = k_start; k < k_start + order; k++)
                {
                    float W_res = Wx[i - i_start] * Wy[j - j_start] * Wz[k - k_start];
                    if (i < grid_start || i >= grid_end)
                    {
                        printf("[ERROR] i = %d, grid_start = %d, i_start = %d \n", i, grid_start, i_start);
                    }
// Deposit the mass onto grid(i,j,k)
#pragma omp atomic
                    grid(i, wrap_edge(j, nGrid), wrap_edge(k, nGrid)) += W_res;
                }
            }
        }
    }
}

void project_grid(blitz::Array<float, 3> &grid, int nGrid, const char *out_filename)
{
    blitz::Array<float, 2> projected(nGrid, nGrid);
    for (int i = 0; i < nGrid; ++i)
    {
        for (int j = 0; j < nGrid; ++j)
        {
            projected(i, j) = max(grid(i, j, blitz::Range::all()));
        }
    }

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

struct particle
{
    float x, y, z;
};

bool compare_particles(const particle &a, const particle &b)
{
    return (a.x < b.x);
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

    int order = 4;
    if (argc > 3)
    {
        order = atoi(argv[3]);
    }

    const char *out_filename = (argc > 4) ? argv[4] : "projected.csv";

    // Init FFTW
    fftwf_mpi_init();
    ptrdiff_t start0, local0, start1, local1;
    // auto alloc_local = fftwf_mpi_local_size_3d(nGrid, nGrid, nGrid, MPI_COMM_WORLD, &local0, &start0);
    //EX1
    //auto alloc_local = fftw_mpi_local_size_3d_transposed(nGrid, nGrid, nGrid, MPI_COMM_WORLD, &local0, &start0, &local1, &start1);
    ptrdiff_t dim[2] = {nGrid, nGrid}
    

    //EX2
    auto alloc_local = fftwf_mpi_local_size_many_transposed(2,dim,nGrid+2, FFTW_MPI_DEFAULT_BLOCK, FFTW_MPI_DEFAULT_BLOCK,
    MPI_COMM_WORLD ,&local0 , &start0 ,&local1 , &start1)
    
    // Collect all start0 and local0
    std::vector<int> all_start0(N_rank);
    std::vector<int> all_local0(N_rank);
    MPI_Allgather(&start0, 1, MPI_INT, &all_start0[0], 1, MPI_INT, MPI_COMM_WORLD);
    MPI_Allgather(&local0, 1, MPI_INT, &all_local0[0], 1, MPI_INT, MPI_COMM_WORLD);

    // Fill particles x position to rank
    blitz::Array<int, 1> slab2rank(nGrid);
    for (int i = 0; i < N_rank; ++i)
    {
        for (int j = all_start0[i]; j < all_start0[i] + all_local0[i]; ++j)
        {
            slab2rank(j) = i;
        }
    }

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

    std::cerr << "Loading " << N_per << " particles" << std::endl;
    blitz::Array<float, 2> r(blitz::Range(i_start, i_end - 1), blitz::Range(0, 2));
    io.load(r);

    particle *prows = reinterpret_cast<particle *>(r.data());
    std::sort(prows, prows + r.rows(), compare_particles);

    std::vector<int> send_counts(N_rank, 0);
    int rank_to_send = 0;
    int send_to_last_count = 0;
    for (int pn = i_start; pn < i_end; ++pn)
    {
        float x = r(pn, 0);
        int i_start = floorf((x + 0.5) * nGrid - (order - 1) * 0.5);

        if (i_start < 0)
        {
            send_to_last_count++;
        }
        else
        {
            send_counts[slab2rank(i_start)]++;
        }
    }

    for (int i = 0; i < N_rank; i++)
    {
        send_counts[i] *= 3;
    }

    std::vector<int> send_displs(N_rank);
    send_displs[0] = send_to_last_count * 3;
    for (int i = 1; i < N_rank; ++i)
    {
        send_displs[i] = send_displs[i - 1] + send_counts[i - 1];
    }

    // Compute recv counts and displacements
    std::vector<int> recv_counts(N_rank);
    std::vector<int> recv_displs(N_rank);
    MPI_Alltoall(&send_counts[0], 1, MPI_INT, &recv_counts[0], 1, MPI_INT, MPI_COMM_WORLD);

    int total_recv = 0;
    for (int i = 0; i < N_rank; ++i)
    {
        if (i == 0)
            recv_displs[i] = 0;
        else
            recv_displs[i] = recv_displs[i - 1] + recv_counts[i - 1];
        total_recv += recv_counts[i];
    }
    int part_count = total_recv / 3;

    // =========================== Send ghost region to last rank
    int send_to_last_count_all[N_rank];
    int last_count_displ[N_rank];
    last_count_displ[0] = total_recv;
    MPI_Allgather(&send_to_last_count, 1, MPI_INT, &send_to_last_count_all[0], 1, MPI_INT, MPI_COMM_WORLD);
    int send_to_last_count_total = 0;
    for (int i = 0; i < N_rank; i++)
    {
        send_to_last_count_total += send_to_last_count_all[i];
        send_to_last_count_all[i] *= 3;
        if (i >= 1)
            last_count_displ[i] = last_count_displ[i - 1] + send_to_last_count_all[i - 1];
    }

    if (i_rank == N_rank - 1)
    {
        part_count += send_to_last_count_total;
    }
    // =========================== End of send ghost region to last rank

    printf("[Rank %d] Total recv = %d\n", i_rank, part_count);
    blitz::Array<float, 2> r_local(part_count, 3);
    MPI_Alltoallv(r.dataFirst(), &send_counts[0], &send_displs[0], MPI_FLOAT,
                  r_local.data(), &recv_counts[0], &recv_displs[0], MPI_FLOAT,
                  MPI_COMM_WORLD);
    // Send ghost region to last rank
    MPI_Gatherv(r.data(), send_to_last_count * 3, MPI_FLOAT,
                r_local.data(), send_to_last_count_all, last_count_displ, MPI_FLOAT, N_rank - 1,
                MPI_COMM_WORLD);

    int grid_start = start0;
    int grid_end = std::min(int(start0 + local0 - 1), int(nGrid)) + order;

    float *data = new (std::align_val_t(64)) float[(grid_end - grid_start) * nGrid * (nGrid + 2)];
    blitz::GeneralArrayStorage<3> storage;
    storage.base() = grid_start, 0, 0;
    blitz::Array<float, 3> grid_data(data, blitz::shape(grid_end - grid_start, nGrid, nGrid + 2), blitz::deleteDataWhenDone, storage);
    grid_data = 0.0;
    blitz::Array<float, 3> grid = grid_data(blitz::Range(grid_start, grid_end - 1), blitz::Range::all(), blitz::Range(0, nGrid - 1));
    blitz::Array<float, 3> ghost_region = grid(blitz::Range(grid_end - order + 1, grid_end - 1), blitz::Range::all(), blitz::Range::all());

    std::complex<float> *complex_data = reinterpret_cast<std::complex<float> *>(data);
    blitz::Array<std::complex<float>, 3> kdata(complex_data, blitz::shape(grid_end - grid_start, nGrid, nGrid / 2 + 1));
    std::cout << kdata.base() << std::endl;
    assign_mass(r_local, 0, part_count, nGrid, grid, order, grid_start, grid_end);
    printf("[Rank %d] Grid sum after mass assignment = %f\n", i_rank, sum(grid));

    MPI_Request req[3];
    // First split the MPI_COMM_WORLD communicator into (0,1), (2,3), (4,5)
    MPI_Comm newcomm1, newcomm2;
    int color = i_rank / 2;
    int key = (i_rank + 1) % 2;
    int send_rank, recv_rank;
    MPI_Comm_split(MPI_COMM_WORLD, color, key, &newcomm1);
    // Send and receive data in a ring fashion
    if (key == 0)
    {
        MPI_Ireduce(MPI_IN_PLACE, grid.data(), ghost_region.size(), MPI_FLOAT, MPI_SUM, 0, newcomm1, &req[0]);
    }
    else
    {
        MPI_Ireduce(ghost_region.data(), nullptr, ghost_region.size(), MPI_FLOAT, MPI_SUM, 0, newcomm1, &req[0]);
    }

    // Second split the communicator into (1,2), (3,4), (5,0)
    if (i_rank == N_rank - 1)
    {
        color = 0;
        key = 1;
    }
    else
    {
        color = (i_rank + 1) / 2;
        key = i_rank % 2;
    }

    MPI_Comm_split(MPI_COMM_WORLD, color, key, &newcomm2);
    if (key == 0)
    {
        MPI_Ireduce(MPI_IN_PLACE, grid.data(), ghost_region.size(), MPI_FLOAT, MPI_SUM, 0, newcomm2, &req[1]);
    }
    else
    {
        MPI_Ireduce(ghost_region.data(), nullptr, ghost_region.size(), MPI_FLOAT, MPI_SUM, 0, newcomm2, &req[1]);
    }

    if (N_rank % 2 == 1)
    {
        color = 0;
        // Odd number of proc
        MPI_Comm newcomm3;
        if (i_rank == N_rank - 2)
        {
            key = 1;
        }
        else if (i_rank == N_rank - 1)
        {
            key = 0;
        }
        else
        {
            key = 1;
            color = MPI_UNDEFINED;
        }

        if (key == 0)
        {
            MPI_Ireduce(MPI_IN_PLACE, grid.data(), ghost_region.size(), MPI_FLOAT, MPI_SUM, 0, newcomm3, &req[2]);
        }
        else
        {
            MPI_Ireduce(ghost_region.data(), nullptr, ghost_region.size(), MPI_FLOAT, MPI_SUM, 0, newcomm3, &req[2]);
        }
        MPI_Waitall(3, &req[0], MPI_STATUSES_IGNORE);
    }
    else
    {
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);
    }

    printf("[Rank %d] After all reduction grid sum = %f\n", i_rank, sum(grid(blitz::Range(start0, grid_end - order), blitz::Range::all(), blitz::Range::all())));

    // Overdensity
    grid = grid - 1;

    //fftwf_plan plan = fftwf_mpi_plan_dft_r2c_3d(nGrid, nGrid, nGrid, data, (fftwf_complex *)complex_data, MPI_COMM_WORLD, FFTW_MPI_TRANSPOSED_OUT | FFTW_ESTIMATE);
    fftwf_plan plan2d = fftwf_mpi_plan_dft_r2c_2d(nGrid, nGrid, nGrid, data, (fftwf_complex *)complex_data, MPI_COMM_WORLD, FFTW_ESTIMATE);

    printf("[Rank %d] Plan created\n", i_rank);
    for (int i; i<grid_end - grid_start - order; i++) {
        float* in = &grid(i,0,0);
        fftw_complex* out = reinterpret_cast<fftwf_complex>(&data(i,0,0))
        fftw_execute_dft_r2c(plan2d, in, out);
    }

    fftwf_destroy_plan(plan2d);

    fftwf_plan planT = fftw_mpi_plan_many_transpose(nGrid, nGrid, nGrid+2,FFTW_MPI_DEFAULT_BLOCK,FFTW_MPI_DEFAULT_BLOCK,grid.data(),grid.data(),MPI_COMM_WORLD,FFTW_ESTIMATE)

    fftwf_execute(planT);
    printf("[Rank %d] Plan executed\n", i_rank);
    fftwf_destroy_plan(planT);
    printf("[Rank %d] Plan destroyed\n", i_rank);


    blitz::Array<std::complex<float>,3> kdata(complex_data,blitz::shape(nGrid, grid_end, grid_start - order, nGrid/2 + 1));
    fftwf_complex* input1d = reinterpret_cast<fftw_complex*>(kdata.data());
    int rank = 1;
    int n[1] = {nGrid};
    int howmany = nGrid / 2 + 1;
    int idist = 1;
    int odist = 1;
    int istride = nGrid / 2 + 1;
    int ostride = nGrid / 2 + 1;
    int* inembed = n;
    int* onembed = n;

    fftw_plan plan1d = fftw_plan_many_dft(rank,n,howmany,input1d, inembed, istride, idist, input1d, onembed, ostride, odist, FFTW_FORWARD,FFTW_ESTIMATE);
    for (int i = 0; i < grid_end - grid_start - order; i++) {
        fftw_complex* in = reinterpret_cast<fftw_complex*>(&kdata(0,i,0));
        fftw_execute_dft(plan1d, in, in);
    }

    printf("1d fft done\n");

    // Linear binning is 1
    // Variable binning is 2
    // Log binning is 3
    const int binning = 2;

    int n_bins = 80;
    if (binning == 1)
    {
        n_bins = nGrid;
    }
    std::vector<float> fPower(n_bins, 0.0);
    std::vector<int> nPower(n_bins, 0);
    float k_max = sqrt((nGrid / 2.0) * (nGrid / 2.0) * 3.0);

    // loop over δ(k) and compute k from kx, ky and kz
    for (int i = 0; i < grid_end - grid_start; i++)
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
                if (i_bin == fPower.size())
                    i_bin--;
                fPower[i_bin] += std::norm(kdata(i, j, l));
                nPower[i_bin] += 1;
            }
        }
    }

    if (i_rank == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, fPower.data(), fPower.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Reduce(fPower.data(), nullptr, fPower.size(), MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (i_rank == 0)
    {
        MPI_Reduce(MPI_IN_PLACE, nPower.data(), nPower.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    else
    {
        MPI_Reduce(nPower.data(), nullptr, nPower.size(), MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }

    if (i_rank == 0)
    {
        save_binning(binning, fPower, nPower);
    }

    MPI_Finalize();
}