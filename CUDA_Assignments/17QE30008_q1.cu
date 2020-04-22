#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
    { \
        fprintf(stderr, "Failed with error code %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__device__ __constant__ float d_Filter[3 * 3];

__global__ void convolute(float *d_i, float *d_o, int dim, int block_dim)
{
    int tile_dim = block_dim + 2;
    extern __shared__ float sData[];

    int n_subs = (tile_dim + blockDim.y - 1) / blockDim.y;
    int tid_x = blockIdx.x * block_dim + threadIdx.x;
    int i;

    int t_e_col = (blockIdx.x * block_dim + tile_dim < dim)? (blockIdx.x * block_dim + tile_dim) : dim;
    int t_e_row = (blockIdx.y * block_dim + tile_dim < dim)? (blockIdx.y * block_dim + tile_dim) : dim;

    int t_row;
    int tid_y;
    for (i = 0; i < n_subs; ++i)
    {
        t_row = threadIdx.y + i * blockDim.y;
        tid_y = blockIdx.y * block_dim + t_row;

        if (tid_x < t_e_col && tid_y < t_e_row)
        {
            sData[t_row * tile_dim + threadIdx.x] = d_i[tid_y * dim + tid_x];
        }
    }
    __syncthreads();

    for (int sub = 0; sub < n_subs; ++sub)
    {
        t_row = threadIdx.y + sub * blockDim.y;
        tid_y = blockIdx.y * block_dim + t_row;

        if (tid_x >= blockIdx.x * block_dim + 1 && tid_x < t_e_col - 1 && tid_y >= blockIdx.y * block_dim + 1 && tid_y < t_e_row - 1)
        {
            d_o[(tid_y - 1) * (dim - 2) + tid_x - 1] = 0.0;
            for (int i = -1; i <= 1; ++i)
            {
                for (int j = -1; j <= 1; ++j)
                {
                    d_o[(tid_y - 1) * (dim - 2) + tid_x - 1] += sData[t_row * tile_dim + threadIdx.x + i * tile_dim + j] * d_Filter[(i + 1) * 3 + (j + 1)];
                }
            }
        }
    }
}

int main()
{
    int t;
    int i;

    float *h_f;
    h_f = (float *) malloc(9 * sizeof(float));
    for (i = 0; i < 9; ++i)
    {
        h_f[i] = 1.0 / 9.0;
    }

    CHECK(cudaMemcpyToSymbol(d_Filter, h_f, 9 * sizeof(float), 0, cudaMemcpyHostToDevice));

    scanf("%d", &t);
    while (t--)
    {
        float *h_i, *d_i;
        float *h_o, *d_o;
        int n;

        scanf("%d", &n);
        h_i = (float *) malloc((n + 2) * (n + 2) * sizeof(float));
        h_o = (float *) malloc(n * n * sizeof(float));

        for (i = 0; i < (n + 2) * (n + 2); ++i)
        {
            h_i[i] = 0;
        }

        for (i = 0; i < n; ++i)
        {
            for (int j = 0; j < n; ++j)
            {
                scanf("%f", &h_i[(i + 1) * (n + 2) + j + 1]);
            }
        }

        CHECK(cudaMalloc((void **) &d_i, (n + 2) * (n + 2) * sizeof(float)));
        CHECK(cudaMalloc((void **) &d_o, n * n * sizeof(float)));

        CHECK(cudaMemcpy(d_i, h_i, (n + 2) * (n + 2) * sizeof(float), cudaMemcpyHostToDevice));
        int block_dim = 32;
        int tile_dim = block_dim + 2;
        dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim, 1);
        dim3 block((block_dim + 2), 8, 1);
        int shared_size = tile_dim * tile_dim * sizeof(float);

        convolute<<<grid, block, shared_size>>>(d_i, d_o,  n + 2, block_dim);
        CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(h_o, d_o, n * n * sizeof(float), cudaMemcpyDeviceToHost));

        /*
        for (i = 0; i < (n + 2) * (n + 2); ++i)
        {
            printf("%4.2f ", h_i[i]);
            if (i % (n + 2) == n + 1)
            {
                printf("\n");
            }
        }
        */

        for (i = 0; i < n * n; ++i)
        {
            printf("%4.2f ", h_o[i]);
            if (i % n == n - 1)
            {
                printf("\n");
            }
        }
        printf("\n");
    }
}