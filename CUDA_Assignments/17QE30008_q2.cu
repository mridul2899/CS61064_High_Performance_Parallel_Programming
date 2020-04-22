#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM_X 32
#define TILE_DIM_Y 16

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
    { \
        fprintf(stderr, "Failed with error code %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__ void transpose(float *odata, float *idata, const int nx, const int ny)
{
    __shared__ float tile[TILE_DIM_Y][TILE_DIM_X];
    int x = blockIdx.x * TILE_DIM_X + threadIdx.x;
    int y = blockIdx.y * TILE_DIM_Y + threadIdx.y;
    int width = gridDim.x * TILE_DIM_X;

    tile[threadIdx.y][threadIdx.x] = idata[y * width + x];
    __syncthreads();

    x = blockIdx.y * TILE_DIM_Y + threadIdx.x % TILE_DIM_Y;
    y = blockIdx.x * TILE_DIM_X + threadIdx.y * (TILE_DIM_X / TILE_DIM_Y) + threadIdx.x / TILE_DIM_Y;
    odata[y * width + x] = tile[threadIdx.x % TILE_DIM_Y][threadIdx.y * (TILE_DIM_X / TILE_DIM_Y) + threadIdx.x / TILE_DIM_Y];
}

int main()
{
    int t;
    scanf("%d", &t);
    while (t--)
    {
        float *h_i, *d_i;
        float *h_o, *d_o;
        int n;
        int i, j;

        scanf("%d", &n);
        h_i = (float *) malloc(n * n * sizeof(float));
        h_o = (float *) malloc(n * n * sizeof(float));

        for (i = 0; i < n; ++i)
        {
            for (j = 0; j < n; ++j)
            {
                scanf("%f", &h_i[i * n + j]);
            }
        }

        CHECK(cudaMalloc((void **) &d_i, n * (n + 2) * sizeof(float)));
        CHECK(cudaMalloc((void **) &d_o, n * n * sizeof(float)));

        CHECK(cudaMemcpy(d_i, h_i, n * n * sizeof(float), cudaMemcpyHostToDevice));
        int block_dim_x = 32;
        dim3 grid((n + block_dim_x - 1) / block_dim_x, (n + TILE_DIM_Y - 1) / TILE_DIM_Y, 1);
        dim3 block(block_dim_x, TILE_DIM_Y, 1);

        transpose<<<grid, block>>>(d_o, d_i, n, n);
        CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(h_o, d_o, n * n * sizeof(float), cudaMemcpyDeviceToHost));

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