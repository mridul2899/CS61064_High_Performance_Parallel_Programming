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

__global__ void transposeNoBankConflicts(float *odata, float *idata, const int nx, const int ny)
{
    extern __shared__ float tile[];
    int block_dim = blockDim.x;
    int x = blockIdx.x * block_dim + threadIdx.x;
    int y = blockIdx.y * block_dim + threadIdx.y;
    int width = gridDim.x * block_dim;
    tile[threadIdx.y * (block_dim + 1) + threadIdx.x] = idata[y * width + x];
    __syncthreads();

    x = blockIdx.y * block_dim + threadIdx.x;
    y = blockIdx.x * block_dim + threadIdx.y;
    odata[y * width + x] = tile[(threadIdx.x) * (block_dim + 1) + threadIdx.y];
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
                // printf("Hello\n");
                scanf("%f", &h_i[i * n + j]);
            }
        }

        CHECK(cudaMalloc((void **) &d_i, n * n * sizeof(float)));
        CHECK(cudaMalloc((void **) &d_o, n * n * sizeof(float)));

        CHECK(cudaMemcpy(d_i, h_i, n * n * sizeof(float), cudaMemcpyHostToDevice));
        int block_dim = 32;
        dim3 grid((n + block_dim - 1) / block_dim, (n + block_dim - 1) / block_dim, 1);
        dim3 block(block_dim, block_dim, 1);
        int shared_size = block_dim * (block_dim + 1) * sizeof(float);

        transposeNoBankConflicts<<<grid, block, shared_size>>>(d_o, d_i, n, n);
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