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

__device__ void warpReduce(volatile float *tile, int tid)
{
    tile[tid] += tile[tid + 32];
    tile[tid] += tile[tid + 16];
    tile[tid] += tile[tid + 8];
    tile[tid] += tile[tid + 4];
    tile[tid] += tile[tid + 2];
    tile[tid] += tile[tid + 1];
}

__global__ void reduce(float *d_i, float *d_o, int n)
{
    extern __shared__ float tile[];
    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x * 2 + tid;
    tile[tid] = d_i[pos] + d_i[pos + blockDim.x];
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            tile[tid] += tile[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        warpReduce(tile, tid);
    }
    if (tid == 0)
    {
        d_o[blockIdx.x] = tile[0];
    }
}

__global__ void dotproduct(float *d_i1, float *d_i2, float *d_o, int n)
{
    extern __shared__ float tile[];
    int tid = threadIdx.x;
    int pos = blockIdx.x * blockDim.x + tid;
    tile[tid] = (pos < n)? (d_i1[pos] * d_i2[pos]) : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 32; s >>= 1)
    {
        if (tid < s)
        {
            tile[tid] += tile[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32)
    {
        warpReduce(tile, tid);
    }
    if (tid == 0)
    {
        d_o[blockIdx.x] = tile[0];
    }
}

int main()
{
    int t;
    scanf("%d", &t);
    while (t--)
    {
        float *h_i1, *d_i1;
        float *h_i2, *d_i2;
        float *h_o1, *d_o1;

        int n;
        int i;

        scanf("%d", &n);
        h_i1 = (float *) malloc(n * sizeof(float));
        h_i2 = (float *) malloc(n * sizeof(float));

        for (i = 0; i < n; ++i)
        {
            scanf("%f", &h_i1[i]);
        }
        for (i = 0; i < n; ++i)
        {
            scanf("%f", &h_i2[i]);
        }

        CHECK(cudaMalloc((void **) &d_i1, n * sizeof(float)));
        CHECK(cudaMalloc((void **) &d_i2, n * sizeof(float)));

        CHECK(cudaMemcpy(d_i1, h_i1, n * sizeof(float), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(d_i2, h_i2, n * sizeof(float), cudaMemcpyHostToDevice));

        int k = 256;
        int m = (n + k - 1) / k;

        CHECK(cudaMalloc((void **) &d_o1, m * sizeof(float)));
        int shared_size = k * sizeof(float);

        dotproduct<<<m, k, shared_size>>>(d_i1, d_i2, d_o1, n);
        CHECK(cudaGetLastError());

        while (m >= 1024)
        {
            CHECK(cudaMemcpy(d_i1, d_o1, m * sizeof(float), cudaMemcpyDeviceToDevice));

            shared_size = k * 2 * sizeof(float);
            int grid_dim = (((m + k - 1) / k) + 1) / 2;

            reduce<<<grid_dim, k, shared_size>>>(d_i1, d_o1, m);
            CHECK(cudaGetLastError());

            m = grid_dim;
        }
        h_o1 = (float *) malloc(m * sizeof(float));

        CHECK(cudaMemcpy(h_o1, d_o1, m * sizeof(float), cudaMemcpyDeviceToHost));
        double final_sum = 0;
        for (i = 0; i < m; ++i)
        {
            final_sum += h_o1[i];
        }

        printf("%.2lf\n\n", final_sum);
    }
}
