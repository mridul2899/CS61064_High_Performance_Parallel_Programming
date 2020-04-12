#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__
void sumTriangle(float *M,  float *V, int N)
{
    int j = threadIdx.x;
    int i;
    float sum = 0; 
    for (i = 0; i <= j; ++i)
    {
        sum += M[i * N + j];
    }
    V[j] = sum;
    __syncthreads();

    if (j == N - 1)
    {
        sum = 0.0;
        for (i = 0; i < N; ++i)
        {
            sum += V[i];
        }
        V[N] = sum;
    }
}

__global__
void sumTriangle2(float *M, float *V, int N)
{
    int j = threadIdx.x;
    float sum = 0.0;
    int i;
    for (i = 0; i <= j; ++i)
    {
        if (i % 2 == 0)
        {
            sum += M[i * N + j];
        }
    }
    V[j] = sum;
    __syncthreads();

    if (j == N - 1)
    {
        sum = 0;
        for (i = 0; i < N; ++i)
        {
            sum += V[i];
        }
        V[N] = sum;
    }
}

__global__
void sumTriangle3(float *M, float *V, int N)
{
    int j = threadIdx.x;
    int i;
    float sum = 0;
    for (i = 0; i <= j; ++i)
    {
        sum += M[i * N + j];
    }
    V[j] = sum;
    __syncthreads();

    int s;
    for (s = 1; s < N; s *= 2)
    {
        if (j % (2 * s) == 0 && j + s < N)
        {
            V[j] += V[j + s];
        }
        __syncthreads();
    }
    V[N] = V[0];
}

int main()
{
    int N = 11;
    int size_M = N * N;
    int size_V = N + 1;

    float *M, *V;
    M = (float *) malloc(sizeof(float) * size_M);
    V = (float *) malloc(sizeof(float) * size_V);

    int i, j;
    srand(time(0));
    V[0] = 0;
    for (i = 0; i < N; ++i)
    {
        V[i + 1] = 0;
        for (j = 0; j < N; ++j)
        {
            M[i * N + j] = rand() % 10000;
        }
    }

    cudaError_t err = cudaSuccess;

    float *d_M = NULL, *d_V = NULL;

    err = cudaMalloc((void **) &d_M, size_M * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error allocating device vector d_M (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **) &d_V, size_V * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error allocating device vector d_V (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_M, M, size_M * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error copying vector M from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_V, V, size_V * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error copying vector V from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 grid(1, 1, 1);
    dim3 block(N, 1, 1);
    printf("Lauching cuda kernel sumTriangle with blocks: (%d, %d, %d) and threads: (%d, %d, %d).\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    sumTriangle3<<<grid, block>>>(d_M, d_V, N);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error launching kernel sumTriangle (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(V, d_V, sizeof(float) * size_V, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Error copying vector V from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float sum_all = 0;
    float sum;
    for (j = 0; j < N; ++j)
    {
        sum = 0;
        for (i = 0; i <= j; i += 1)
        {
            sum += M[i * N + j];
        }
        // if (fabs(sum - V[j]) > 1e-5)
        // {
            // fprintf(stderr, "Error in kernel's computation - kernel gives incorrect results for triangle sum.\n");
            // exit(EXIT_FAILURE);
        // }
        sum_all += sum;
    }
    if (fabs(sum_all - V[N]) > 1e-5)
    {
        fprintf(stderr, "Error in kernel's computation - kernel gives incorrect result for overall sum.\n");
        exit(EXIT_FAILURE);
    }

    printf("TEST PASSED.\n");
    return 0;
}