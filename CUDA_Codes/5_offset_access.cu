#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void offset_access(float *a, int s, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid + s < n)
    {
        a[tid + s] = a[tid + s] + 1;
    }
}

__global__ void strided_access(float *a, int s, int n)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid * s < n)
    {
        a[tid * s] = a[tid * s] + 1;
    }
}

int main()
{
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);
    cudaError_t err = cudaSuccess;
    int nMB = 128;
    float ms;
    int blockSize = 1024;
    int n = nMB * 1024 * 1024 / sizeof(float);
    float *d_a;
    err = cudaMalloc(&d_a, n * sizeof(float));
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Memory not allocated (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    int i;
    for (i = 0; i <= 32; ++i)
    {
        err = cudaMemset(d_a, 0.0, n * sizeof(float));
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Data not written (error code %s)!\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);
        }
        cudaEventRecord(startEvent);
        strided_access<<<n / blockSize, blockSize>>>(d_a, i, n);
        cudaEventRecord(stopEvent);
        cudaEventSynchronize(stopEvent);
        cudaEventElapsedTime(&ms, startEvent, stopEvent);
        printf("%d, %f\n", i, ms);
    }
    printf("Just checking\n");
}