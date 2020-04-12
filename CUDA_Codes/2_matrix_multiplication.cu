#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__
void mul(float *d_A, float *d_B, float *d_C, int n);

void matMul(float **h_Mat1, float **h_Mat2, float **h_Mat3, int n);

int main()
{
    int n;
    int i, j;
    float **h_Mat1, **h_Mat2, **h_Mat3;

    printf("Enter the dimension of square matrix, n for n X n: ");
    scanf("%d", &n);

    h_Mat1 = (float **) malloc(n * sizeof(float *));
    for (i = 0; i < n; ++i)
    {
        h_Mat1[i] = (float *) malloc(n * sizeof(float));
    }

    h_Mat2 = (float **) malloc(n * sizeof(float *));
    for (i = 0; i < n; ++i)
    {
        h_Mat2[i] = (float *) malloc(n * sizeof(float));
    }

    h_Mat3 = (float **) malloc(n * sizeof(float *));
    for (i = 0; i < n; ++i)
    {
        h_Mat3[i] = (float *) malloc(n * sizeof(float));
    }

    srand(time(0));
    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            h_Mat1[i][j] = rand() % 1000;
            h_Mat2[i][j] = rand() % 1000;
        }
    }

    matMul(h_Mat1, h_Mat2, h_Mat3, n);

    return 0;
}

__global__
void mul(float *d_A, float *d_B, float *d_C, int n)
{
    int i, j, k;

    i = blockIdx.y * blockDim.y + threadIdx.y;
    j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= n || j >= n)
    {
        return;
    }
    d_C[i * n + j] = 0;
    for (k = 0; k < n; ++k)
    {
        d_C[i * n + j] += d_A[i * n + k] * d_B[k * n + j];
    }

    return;
}

void matMul(float **h_Mat1, float **h_Mat2, float **h_Mat3, int n)
{
    int size = n * n * sizeof(float);
    int i, j, k;
    float *h_A, *h_B, *h_C;
    float *d_A = NULL, *d_B = NULL, *d_C = NULL;
    cudaError_t err = cudaSuccess;

    h_A = (float *) malloc(size);
    h_B = (float *) malloc(size);
    h_C = (float *) malloc(size);

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            h_A[i * n + j] = h_Mat1[i][j];
            h_B[i * n + j] = h_Mat2[i][j];
        }
    }

    err = cudaMalloc((void **) &d_A, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector A (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **) &d_B, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector B (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **) &d_C, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector C (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector A from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector B from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    printf("Launching CUDA mul kernel with (%d, %d, %d) blocks and (%d, %d, %d) threads per block.\n", (n + 15) / 16, (n + 15) / 16, 1, 16, 16, 1);
    dim3 grid((n + 15) / 16, (n + 15) / 16, 1);
    dim3 block(16, 16, 1);
    mul<<<block, grid>>>(d_A, d_B, d_C, n);
    
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch mul kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector C from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    for (i = 0; i < n; ++i)
    {
        for (j = 0; j < n; ++j)
        {
            h_Mat3[i][j] = 0;
            for (k = 0; k < n; ++k)
            {
                h_Mat3[i][j] += h_Mat1[i][k] * h_Mat2[k][j];
            }
            if (fabs(h_C[i * n + j] - h_Mat3[i][j]) > 1e-5)
            {
                fprintf(stderr, "Result verification failed at element (%d, %d)!\n", i, j);
                exit(EXIT_FAILURE);
            }
            h_Mat3[i][j] = h_C[i * n + j];
        }
    }
    printf("TEST PASSED\n");
    
    return;
}
