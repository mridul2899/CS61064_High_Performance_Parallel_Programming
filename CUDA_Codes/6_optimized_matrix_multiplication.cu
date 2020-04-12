#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

const int TILE_WIDTH = 4;

__global__
void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;
    float Pvalue = 0;

    for (int m = 0; m < Width / TILE_WIDTH; ++m)
    {
        Mds[ty][tx] = d_M[Row * Width + m * TILE_WIDTH + tx];
        Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();
        for (int k = 0; k < TILE_WIDTH; ++k)
        {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();
    }

    d_P[Row * Width + Col] = Pvalue;
}

int main()
{
    int Width = 12;
    float *h_M, *h_N, *h_P;

    int size = sizeof(float) * Width * Width;
    h_M = (float *) malloc(size);
    h_N = (float *) malloc(size);
    h_P = (float *) malloc(size);

    int i, j;
    srand(time(0));

    for (i = 0; i < Width; ++i)
    {
        for (j = 0; j < Width; ++j)
        {
            h_M[i * Width + j] = rand() % 1000;
            h_N[i * Width + j] = rand() % 1000;
        }
    }

    cudaError_t err = cudaSuccess;
    float *d_M, *d_N, *d_P;

    err = cudaMalloc((void **) &d_M, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector M (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMalloc((void **) &d_N, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector N (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);#include <cuda.h>
        #include <cuda_runtime.h>
        #include <stdio.h>
        #include <stdlib.h>
        #include <time.h>

        const int TILE_WIDTH = 4;

        __global__
        void MatrixMulKernel(float *d_M, float *d_N, float *d_P, int Width)
        {
            __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
            __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

            int bx = blockIdx.x;
            int by = blockIdx.y;
            int tx = threadIdx.x;
            int ty = threadIdx.y;

            int Row = by * TILE_WIDTH + ty;
            int Col = bx * TILE_WIDTH + tx;
            float Pvalue = 0;

            for (int m = 0; m < Width / TILE_WIDTH; ++m)
            {
                Mds[ty][tx] = d_M[Row * Width + m * TILE_WIDTH + tx];
                Nds[ty][tx] = d_N[(m * TILE_WIDTH + ty) * Width + Col];
                __syncthreads();
                for (int k = 0; k < TILE_WIDTH; ++k)
                {
                    Pvalue += Mds[ty][k] * Nds[k][tx];
                }
                __syncthreads();
            }

            d_P[Row * Width + Col] = Pvalue;
        }

        int main()
        {
            int Width = 12;
            float *h_M, *h_N, *h_P;

            int size = sizeof(float) * Width * Width;
            h_M = (float *) malloc(size);
            h_N = (float *) malloc(size);
            h_P = (float *) malloc(size);

            int i, j;
            srand(time(0));

            for (i = 0; i < Width; ++i)
            {
                for (j = 0; j < Width; ++j)
                {
                    h_M[i * Width + j] = rand() % 1000;
                    h_N[i * Width + j] = rand() % 1000;
                }
            }

            cudaError_t err = cudaSuccess;
            float *d_M, *d_N, *d_P;

            err = cudaMalloc((void **) &d_M, size);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector M (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            err = cudaMalloc((void **) &d_N, size);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector N (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            err = cudaMalloc((void **) &d_P, size);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to allocate device vector V (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }


            err = cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector M from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }
            err = cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector N from host to device (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            dim3 grid(Width / TILE_WIDTH, Width / TILE_WIDTH, 1);
            dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
            printf("Launching MatrixMulKernel kernel with grid dimensions: (%d, %d, %d) and block dimensions: (%d, %d, %d).\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
            MatrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, Width);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to launch kernel MatrixMulKernel (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            err = cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                fprintf(stderr, "Failed to copy vector P from device to host (error code %s)!\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);
            }

            float sum = 0;
            int k;
            for (i = 0; i < Width; ++i)
            {
                for (j = 0; j < Width; ++j)
                {
                    sum = 0;
                    for (k = 0; k < Width; ++k)
                    {
                        sum += h_M[i * Width + k] * h_N[k * Width + j];
                    }
                    if (fabs(sum - h_P[i * Width + j]) > 1e-5)
                    {
                        fprintf(stderr, "%f %f\n", sum, h_P[i * Width + j]);
                        fprintf(stderr, "Kernel MatMulKernel does not multiply the matrices properly.\n");
                        exit(EXIT_FAILURE);
                    }
                }
            }
            printf("TEST PASSED.\n");
        }
    }
    err = cudaMalloc((void **) &d_P, size);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector V (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }


    err = cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector M from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err = cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector N from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    dim3 grid(Width / TILE_WIDTH, Width / TILE_WIDTH, 1);
    dim3 block(TILE_WIDTH, TILE_WIDTH, 1);
    printf("Launching MatrixMulKernel kernel with grid dimensions: (%d, %d, %d) and block dimensions: (%d, %d, %d).\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
    MatrixMulKernel<<<grid, block>>>(d_M, d_N, d_P, Width);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch kernel MatrixMulKernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector P from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    float sum = 0;
    int k;
    for (i = 0; i < Width; ++i)
    {
        for (j = 0; j < Width; ++j)
        {
            sum = 0;
            for (k = 0; k < Width; ++k)
            {
                sum += h_M[i * Width + k] * h_N[k * Width + j];
            }
            if (fabs(sum - h_P[i * Width + j]) > 1e-5)
            {
                fprintf(stderr, "%f %f\n", sum, h_P[i * Width + j]);
                fprintf(stderr, "Kernel MatMulKernel does not multiply the matrices properly.\n");
                exit(EXIT_FAILURE);
            }
        }
    }
    printf("TEST PASSED.\n");
}