#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define TILE_DIM 32
#define BLOCK_ROWS 32

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) \
    { \
        fprintf(stderr, "Failed with error code %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

__global__
void transposeDiagonalBlocks(float *odata, float *idata, int width, int height)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int blockIdx_x, blockIdx_y;

    if (width == height)
    {
        blockIdx_y = blockIdx.x;
        blockIdx_x = (blockIdx.x + blockIdx.y) % gridDim.x;
    }
    else
    {
        int bid = blockIdx.y * gridDim.x + blockIdx.x;
        blockIdx_y = bid % gridDim.y;
        blockIdx_x = ((bid / gridDim.y) + blockIdx_y) % gridDim.x;
    }

    int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
    int index_in = yIndex * width + xIndex;

    int i;
    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }
    __syncthreads();

    xIndex = blockIdx_y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx_x * TILE_DIM + threadIdx.y;
    int index_out = yIndex * height + xIndex;

    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
    }

}

__global__
void transposeNoBankConflicts(float *odata, float *idata, const int nx, const int ny)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];
    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int j;

    for (j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;
    for (j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

__global__
void transposeFineGrained(float *odata, float *idata, int width, int height)
{
    __shared__ float block[TILE_DIM][TILE_DIM + 1];
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index = yIndex * width + xIndex;

    int i;
    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        block[threadIdx.y + i][threadIdx.x] = idata[index + i * width];
    }
    __syncthreads();

    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[index + i * height] = block[threadIdx.x][threadIdx.y + i];
    }
}

__global__
void transposeCoarseGrained(float *odata, float *idata, int width, int height)
{
    __shared__ float block[TILE_DIM][TILE_DIM + 1];
    int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
    int index_in = yIndex * width + xIndex;

    int i;
    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        block[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
    }
    __syncthreads();

    xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
    yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
    int index_out = (yIndex) * height + xIndex;
    for (i = 0; i < TILE_DIM; i += BLOCK_ROWS)
    {
        odata[index_out + i * height] = block[threadIdx.y + i][threadIdx.x];
    }
}

__global__
void transposeCoalesced(float *odata, float *idata, const int nx, const int ny)
{
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = gridDim.x * TILE_DIM;
    int j = 0;
    for (j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        tile[threadIdx.y + j][threadIdx.x] = idata[(y + j) * width + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (j = 0; j < TILE_DIM; j += BLOCK_ROWS)
    {
        odata[(y + j) * width + x] = tile[threadIdx.x][threadIdx.y + j];
    }
}

int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1 << 13;
    int ny = 1 << 13;

    int blockx = TILE_DIM;
    int blocky = BLOCK_ROWS;

    size_t nBytes = nx * ny * sizeof(float);

    dim3 block(blockx, blocky, 1);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y, 1);

    float *h_A = (float *) malloc(nBytes);
    float *gpuRef = (float *) malloc(nBytes);

    int i;
    srand(time(0));
    for (i = 0; i < nx * ny; ++i)
    {
        h_A[i] = rand() % 10000;
    }

    float *d_A, *d_C;
    CHECK(cudaMalloc((float **) &d_A, nBytes));
    CHECK(cudaMalloc((float **) &d_C, nBytes));

    CHECK(cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice));
    void (*kernel)(float *, float *, int, int);
    int iKernel = 4;
    char *kernelName;

    if (argc > 0)
    {
        iKernel = atoi(argv[1]);
    }

    switch (iKernel)
    {
        case 0:
            kernel = &transposeCoalesced;
            kernelName = "Coalesced_transpose";
            break;
        case 1:
            kernel = &transposeCoarseGrained;
            kernelName = "Coarse_Grained_Transpose";
            break;
        case 2:
            kernel = &transposeFineGrained;
            kernelName = "Fine_Grained_Transpose";
            break;
        case 3:
            kernel = &transposeNoBankConflicts;
            kernelName = "Transpose_without_Bank_Conflicts";
            break;
        default:
            kernel = &transposeDiagonalBlocks;
            kernelName = "Transpose_with_Diagonal_Blocks";
    }

    kernel<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    int j;
    if (iKernel != 1 && iKernel != 2)
    {
        for (i = 0; i < nx; ++i)
        {
            for (j = 0; j < ny; ++j)
            {
                if (fabs(gpuRef[i * nx + j] - h_A[j * ny + i]) > 1e-5)
                {
                    fprintf(stderr, "Error in the matrix transposition kernel %s.\n", kernelName);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }

    printf("TEST PASSED with kernel %s!\n", kernelName);
    return 0;
}