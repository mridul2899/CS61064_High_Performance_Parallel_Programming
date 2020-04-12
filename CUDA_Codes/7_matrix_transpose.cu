#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

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
void transposeNaiveRow(float *out, float *in, const int nx, int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny)
    {
        out[ix * ny + iy] = in[iy * nx + ix];
    }
}

__global__
void transposeNaiveCol(float *out, float *in, const int nx, int ny)
{
    unsigned int ix = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int iy = blockDim.y * blockIdx.y + threadIdx.y;
    if (ix < nx && iy < ny)
    {
        out[iy * nx + ix] = in[ix * ny + iy];
    }
}

int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting transpose at ", argv[0]);
    printf("device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    int nx = 1 << 13;
    int ny = 1 << 13;

    int iKernel = 0;
    int blockx = 32;
    int blocky = 32;

    if (argc > 1)
    {
        iKernel = atoi(argv[1]);
    }

    size_t nBytes = nx * ny * sizeof(float);

    dim3 block(blockx, blocky);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    float *h_A = (float *) malloc(nBytes);
    float *hostRef = (float *) malloc(nBytes);
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
    char *kernelName;

    switch (iKernel)
    {
        case 0:
            kernel = &transposeNaiveRow;
            kernelName = "NaiveRow";
            break;
        case 1:
            kernel = &transposeNaiveCol;
            kernelName = "NaiveCol";
            break;
    }

    kernel<<<grid, block>>>(d_C, d_A, nx, ny);
    CHECK(cudaGetLastError());
    CHECK(cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost));

    int j;
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
    printf("TEST PASSED with kernel %s!\n", kernelName);
    return 0;
}