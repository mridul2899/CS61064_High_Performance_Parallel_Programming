/*
    Name: Mridul Agarwal
    Roll Number: 17QE30008
    Assignment 1: Linear Transformations
    The program executes the CUDA kernels one after another sequentially and prints the final output
    Also stores the final output to the file 'output.txt'
*/

#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void process_kernel1(const float *input1, const float *input2, float *output, int datasize)
{
    int numblock = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * (gridDim.x) + blockIdx.x;
    int threadnum = numblock * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int i = threadnum;
    if (i < datasize)
    {
        output[i] = sin(input1[i]) + cos(input2[i]);
    }
    return;
}

__global__ void process_kernel2(const float *input, float *output, int datasize)
{
    int numblock = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * (gridDim.x) + blockIdx.x;
    int threadnum = numblock * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int i = threadnum;
    if (i < datasize)
    {
        output[i] = log(input[i]);
    }
    return;
}

__global__ void process_kernel3(const float *input, float *output, int datasize)
{
    int numblock = blockIdx.z * (gridDim.x * gridDim.y) + blockIdx.y * (gridDim.x) + blockIdx.x;
    int threadnum = numblock * blockDim.x * blockDim.y * blockDim.z + threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
    int i = threadnum;
    if (i < datasize)
    {
        output[i] = sqrt(input[i]);
    }
    return;
}

int main(void)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    int numElements = 32 * 32 * 1 * 4 * 2 * 2;

    size_t size = numElements * sizeof(float);

    // Allocate the host vector input1
    float *input1 = (float *) malloc(size);

    // Allocate the host vector input2
    float *input2 = (float *) malloc(size);

    // Allocate the host vector output for kernel 1
    float *output = (float *) malloc(size);

    // Allocate the host vector output for kernel 2
    float *output2 = (float *) malloc(size);
 
     // Allocate the host vector output for kernel 3
    float *output3 = (float *) malloc(size);

    // Verify that allocations succeeded
    if (input1 == NULL || input2 == NULL || output == NULL || output2 == NULL || output3 == NULL)
    {
        fprintf(stderr, "Failed to allocate host vectors!\n");
        exit(EXIT_FAILURE);
    }

    // Accept vector inputs from the user
    for (int i = 0; i < numElements; ++i)
    {
        scanf("%f", &input1[i]);
    }
    for (int i = 0; i < numElements; ++i)
    {
        scanf("%f", &input2[i]);
    }
    
    // Allocate the device input vector input1
    float *d_input1 = NULL;
    err = cudaMalloc((void **) &d_input1, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector input1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device input vector input2
    float *d_input2 = NULL;
    err = cudaMalloc((void **) &d_input2, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector input2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Allocate the device output vector
    float *d_output = NULL;
    err = cudaMalloc((void **) &d_output, size);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector output (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the host input vectors to device input vectors
    err = cudaMemcpy(d_input1, input1, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input1 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_input2, input2, size, cudaMemcpyHostToDevice);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector input2 from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Launch the Processor Kernel 1
    dim3 dimBlock(32, 32, 1);
    dim3 dimGrid(4, 2, 2);
    process_kernel1<<<dimGrid, dimBlock>>>(d_input1, d_input2, d_output, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process kernel1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device output vector to host output vector
    err = cudaMemcpy(output, d_output, size, cudaMemcpyDeviceToHost);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector output from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
 
    err = cudaMemcpy(d_input1, output, size, cudaMemcpyHostToDevice);
 
    dim3 dimBlock2(8, 8, 16);
    dim3 dimGrid2(2, 8, 1);
    process_kernel2<<<dimGrid2, dimBlock2>>>(d_input1, d_output, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel1 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device output vector to the host output vector
    err = cudaMemcpy(output2, d_output, size, cudaMemcpyDeviceToHost);
 
    // copy the output vector of kernel 2 from host into device input vector
    err = cudaMemcpy(d_input1, output2, size, cudaMemcpyHostToDevice);
 
    dim3 dimBlock3(128, 8, 1);
    dim3 dimGrid3(16, 1, 1);
    process_kernel3<<<dimGrid3, dimBlock3>>>(d_input1, d_output, numElements);
    err = cudaGetLastError();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to launch process_kernel1 kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Copy the device output vector to host output vector
    err = cudaMemcpy(output3, d_output, size, cudaMemcpyDeviceToHost);
    
    // Verify that the result vector is correct
    for (int i = 0; i < numElements; ++i)
    {
        if (fabs(sqrt(log(sin(input1[i]) + cos(input2[i]))) - output3[i]) > 1e-5)
        {
            fprintf(stderr, "Result verification failed at element %d!\n", i);
            exit(EXIT_FAILURE);
        }
    }
 
    // print the output to the screen and also store it in the file output.txt
    FILE *fp;
    fp = fopen("output.txt", "w");
    for (int i = 0; i < numElements; ++i)
    {
        printf("%.2f ", output3[i]);
        fprintf(fp, "%.2f ", output3[i]);
    }
    printf("\n");
    fclose(fp);
 
    // Free device global memory
    err = cudaFree(d_input1);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector input1 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_input2);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector input2 (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaFree(d_output);

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to free device output vector (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    // Free host memory
    free(input1);
    free(input2);
    free(output);
    free(output2);
    free(output3);

    // Reset the device and exit
    err = cudaDeviceReset();

    if (err != cudaSuccess)
    {
        fprintf(stderr, "Failed to deinitialize the device! error=%s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    return 0;
}