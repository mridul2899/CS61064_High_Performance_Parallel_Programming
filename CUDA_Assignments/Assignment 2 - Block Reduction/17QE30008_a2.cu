#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void reduce(float* v, float* v_r, float N, float K, int *size_f) 
{
    int blocknum = blockIdx.x + blockIdx.y * gridDim.x;
    int tid = blocknum * blockDim.x + threadIdx.x;
    
    int a = (int) N;
    int b = (int) K;
    int s;
    for (s = 1; s < a && a / s >= b; s *= b) 
    {
        if (tid % (s * b) == 0 && tid + s < a)
        {
            for (int i = 1; i < b; ++i)
            {
                v[tid * s] += v[tid * s + i];
            }
        }
        __syncthreads();
        if (tid % (s * b) == 0)
            v[tid * s] /= K;
    }
    __syncthreads();

    if (tid % s == 0)
    {
            v_r[tid / s] = v[tid];
    }
    *size_f = a / s;
}

int main(void)
{
    int t;
    scanf("%d", &t);
    while (t--)
    {
        int p, q;
        scanf("%d %d", &p, &q);
        float n, k;
        n = pow(2, p);
        k = pow(2, q);
        printf("%f  %f\n", n, k);
        float *h_A, *d_A = NULL;
        float *h_result;
        float *d_result = NULL;
        size_t size = (int) n * sizeof(float); 
        size_t size2 = (int) n * sizeof(float);
        h_A = (float *) malloc(size);
        h_result = (float *) malloc(size);
        int i;
        for (i = 0; i < n; ++i)
        {
            scanf("%f", &h_A[i]);
        }
        for (i = 0; i < n; ++i)
        {
            printf("%f ", h_A[i]);
        }
        int *size_f;
        int *h_s;
        h_s = (int *) malloc(sizeof(int));
        cudaMalloc((void **) &d_A, size);
        cudaMalloc((void **) &d_result, size);
        cudaMalloc((void **) &size_f, sizeof(int));
        cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
        dim3 dimBlock(k, 1, 1);
        dim3 dimGrid(sqrt(n / k), sqrt(n / k), 1);
        reduce<<<dimGrid, dimBlock>>>(d_A, d_result, n, k, size_f);
        cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
        cudaMemcpy(h_s, size_f, sizeof(int), cudaMemcpyDeviceToHost);
        printf("\n%d\n", *h_s);
        
        for (int i = 0; i < *h_s; i++) 
        {
            printf("%f ", h_result[i]);
        }
    }
    return 0;
}