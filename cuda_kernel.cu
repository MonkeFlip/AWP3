#include "cuda_runtime.h"
#include <iostream>
#include "./cuda_kernel.cuh"

__global__ void MatrixTransformKernel(float* in_matrix, float* out_matrix, int N, int in_M, int out_M, int K)
{
    // Get thread ID.
    long f = blockDim.x * blockIdx.x + threadIdx.x;

    if (f < N * in_M * K)
    {
        long i = f / (in_M * K);
        long j = (f % (in_M * K)) / K;
        long k = (f % (in_M * K)) % K;
        out_matrix[i * out_M * K + (k + (j / K) * K) * K + j % K]
            = in_matrix[i * in_M * K + j * K + k];
    }
}


void kernel(float* in_matrix, float* out_matrix, int N, int in_M, int out_M, int K, float* elapsedTime)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    // Initialize device pointers.
    float* d_In, * d_Out;

    // Allocate device memory.
    cudaMalloc(&d_In, N * in_M * K * sizeof(float));
    cudaMalloc(&d_Out, N * out_M * K * sizeof(float));

    // Transfer arrays a and b to device.
    cudaMemcpy(d_In, in_matrix, N * in_M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Out, out_matrix, N * out_M * K * sizeof(float), cudaMemcpyHostToDevice);

    // Calculate blocksize and gridsize.
    dim3 threadsPerBlock(1024);
    long s = N * out_M * K / threadsPerBlock.x + 1;
    s = s == 0 ? 1 : s;
    dim3 numBlocks(s);

    // Launch CUDA kernel.
    MatrixTransformKernel << <numBlocks, threadsPerBlock >> > (d_In, d_Out, N, in_M, out_M, K);

    // Copy result array c back to host memory.
    cudaMemcpy(out_matrix, d_Out, N * out_M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_In);
    cudaFree(d_Out);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsedTime, start, stop);
}