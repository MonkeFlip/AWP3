#include "cuda_runtime.h"
#include <iostream>
#include "./cuda_kernel.cuh"


__global__ void MatrixTransformKernel(float* in_matrix, float* out_matrix, int N, int in_M, int out_M, int K)
{
    long f = blockDim.x * blockIdx.x + threadIdx.x;

    if (f < N * in_M * K)
    {
        long i = f / (in_M * K);
        long j = (f % (in_M * K)) / K;
        long k = (f % (in_M * K)) % K;
        out_matrix[i * out_M * K + (k + (j / K) * K) * K + j % K]
            = in_matrix[f];
    }
}

__global__ void SharedMemoryMatrixTransformKernel(float* in_matrix, float* out_matrix, int N, int in_M, int out_M, int K)
{
    long f = blockDim.x * blockIdx.x + threadIdx.x;

    if (f < N * in_M * K)
    {
        __shared__ float smem[1024];
        
        long i = f / (in_M * K);
        long j = (f % (in_M * K)) / K;
        long k = (f % (in_M * K)) % K;

        smem[threadIdx.x] = in_matrix[blockIdx.x * 1024 + threadIdx.x];
        __syncthreads();

        out_matrix[i * out_M * K + (k + (j / K) * K) * K + j % K]
            = smem[threadIdx.x];
    }
}

void kernel(float* in_matrix, float* out_matrix, int N, int in_M, int out_M, int K, float* elapsedTime)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float* d_In, * d_Out;

    cudaMalloc(&d_In, N * in_M * K * sizeof(float));
    cudaMalloc(&d_Out, N * out_M * K * sizeof(float));

    cudaMemcpy(d_In, in_matrix, N * in_M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Out, out_matrix, N * out_M * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1024);
    long s = N * out_M * K / threadsPerBlock.x + 1;
    s = s == 0 ? 1 : s;
    dim3 numBlocks(s);

    cudaEventRecord(start, 0);
    MatrixTransformKernel << <numBlocks, threadsPerBlock >> > (d_In, d_Out, N, in_M, out_M, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsedTime, start, stop);
    cudaMemcpy(out_matrix, d_Out, N * out_M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_In);
    cudaFree(d_Out);
}


void kernel_SharedMemory(float* in_matrix, float* out_matrix, int N, int in_M, int out_M, int K, float* elapsedTime)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float* d_In, * d_Out;

    cudaMalloc(&d_In, N * in_M * K * sizeof(float));
    cudaMalloc(&d_Out, N * out_M * K * sizeof(float));

    cudaMemcpy(d_In, in_matrix, N * in_M * K * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Out, out_matrix, N * out_M * K * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(1024);
    long s = N * out_M * K / threadsPerBlock.x + 1;
    s = s == 0 ? 1 : s;
    dim3 numBlocks(s);
    
    cudaEventRecord(start, 0);
    SharedMemoryMatrixTransformKernel << <numBlocks, threadsPerBlock >> > (d_In, d_Out, N, in_M, out_M, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsedTime, start, stop);
    cudaMemcpy(out_matrix, d_Out, N * out_M * K * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_In);
    cudaFree(d_Out);
}

void kernel_PinnedMemory(float* in_matrix, float* out_matrix, int N, int in_M, int out_M, int K, float* elapsedTime)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float* d_In, * d_Out;

    d_In = in_matrix;
    d_Out = out_matrix;

    dim3 threadsPerBlock(1024);
    long s = N * out_M * K / threadsPerBlock.x + 1;
    s = s == 0 ? 1 : s;
    dim3 numBlocks(s);
    cudaEventRecord(start, 0);
    MatrixTransformKernel << <numBlocks, threadsPerBlock >> > (d_In, d_Out, N, in_M, out_M, K);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(elapsedTime, start, stop);
}