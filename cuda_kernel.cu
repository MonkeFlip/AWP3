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
        int new_i = i % 2 == 0 ? i + 1 : i - 1;
        int new_k = k;
        if (k % 4 == 0 && 0 < new_i * out_M * K + j * K + new_k && new_i * out_M * K + j * K + new_k < N * out_M * K && j + k != 0)
        {
            new_k = k - 1;            
        }
        else if ((k + 1) % 4 == 0 && 0 < new_i * out_M * K + j * K + new_k && new_i * out_M * K + j * K + new_k < N * out_M * K && (j + 1) * K + k != out_M * K + K - 1)
        {
            new_k = k + 1;            
        }
        
        if (i % 2 == 1)
        {
            out_matrix[f] = in_matrix[(i - 1) * in_M * K + j * K + new_k];
        }
        else
        {
            out_matrix[f] = in_matrix[(i + 1) * in_M * K + j * K + new_k];
        }
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
        int new_i = i % 2 == 0 ? i + 1 : i - 1;
        int new_k = k;

        if (k % 4 == 0 && 0 < new_i * out_M * K + j * K + new_k && new_i * out_M * K + j * K + new_k < N * out_M * K && j + k != 0)
        {
            new_k = k - 1;
        }
        else if ((k + 1) % 4 == 0 && 0 < new_i * out_M * K + j * K + new_k && new_i * out_M * K + j * K + new_k < N * out_M * K && (j + 1) * K + k != out_M * K + K - 1)
        {
            new_k = k + 1;
        }

        if (i % 2 == 1)
        {
            smem[threadIdx.x] = in_matrix[(i - 1) * in_M * K + j * K + new_k];
        }
        else
        {
            smem[threadIdx.x] = in_matrix[(i + 1) * in_M * K + j * K + new_k];
        }
        __syncthreads();


        out_matrix[f] = smem[threadIdx.x];
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

    cudaHostGetDevicePointer(&d_In, in_matrix, 0);
    cudaHostGetDevicePointer(&d_Out, out_matrix, 0);

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