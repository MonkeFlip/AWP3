#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include "cuda_runtime.h"
#include "./cuda_kernel.cuh"

void PrintMatrix(float* matrix, long matrix_N, long matrix_M, long M);
float* TransformMatrixCPU(float* input_matrix, float* output_matrix);
void ClearMatrix(float* matrix, long n, long m, long k);
bool CompareMatrices(float* matrix1, float* matrix2, long size);
float* CPUimplementation(float* input_matrix, float* output_matrix);
float* GPUimplementation_globalMemory(float* input_matrix, float* output_matrix);
float* GPUimplementation_sharedMemory(float* input_matrix, float* output_matrix);
float* GPUimplementation_pinnedMemory(float* input_matrix, float* output_matrix);

const int K = 4;
const int in_matrix_N = 7000;
const int in_matrix_M = 8000;
const int out_matrix_M = in_matrix_M + in_matrix_M % 2;

int main()
{
    float* input_matrix = new float[in_matrix_N * in_matrix_M * K];
    float* output_matrix1 = new float[in_matrix_M * out_matrix_M * K];
    float* output_matrix2 = new float[in_matrix_M * out_matrix_M * K];
    for (long i = 0; i < in_matrix_N * in_matrix_M * K; i++)
    {
        //input_matrix[i] = rand();
        input_matrix[i] = i;
    }
    /*PrintMatrix(input_matrix, in_matrix_N, in_matrix_M, M);
    std::cout<< std::endl << "Transformed matrix:" << std::endl<< std::endl;*/
    float* matrix1 = CPUimplementation(input_matrix, output_matrix1);
    //PrintMatrix(matrix1, in_matrix_N, out_matrix_M, K);
    float* matrix2 = GPUimplementation_globalMemory(input_matrix, output_matrix2);
    //PrintMatrix(matrix2, in_matrix_N, out_matrix_M, K);


    if (CompareMatrices(matrix1, matrix2, in_matrix_N * out_matrix_M * K))
    {
        std::cout << "Matrices are equal." << std::endl;
    }
    else
    {
        std::cout << "Matrices are not equal." << std::endl;
    }

    delete[] input_matrix;
    delete[] output_matrix1;
    delete[] output_matrix2;
}

float* GPUimplementation_pinnedMemory(float* input_matrix, float* output_matrix)
{
    cudaHostRegister(input_matrix, in_matrix_N * in_matrix_M * K * sizeof(float), cudaHostRegisterMapped);
    cudaHostRegister(output_matrix, in_matrix_M * out_matrix_M * K * sizeof(float), cudaHostRegisterMapped);

    ClearMatrix(output_matrix, in_matrix_N, out_matrix_M, K);
    float time = 0;
    float* GPU_elapsedTime = &time;
    kernel_PinnedMemory(input_matrix, output_matrix, in_matrix_N, in_matrix_M, out_matrix_M, K, GPU_elapsedTime);
    std::cout << "GPU with pinned memory implementation time: " << time / 1000 << " seconds." << std::endl;

    return output_matrix;
}

float* GPUimplementation_globalMemory(float* input_matrix, float* output_matrix)
{
    ClearMatrix(output_matrix, in_matrix_N, out_matrix_M, K);
    float time = 0;
    float* GPU_elapsedTime = &time;
    kernel(input_matrix, output_matrix, in_matrix_N, in_matrix_M, out_matrix_M, K, GPU_elapsedTime);
    std::cout << "GPU with global memory implementation time: " << time / 1000 << " seconds." << std::endl;

    return output_matrix;
}

float* GPUimplementation_sharedMemory(float* input_matrix, float* output_matrix)
{
    ClearMatrix(output_matrix, in_matrix_N, out_matrix_M, K);
    float time = 0;
    float* GPU_elapsedTime = &time;
    kernel_SharedMemory(input_matrix, output_matrix, in_matrix_N, in_matrix_M, out_matrix_M, K, GPU_elapsedTime);
    std::cout << "GPU with shared memory implementation time: " << time / 1000 << " seconds." << std::endl;

    return output_matrix;
}

float* CPUimplementation(float* input_matrix, float* output_matrix)
{
    using namespace std::chrono;

    ClearMatrix(output_matrix, in_matrix_N, out_matrix_M, K);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    output_matrix = TransformMatrixCPU(input_matrix, output_matrix);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "CPU implementation time: " << time_span.count() << " seconds." << std::endl;

    return output_matrix;
}

float* TransformMatrixCPU(float* input_matrix, float* output_matrix)
{
    for (long f = 0; f < in_matrix_N * in_matrix_M * K; f++)
    {
        long i = f / (in_matrix_M * K);
        long j = (f % (in_matrix_M * K)) / K;
        long k = (f % (in_matrix_M * K)) % K;

        int new_k = k;
        if (k % 4 == 0 && 0 < i * out_matrix_M * K + j * K + new_k && i * out_matrix_M * K + j * K + new_k < in_matrix_N * out_matrix_M * K && j + k != 0)
        {
            new_k = k - 1;

            output_matrix[i * out_matrix_M * K + j * K + new_k] = input_matrix[f];

        }
        else if ((k + 1) % 4 == 0 && 0 < i * out_matrix_M * K + j * K + new_k && i * out_matrix_M * K + j * K + new_k < in_matrix_N * out_matrix_M * K && (j + 1) * K + k != out_matrix_M * K + K - 1)
        {
            new_k = k + 1;

            output_matrix[i * out_matrix_M * K + j * K + new_k] = input_matrix[f];

        }
        else
        {
            output_matrix[i * out_matrix_M * K + j * K + k] = input_matrix[f];
        }
    }

    for (long f = 0; f < in_matrix_N * in_matrix_M * K; f++)
    {
        long i = f / (in_matrix_M * K);
        long j = (f % (in_matrix_M * K)) / K;
        long k = (f % (in_matrix_M * K)) % K;
        if (i % 2 == 1)
        {
            float temp = output_matrix[i * out_matrix_M * K + j * K + k];
            output_matrix[i * out_matrix_M * K + j * K + k] = output_matrix[(i - 1) * out_matrix_M * K + j * K + k];
            output_matrix[(i - 1) * out_matrix_M * K + j * K + k] = temp;
        }
    }

    return output_matrix;
}

void ClearMatrix(float* matrix, long n, long m, long k)
{
    for (long i = 0; i < n * m * k; i++)
    {
        matrix[i] = 0;
    }
}

void PrintMatrix(float* matrix, long n, long m, long k)
{
    for (long i = 0; i < n * m * k; i++)
    {

        std::cout << matrix[i] << " ";

        if (((i+1) >= k) && ((i + 1) % k == 0))
        {
            std::cout << "\t";
        }
        
        if (((i + 1) >= m * k) && ((i + 1) % (m * k) == 0))
        {
            std::cout << std::endl;
        }
    }
}

bool CompareMatrices(float* matrix1, float* matrix2, long size)
{
    for (long i = 0; i < size; i++)
    {
        if (matrix1[i] != matrix2[i])
        {
            std::cout << "matrix1[" << i << "] " << matrix1[i] << std::endl;
            std::cout << "matrix2[" << i << "] " << matrix2[i] << std::endl;
            return false;
        }
    }

    return true;
}