#include <iostream>
#include <ctime>
#include <ratio>
#include <chrono>
#include "cuda_runtime.h"
#include "./cuda_kernel.cuh"

void PrintMatrix(float* matrix, long matrix_N, long matrix_M, long M);
void TransformMatrixCPU(float* input_matrix, float* output_matrix);
void ClearMatrix(float* matrix, long n, long m, long k);
bool CompareMatrices(float* matrix1, float* matrix2, long size);
float* CPUimplementation();
float* GPUimplementation_globalMemory();
float* GPUimplementation_pinnedMemory();

//const int N = 1;
const int M = 4;
const int in_matrix_N = 7000;
const int in_matrix_M = 10000;
const int out_matrix_M = in_matrix_M + (in_matrix_M % 4 == 0 ? 0 : 4 - (in_matrix_M % 4));

int main()
{
    float* gpu_matrix = GPUimplementation_pinnedMemory();
    float* cpu_matrix = GPUimplementation_globalMemory();

    if (CompareMatrices(cpu_matrix, gpu_matrix, in_matrix_N * out_matrix_M * M))
    {
        std::cout << "Matrices are equal." << std::endl;
    }
    else
    {
        std::cout << "Matrices are not equal." << std::endl;
    }

    delete[] cpu_matrix;
    //delete[] gpu_matrix;
}

float* GPUimplementation_pinnedMemory()
{
    float* input_matrix;
    float* output_matrix;

    cudaMallocHost(&input_matrix, in_matrix_N * in_matrix_M * M * sizeof(float));
    cudaMallocHost(&output_matrix, in_matrix_M * out_matrix_M * M * sizeof(float));

    long counter = 0;
    for (long i = 0; i < in_matrix_N * in_matrix_M * M; i++)
    {
        input_matrix[i] = i;
    }
    ///change this

    ClearMatrix(output_matrix, in_matrix_N, out_matrix_M, M);
    float time = 0;
    float* GPU_elapsedTime = &time;
    kernel(input_matrix, output_matrix, in_matrix_N, in_matrix_M, out_matrix_M, M, GPU_elapsedTime);
    std::cout << "GPU implementation time: " << time / 1000 << " seconds." << std::endl;

    cudaFreeHost(input_matrix);

    return output_matrix;
}

float* GPUimplementation_globalMemory()
{
    float* input_matrix = new float[in_matrix_N * in_matrix_M * M];
    float* output_matrix = new float[in_matrix_M * out_matrix_M * M];

    long counter = 0;
    for (long i = 0; i < in_matrix_N * in_matrix_M * M; i++)
    {
        input_matrix[i] = i;
    }
    ///change this

    ClearMatrix(output_matrix, in_matrix_N, out_matrix_M, M);
    float time = 0;
    float* GPU_elapsedTime = &time;
    kernel(input_matrix, output_matrix, in_matrix_N, in_matrix_M, out_matrix_M, M, GPU_elapsedTime);
    std::cout << "GPU implementation time: " << time / 1000 << " seconds." << std::endl;

    delete[] input_matrix;
    return output_matrix;
}

float* CPUimplementation()
{
    using namespace std::chrono;
    float* input_matrix = new float [in_matrix_N * in_matrix_M * M];
    float* output_matrix = new float [in_matrix_M * out_matrix_M * M];

    ///change this
    long counter = 0;
    for (long i = 0; i < in_matrix_N * in_matrix_M * M; i++)
    {
        input_matrix[i] = i;
    }
    ///change this

    ClearMatrix(output_matrix, in_matrix_N, out_matrix_M, M);
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    TransformMatrixCPU(input_matrix, output_matrix);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    duration<double> time_span = duration_cast<duration<double>>(t2 - t1);
    std::cout << "CPU implementation time: " << time_span.count() << " seconds." << std::endl;

    //delete[] input_matrix;
    return output_matrix;
}

void TransformMatrixCPU(float* input_matrix, float* output_matrix)
{
    for (long f = 0; f < in_matrix_N * in_matrix_M * M; f++)
    {
        long i = f / (in_matrix_M * M);
        long j = (f % (in_matrix_M * M)) / M;
        long k = (f % (in_matrix_M * M)) % M;
        //std::cout << "f: " << f << " i: " << i << " j: " << j << " k: " << k << std::endl;
        //std::cout << i * out_matrix_M * M + (k + (j / M) * M) * M + j % M<<"  "<< i * in_matrix_M * M + j * M + k << std::endl;
        output_matrix[i * out_matrix_M * M + (k + (j / M) * M) * M + j % M]
            = input_matrix[i * in_matrix_M * M + j * M + k];
    }
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
            return false;
        }
    }

    return true;
}