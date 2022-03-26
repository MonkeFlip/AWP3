#include <iostream>
#include "./cuda_kernel.cuh"

void PrintMatrix(float* matrix, int matrix_N, int matrix_M, int M);
void TransformMatrixCPU(float* input_matrix, float* output_matrix);
void ClearMatrix(float* matrix, int n, int m, int k);

const int N = 1;
const int M = 4;
const int in_matrix_N = 3;
const int in_matrix_M = 5;
const int out_matrix_M = in_matrix_M + (in_matrix_M % 4 == 0 ? 0 : 4 - (in_matrix_M % 4));

int main()
{
    float* input_matrix = new float [in_matrix_N * in_matrix_M * M];
    float* output_matrix = new float [in_matrix_M * out_matrix_M * M];

    ///change this
    int counter = 0;
    for (int i = 0; i < in_matrix_N * in_matrix_M * M; i++)
    {
        input_matrix[i] = i;
    }
    ///change this

    ClearMatrix(output_matrix, in_matrix_N, out_matrix_M, M);
    TransformMatrixCPU(input_matrix, output_matrix);
    std::cout << "CPU implementation:" << std::endl;
    PrintMatrix(input_matrix, in_matrix_N, in_matrix_M, M);
    std::cout << "Result:" << std::endl;
    PrintMatrix(output_matrix, in_matrix_N, out_matrix_M, M);
    

    ClearMatrix(output_matrix, in_matrix_N, out_matrix_M, M);
    kernel(input_matrix, output_matrix, in_matrix_N, in_matrix_M, out_matrix_M, M);
    std::cout << "GPU implementation:" << std::endl;
    PrintMatrix(input_matrix, in_matrix_N, in_matrix_M, M);
    std::cout << "Result:" << std::endl;
    PrintMatrix(output_matrix, in_matrix_N, out_matrix_M, M);
}

void TransformMatrixCPU(float* input_matrix, float* output_matrix)
{
    for (int f = 0; f < in_matrix_N * in_matrix_M * M; f++)
    {
        int i = f / (in_matrix_M * M);
        int j = (f % (in_matrix_M * M)) / M;
        int k = (f % (in_matrix_M * M)) % M;
        //std::cout << "f: " << f << " i: " << i << " j: " << j << " k: " << k << std::endl;
        //std::cout << i * out_matrix_M * M + (k + (j / M) * M) * M + j % M<<"  "<< i * in_matrix_M * M + j * M + k << std::endl;
        output_matrix[i * out_matrix_M * M + (k + (j / M) * M) * M + j % M]
            = input_matrix[i * in_matrix_M * M + j * M + k];
    }
}

void ClearMatrix(float* matrix, int n, int m, int k)
{
    for (int i = 0; i < n * m * k; i++)
    {
        matrix[i] = 0;
    }
}

void PrintMatrix(float* matrix, int n, int m, int k)
{
    for (int i = 0; i < n * m * k; i++)
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