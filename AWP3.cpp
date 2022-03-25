#include <iostream>
#include "./cuda_kernel.cuh"

void Swap(float* array1, float* array2, int index);
void PrintMatrix(float*** matrix, int matrix_N, int matrix_M, int M);
void TransformMatrixCPU(float*** input_matrix, float*** output_matrix);
void ClearMatrix(float*** matrix, int n, int m, int k);

const int N = 1;
const int M = 4;
const int in_matrix_N = 1;
const int in_matrix_M = 5;
const int out_matrix_M = in_matrix_M + (in_matrix_M % 4 == 0 ? 0 : 4 - (in_matrix_M % 4));

int main()
{
    float*** input_matrix = new float**[in_matrix_M];
    float*** output_matrix = new float** [out_matrix_M];
    for (int i = 0; i < in_matrix_N; i++)
    {
        input_matrix[i] = new float*[in_matrix_M];
    }

    for (int i = 0; i < in_matrix_N; i++)
    {
        output_matrix[i] = new float* [out_matrix_M];
    }

    for (int i = 0; i < in_matrix_N; i++)
    {
        for (int j = 0; j < in_matrix_M; j++)
        {
            input_matrix[i][j] = new float[M];
        }
    }

    for (int i = 0; i < in_matrix_N; i++)
    {
        for (int j = 0; j < out_matrix_M; j++)
        {
            output_matrix[i][j] = new float[M];
        }
    }

    ///change this
    int counter = 0;
    for (int i = 0; i < in_matrix_N; i++)
    {
        for (int j = 0; j < in_matrix_M; j++)
        {
            for (int k = 0; k < M; k++)
            {
                input_matrix[i][j][k] = counter++;
            }
        }
    }
    ///change this

    ClearMatrix(output_matrix, in_matrix_N, out_matrix_M, M);

    TransformMatrixCPU(input_matrix, output_matrix);

    PrintMatrix(input_matrix, in_matrix_N, in_matrix_M, M);
    PrintMatrix(output_matrix, in_matrix_N, out_matrix_M, M);
    std::cout << "CPU implementation." << std::endl;
}

void TransformMatrixCPU(float*** input_matrix, float*** output_matrix)
{
    for (int i = 0; i < in_matrix_N; i++)
    {
        for (int j = 0; j < in_matrix_M; j++)
        {
            for (int k = 0; k < M; k++)
            {
                output_matrix[i][k + (j / 4) * 4][j % 4] = input_matrix[i][j][k];
            }
        }
    }
}

void ClearMatrix(float*** matrix, int n, int m, int k)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < m; j++)
        {
            for (int r = 0; r < k; r++)
            {
                matrix[i][j][r] = 0;
            }
        }
    }
}

void Swap(float* array1, float* array2, int index)
{
    int temp = array1[index];
    array1[index] = array2[index];
    array2[index] = temp;
}

void PrintMatrix(float*** matrix, int matrix_N, int matrix_M, int M)
{
    for (int i = 0; i < matrix_N; i++)
    {
        for (int j = 0; j < matrix_M; j++)
        {
            for (int k = 0; k < M; k++)
            {
                std::cout << matrix[i][j][k] << " ";
            }

            std::cout << "\t";
        }
        std::cout << std::endl;
    }
}