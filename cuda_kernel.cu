//#include "cuda_runtime.h"
//#include "./cuda_kernel.cuh"
//
//__global__ void MatrixTransformKernel(float*** in_matrix, float*** out_matrix, int N, int in_M, int out_M, int K)
//{
//    // Get thread ID.
//    int i = blockDim.x * blockIdx.x + threadIdx.x;
//    int j = blockDim.y * blockIdx.y + threadIdx.y;
//    int k = blockDim.z * blockIdx.z + threadIdx.z;
//
//    if (i < N && j < in_M && k < K)
//    {
//        out_matrix[i][k + (j / 4) * 4][j % 4] = in_matrix[i][j][k];
//    }
//}
//
//
//
///**
// * Wrapper function for the CUDA kernel function.
// * @param A Array A.
// * @param B Array B.
// * @param C Sum of array elements A and B directly across.
// * @param arraySize Size of arrays A, B, and C.
// */
//void kernel(float*** in_matrix, float*** out_matrix, int N, int in_M, int out_M, int K)
//{
//
//    // Initialize device pointers.
//    cudaPitchedPtr* d_In, * d_Out;
//    cudaExtent extent_in;
//    extent_in.height = N;
//    extent_in.width = in_M;
//    extent_in.depth = K;
//
//    cudaExtent extent_out;
//    extent_out.height = N;
//    extent_out.width = out_M;
//    extent_out.depth = K;
//    // Allocate device memory.
//    cudaMalloc3D(d_In, extent_in);
//    cudaMalloc3D(d_Out, extent_out);
//
//    // Transfer arrays a and b to device.
//    cudaMemcpy3D(d_In, in_matrix, arraySize * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_Out, out_matrix, arraySize * sizeof(float), cudaMemcpyHostToDevice);
//
//    // Calculate blocksize and gridsize.
//    dim3 blockSize(512, 1, 1);
//    dim3 gridSize(512 / arraySize + 1, 1);
//
//    // Launch CUDA kernel.
//    MatrixTransformKernel << <gridSize, blockSize >> > (d_A, d_B, d_C, arraySize);
//
//    // Copy result array c back to host memory.
//    cudaMemcpy(out_matrix, d_Out, arraySize * sizeof(float), cudaMemcpyDeviceToHost);
//}