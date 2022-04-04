#include <curand_kernel.h>

void kernel(float* in_matrix, float* out_matrix, int N, int in_M, int out_M, int K, float* elapsedTime);
void kernel_PinnedMemory(float* in_matrix, float* out_matrix, int N, int in_M, int out_M, int K, float* elapsedTime);
void kernel_SharedMemory(float* in_matrix, float* out_matrix, int N, int in_M, int out_M, int K, float* elapsedTime);

