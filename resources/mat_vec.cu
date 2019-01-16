#include <cuda.h>

extern "C" texture<float, 2> mat;

extern "C" __global__ void mat_vec(const float *vec, float *result) {
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    *result += tex2D(mat, y, x) * vec[x];
}