#include <iostream>
#include <cuda_runtime.h>

__global__ void my_reduction(const float* input, float* output, int n) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();

    // Naive reduction with bank conflicts and divergence
    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(output, sdata[0]);
    }
}

int main() {
    int n = 1 << 24; // 16M elements
    size_t size = n * sizeof(float);

    float *h_in = (float*)malloc(size);
    for(int i=0; i<n; i++) h_in[i] = 1.0f;
    float h_out = 0.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, sizeof(float));

    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_out, &h_out, sizeof(float), cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    my_reduction<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Sum: " << h_out << std::endl;

    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);

    return 0;
}
