#ifdef WITH_CUDA

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <vector>
#include <algorithm>

#include "ops/Matmul.cuh"
#include "core/Tensor.h"
#include "core/TensorDispatch.h"

namespace OwnTensor {

    template <typename T>
    __global__ void batched_matmul_kernel(const T* A, const T* B, T* output,
                                        const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                        const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                        size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                        size_t total_batches)
    {
        size_t batch_idx = blockIdx.z;
        size_t i = blockIdx.y * blockDim.y + threadIdx.y;
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx >= total_batches) return;

        // Matrix dimensions
        size_t m = a_shape[a_ndim - 2];
        size_t n = a_shape[a_ndim - 1];
        size_t p = b_shape[b_ndim - 1];

        if (i >= m || j >= p) return;

        // Calculate batch offsets
        // Calculate batch offsets
        size_t a_batch_offset = 0;
        size_t b_batch_offset = 0;
        size_t out_batch_offset = 0;

        // size_t temp_batch = batch_idx;
        // for (int dim = out_ndim - 3; dim >= 0; --dim) {
            //     size_t batch_dim_size = out_shape[dim];
            //     size_t batch_coord = temp_batch % batch_dim_size;
            //     temp_batch /= batch_dim_size;
            
            //     // Calculate offsets using the actual batch coordinates
            //     if (dim < a_ndim - 2) {
                //         a_batch_offset += batch_coord * a_strides[dim];
                //     }
                //     if (dim < b_ndim - 2) {
                    //         b_batch_offset += batch_coord * b_strides[dim];
                    //     }
                    //     out_batch_offset += batch_coord * out_strides[dim];
                    // }
                    
        // FIXED: Proper batch offset calculation
        size_t temp_batch = batch_idx;
        for (int dim = out_ndim - 3; dim >= 0; --dim)
        {
            size_t batch_dim_size = out_shape[dim];
            size_t batch_coord = temp_batch % batch_dim_size;
            temp_batch /= batch_dim_size;

            out_batch_offset += batch_coord * out_strides[dim];

            // RIGHT-ALIGNED: Calculating corresponding dimensions for A and B
            size_t a_corres_dim = dim - (out_ndim - 2 - (a_ndim - 2));
            size_t b_corres_dim = dim - (out_ndim - 2 - (b_ndim - 2));

            // For A and B: Right aligned broadcasting rules
            
            if (dim >= out_ndim - 2 - (a_ndim - 2))
            {
                size_t a_dim_size = a_shape[a_corres_dim];
                size_t a_idx = (a_dim_size > 1) ? batch_coord : 0;
                a_batch_offset += a_idx * a_strides[a_corres_dim];
            }

            if (dim >= out_ndim - 2 - (b_ndim - 2))
            {
                size_t b_dim_size = b_shape[b_corres_dim];
                size_t b_idx = (b_dim_size > 1) ? batch_coord : 0;
                b_batch_offset += b_idx * b_strides[b_corres_dim];
            } 
        }
                    
                    
        T sum{};
        for (size_t k = 0; k < n; ++k) {
            size_t a_idx = a_batch_offset + i * a_strides[a_ndim - 2] + k * a_strides[a_ndim - 1];
            size_t b_idx = b_batch_offset + k * b_strides[b_ndim - 2] + j * b_strides[b_ndim - 1];
            sum += A[a_idx] * B[b_idx];
        }
        
        size_t out_idx = out_batch_offset + i * out_strides[out_ndim - 2] + j * out_strides[out_ndim - 1];
        output[out_idx] = sum;
    }

    // Specializations for bfloat16 and half (similar structure)
    __global__ void batched_matmul_kernel(const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* output,
                                    const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                    const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                    size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                    size_t total_batches)
    {
        size_t batch_idx = blockIdx.z;
        size_t i = blockIdx.y * blockDim.y + threadIdx.y;
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx >= total_batches) return;

        size_t m = a_shape[a_ndim - 2];
        size_t n = a_shape[a_ndim - 1];
        size_t p = b_shape[b_ndim - 1];

        if (i >= m || j >= p) return;

        // Calculate batch offsets
        size_t a_batch_offset = 0;
        size_t b_batch_offset = 0;
        size_t out_batch_offset = 0;

        // FIXED: Proper batch offset calculation
        size_t temp_batch = batch_idx;
        for (int dim = out_ndim - 3; dim >= 0; --dim) {
            size_t batch_dim_size = out_shape[dim];
            size_t batch_coord = temp_batch % batch_dim_size;
            temp_batch /= batch_dim_size;
            
            // Calculate offsets using the actual batch coordinates
            if (dim < a_ndim - 2) {
                a_batch_offset += batch_coord * a_strides[dim];
            }
            if (dim < b_ndim - 2) {
                b_batch_offset += batch_coord * b_strides[dim];
            }
            out_batch_offset += batch_coord * out_strides[dim];
        }

        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k) {
            size_t a_idx = a_batch_offset + i * a_strides[a_ndim - 2] + k * a_strides[a_ndim - 1];
            size_t b_idx = b_batch_offset + k * b_strides[b_ndim - 2] + j * b_strides[b_ndim - 1];
            sum += __bfloat162float(A[a_idx]) * __bfloat162float(B[b_idx]);
        }
        
        size_t out_idx = out_batch_offset + i * out_strides[out_ndim - 2] + j * out_strides[out_ndim - 1];
        output[out_idx] = __float2bfloat16(sum);
    }

    __global__ void batched_matmul_kernel(const __half* A, const __half* B, __half* output,
                                        const size_t* a_shape, const size_t* b_shape, const size_t* out_shape,
                                        const size_t* a_strides, const size_t* b_strides, const size_t* out_strides,
                                        size_t a_ndim, size_t b_ndim, size_t out_ndim,
                                        size_t total_batches)
    {
        size_t batch_idx = blockIdx.z;
        size_t i = blockIdx.y * blockDim.y + threadIdx.y;
        size_t j = blockIdx.x * blockDim.x + threadIdx.x;

        if (batch_idx >= total_batches) return;

        size_t m = a_shape[a_ndim - 2];
        size_t n = a_shape[a_ndim - 1];
        size_t p = b_shape[b_ndim - 1];

        if (i >= m || j >= p) return;

        // Calculate batch offsets
    size_t a_batch_offset = 0;
    size_t b_batch_offset = 0;
    size_t out_batch_offset = 0;

    // FIXED: Proper batch offset calculation
    size_t temp_batch = batch_idx;
    for (int dim = out_ndim - 3; dim >= 0; --dim) {
        size_t batch_dim_size = out_shape[dim];
        size_t batch_coord = temp_batch % batch_dim_size;
        temp_batch /= batch_dim_size;
        
        // Calculate offsets using the actual batch coordinates
        if (dim < a_ndim - 2) {
            a_batch_offset += batch_coord * a_strides[dim];
        }
        if (dim < b_ndim - 2) {
            b_batch_offset += batch_coord * b_strides[dim];
        }
        out_batch_offset += batch_coord * out_strides[dim];
    }

        float sum = 0.0f;
        for (size_t k = 0; k < n; ++k) {
            size_t a_idx = a_batch_offset + i * a_strides[a_ndim - 2] + k * a_strides[a_ndim - 1];
            size_t b_idx = b_batch_offset + k * b_strides[b_ndim - 2] + j * b_strides[b_ndim - 1];
            sum += __half2float(A[a_idx]) * __half2float(B[b_idx]);
        }
        
        size_t out_idx = out_batch_offset + i * out_strides[out_ndim - 2] + j * out_strides[out_ndim - 1];
        output[out_idx] = __float2half(sum);
    }

    void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream) //✨✨✨
    {
        const auto& a_shape = A.shape().dims;
        const auto& b_shape = B.shape().dims;
        const auto& out_shape = output.shape().dims;
        
        size_t a_ndim = a_shape.size();
        size_t b_ndim = b_shape.size();
        size_t out_ndim = out_shape.size();
        
        // Calculate total batches
        size_t total_batches = 1;
        for (int i = 0; i < out_ndim - 2; ++i) {
            total_batches *= out_shape[i];
        }

        // Matrix dimensions
        size_t m = a_shape[a_ndim - 2];
        size_t p = b_shape[b_ndim - 1];

        // 3D grid for batches
        dim3 block(16, 16);
        dim3 grid((p + block.x - 1) / block.x, 
                  (m + block.y - 1) / block.y, 
                  total_batches);

        // Device memory allocation for shapes and strides
        size_t *d_a_shape, *d_b_shape, *d_out_shape;
        size_t *d_a_strides, *d_b_strides, *d_out_strides;

        cudaMalloc(&d_a_shape, a_ndim * sizeof(size_t));
        cudaMalloc(&d_b_shape, b_ndim * sizeof(size_t));
        cudaMalloc(&d_out_shape, out_ndim * sizeof(size_t));
        cudaMalloc(&d_a_strides, a_ndim * sizeof(size_t));
        cudaMalloc(&d_b_strides, b_ndim * sizeof(size_t));
        cudaMalloc(&d_out_strides, out_ndim * sizeof(size_t));

        // Copy data to device
        cudaMemcpyAsync(d_a_shape, a_shape.data(), a_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨
        cudaMemcpyAsync(d_b_shape, b_shape.data(), b_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨
        cudaMemcpyAsync(d_out_shape, out_shape.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨
        cudaMemcpyAsync(d_a_strides, A.stride().strides.data(), a_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨
        cudaMemcpyAsync(d_b_strides, B.stride().strides.data(), b_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨
        cudaMemcpyAsync(d_out_strides, output.stride().strides.data(), out_ndim * sizeof(size_t), cudaMemcpyHostToDevice, stream); //✨✨✨

        dispatch_by_dtype(A.dtype(), [&](auto dummy){
            using T = decltype(dummy);
            
            // FP4 VALIDATION: No Matmul allowed
            if constexpr (std::is_same_v<T, float4_e2m1_t> || std::is_same_v<T, float4_e2m1_2x_t>) {
                throw std::runtime_error("Matrix Multiplication is not supported for FP4 types.");
            } else {
                const T* a_ptr = A.data<T>();
                const T* b_ptr = B.data<T>();
                T* out_ptr = output.data<T>();
    
                batched_matmul_kernel<<<grid, block, 0, stream>>>( 
                    a_ptr, b_ptr, out_ptr,
                    d_a_shape, d_b_shape, d_out_shape,
                    d_a_strides, d_b_strides, d_out_strides,
                    a_ndim, b_ndim, out_ndim, total_batches
                );
            }
            
        });

        // Free device memory
        cudaFree(d_a_shape);
        cudaFree(d_b_shape);
        cudaFree(d_out_shape);
        cudaFree(d_a_strides);
        cudaFree(d_b_strides);
        cudaFree(d_out_strides);
    }
}
#endif









// // ============================================================================
// // CUBLAS MATRIX MULTIPLICATION - NVIDIA cuBLAS Library Wrapper
// // ============================================================================
// // Provides optimized matrix multiplication using NVIDIA's cuBLAS library
// // Performance: Up to 150 TFLOPs on A100 (reference implementation)
// // Supports: FP16, BF16, FP32, FP64, TF32
// // ============================================================================

// #ifdef WITH_CUDA

// #include <cublas_v2.h>
// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
// #include <stdexcept>
// #include <memory>
// #include "core/TensorDispatch.h"

// #include "ops/Matmul.cuh"
// #include "core/Tensor.h"

// namespace OwnTensor {

// // ============================================================================
// // CUBLAS HANDLE MANAGEMENT (Thread-Safe Singleton)
// // ============================================================================

// class CublasHandleManager {
// private:
//     cublasHandle_t handle_;
//     bool initialized_ = false;

//     CublasHandleManager() {
//         cublasStatus_t status = cublasCreate(&handle_);
//         if (status != CUBLAS_STATUS_SUCCESS) {
//             throw std::runtime_error("Failed to create cuBLAS handle");
//         }
//         initialized_ = true;
//     }

//     ~CublasHandleManager() {
//         if (initialized_) {
//             cublasDestroy(handle_);
//         }
//     }

// public:
//     // Singleton instance
//     static CublasHandleManager& getInstance() {
//         static CublasHandleManager instance;
//         return instance;
//     }

//     cublasHandle_t get() { return handle_; }

//     // Delete copy/move constructors
//     CublasHandleManager(const CublasHandleManager&) = delete;
//     CublasHandleManager& operator=(const CublasHandleManager&) = delete;
// };

// // ============================================================================
// // HELPER FUNCTIONS
// // ============================================================================

// // Compute batch offsets with broadcasting support
// __device__ void compute_batch_offset_cublas(
//     int batch_idx,
//     const int* shape, const int* strides, int ndim,
//     const int* out_shape, int out_ndim,
//     int& offset)
// {
//     offset = 0;
//     if (out_ndim <= 2) return;
    
//     int temp_batch = batch_idx;
//     for (int dim = out_ndim - 3; dim >= 0; --dim) {
//         int batch_dim_size = out_shape[dim];
//         int batch_coord = temp_batch % batch_dim_size;
//         temp_batch /= batch_dim_size;
        
//         // Right-aligned broadcasting
//         int corres_dim = dim - (out_ndim - ndim);
//         if (corres_dim >= 0 && corres_dim < ndim - 2) {
//             int dim_size = shape[corres_dim];
//             int idx = (dim_size > 1) ? batch_coord : 0;
//             offset += idx * strides[corres_dim];
//         }
//     }
// }

// // ============================================================================
// // CUBLAS MATMUL IMPLEMENTATIONS BY DATA TYPE
// // ============================================================================

// // FP16 (Half-precision) using Tensor Cores
// void cublas_matmul_fp16(
//     const __half* A,
//     const __half* B,
//     __half* C,
//     int M, int N, int K,
//     int batch_count,
//     cudaStream_t stream)
// {
//     cublasHandle_t handle = CublasHandleManager::getInstance().get();
//     cublasSetStream(handle, stream);
    
//     // Enable Tensor Core acceleration
//     cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
//     const __half alpha = __float2half(1.0f);
//     const __half beta = __float2half(0.0f);
    
//     if (batch_count == 1) {
//         // Single matrix multiplication
//         // cuBLAS uses column-major, so we need to transpose:
//         // C = A @ B => C^T = B^T @ A^T
//         cublasStatus_t status = cublasHgemm(
//             handle,
//             CUBLAS_OP_N,    // B^T operation (no transpose, already col-major)
//             CUBLAS_OP_N,    // A^T operation (no transpose, already col-major)
//             N,              // rows of B^T (cols of B)
//             M,              // cols of A^T (rows of A)
//             K,              // cols of B^T / rows of A^T
//             &alpha,
//             B, N,           // B^T, leading dimension
//             A, K,           // A^T, leading dimension
//             &beta,
//             C, N            // C^T, leading dimension
//         );
        
//         if (status != CUBLAS_STATUS_SUCCESS) {
//             throw std::runtime_error("cuBLAS HGemm failed: " + std::to_string(status));
//         }
//     } else {
//         // Batched matrix multiplication
//         long long int strideA = M * K;
//         long long int strideB = K * N;
//         long long int strideC = M * N;
        
//         cublasStatus_t status = cublasHgemmStridedBatched(
//             handle,
//             CUBLAS_OP_N,
//             CUBLAS_OP_N,
//             N, M, K,
//             &alpha,
//             B, N, strideB,
//             A, K, strideA,
//             &beta,
//             C, N, strideC,
//             batch_count
//         );
        
//         if (status != CUBLAS_STATUS_SUCCESS) {
//             throw std::runtime_error("cuBLAS HGemmStridedBatched failed: " + std::to_string(status));
//         }
//     }
// }

// // BF16 (BFloat16) using Tensor Cores
// void cublas_matmul_bf16(
//     const __nv_bfloat16* A,
//     const __nv_bfloat16* B,
//     __nv_bfloat16* C,
//     int M, int N, int K,
//     int batch_count,
//     cudaStream_t stream)
// {
//     cublasHandle_t handle = CublasHandleManager::getInstance().get();
//     cublasSetStream(handle, stream);
    
//     // Enable Tensor Core acceleration
//     cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH);
    
//     const float alpha = 1.0f;
//     const float beta = 0.0f;
    
//     if (batch_count == 1) {
//         cublasStatus_t status = cublasGemmEx(
//             handle,
//             CUBLAS_OP_N,
//             CUBLAS_OP_N,
//             N, M, K,
//             &alpha,
//             B, CUDA_R_16BF, N,
//             A, CUDA_R_16BF, K,
//             &beta,
//             C, CUDA_R_16BF, N,
//             CUBLAS_COMPUTE_32F,
//             CUBLAS_GEMM_DEFAULT_TENSOR_OP
//         );
        
//         if (status != CUBLAS_STATUS_SUCCESS) {
//             throw std::runtime_error("cuBLAS BF16 Gemm failed: " + std::to_string(status));
//         }
//     } else {
//         long long int strideA = M * K;
//         long long int strideB = K * N;
//         long long int strideC = M * N;
        
//         cublasStatus_t status = cublasGemmStridedBatchedEx(
//             handle,
//             CUBLAS_OP_N,
//             CUBLAS_OP_N,
//             N, M, K,
//             &alpha,
//             B, CUDA_R_16BF, N, strideB,
//             A, CUDA_R_16BF, K, strideA,
//             &beta,
//             C, CUDA_R_16BF, N, strideC,
//             batch_count,
//             CUBLAS_COMPUTE_32F,
//             CUBLAS_GEMM_DEFAULT_TENSOR_OP
//         );
        
//         if (status != CUBLAS_STATUS_SUCCESS) {
//             throw std::runtime_error("cuBLAS BF16 GemmStridedBatched failed: " + std::to_string(status));
//         }
//     }
// }

// // FP32 (Single-precision) with TF32 Tensor Cores on Ampere+
// void cublas_matmul_fp32(
//     const float* A,
//     const float* B,
//     float* C,
//     int M, int N, int K,
//     int batch_count,
//     cudaStream_t stream)
// {
//     cublasHandle_t handle = CublasHandleManager::getInstance().get();
//     cublasSetStream(handle, stream);
    
//     // Enable TF32 Tensor Cores for FP32 (Ampere and newer)
//     cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH);
    
//     const float alpha = 1.0f;
//     const float beta = 0.0f;
    
//     if (batch_count == 1) {
//         cublasStatus_t status = cublasSgemm(
//             handle,
//             CUBLAS_OP_N,
//             CUBLAS_OP_N,
//             N, M, K,
//             &alpha,
//             B, N,
//             A, K,
//             &beta,
//             C, N
//         );
        
//         if (status != CUBLAS_STATUS_SUCCESS) {
//             throw std::runtime_error("cuBLAS SGemm failed: " + std::to_string(status));
//         }
//     } else {
//         long long int strideA = M * K;
//         long long int strideB = K * N;
//         long long int strideC = M * N;
        
//         cublasStatus_t status = cublasSgemmStridedBatched(
//             handle,
//             CUBLAS_OP_N,
//             CUBLAS_OP_N,
//             N, M, K,
//             &alpha,
//             B, N, strideB,
//             A, K, strideA,
//             &beta,
//             C, N, strideC,
//             batch_count
//         );
        
//         if (status != CUBLAS_STATUS_SUCCESS) {
//             throw std::runtime_error("cuBLAS SGemmStridedBatched failed: " + std::to_string(status));
//         }
//     }
// }

// // FP64 (Double-precision)
// void cublas_matmul_fp64(
//     const double* A,
//     const double* B,
//     double* C,
//     int M, int N, int K,
//     int batch_count,
//     cudaStream_t stream)
// {
//     cublasHandle_t handle = CublasHandleManager::getInstance().get();
//     cublasSetStream(handle, stream);
    
//     const double alpha = 1.0;
//     const double beta = 0.0;
    
//     if (batch_count == 1) {
//         cublasStatus_t status = cublasDgemm(
//             handle,
//             CUBLAS_OP_N,
//             CUBLAS_OP_N,
//             N, M, K,
//             &alpha,
//             B, N,
//             A, K,
//             &beta,
//             C, N
//         );
        
//         if (status != CUBLAS_STATUS_SUCCESS) {
//             throw std::runtime_error("cuBLAS DGemm failed: " + std::to_string(status));
//         }
//     } else {
//         long long int strideA = M * K;
//         long long int strideB = K * N;
//         long long int strideC = M * N;
        
//         cublasStatus_t status = cublasDgemmStridedBatched(
//             handle,
//             CUBLAS_OP_N,
//             CUBLAS_OP_N,
//             N, M, K,
//             &alpha,
//             B, N, strideB,
//             A, K, strideA,
//             &beta,
//             C, N, strideC,
//             batch_count
//         );
        
//         if (status != CUBLAS_STATUS_SUCCESS) {
//             throw std::runtime_error("cuBLAS DGemmStridedBatched failed: " + std::to_string(status));
//         }
//     }
// }

// // ============================================================================
// // TEMPLATE WRAPPER IMPLEMENTATION
// // ============================================================================

// template<typename T>
// void cublas_matmul_impl(
//     const Tensor& A,
//     const Tensor& B,
//     Tensor& output,
//     cudaStream_t stream)
// {
//     const auto& a_shape = A.shape().dims;
//     const auto& b_shape = B.shape().dims;
//     const auto& out_shape = output.shape().dims;
    
//     size_t a_ndim = a_shape.size();
//     size_t b_ndim = b_shape.size();
//     size_t out_ndim = out_shape.size();
    
//     // Matrix dimensions
//     int m = static_cast<int>(a_shape[a_ndim - 2]);
//     int k = static_cast<int>(a_shape[a_ndim - 1]);
//     int n = static_cast<int>(b_shape[b_ndim - 1]);
    
//     // Calculate total batches
//     int total_batches = 1;
//     for (size_t i = 0; i < out_ndim - 2; ++i) {
//         total_batches *= static_cast<int>(out_shape[i]);
//     }
    
//     const T* a_ptr = A.data<T>();
//     const T* b_ptr = B.data<T>();
//     T* out_ptr = output.data<T>();
    
//     // Dispatch to appropriate cuBLAS function
//     if constexpr (std::is_same<T, __half>::value) {
//         cublas_matmul_fp16(
//             reinterpret_cast<const __half*>(a_ptr),
//             reinterpret_cast<const __half*>(b_ptr),
//             reinterpret_cast<__half*>(out_ptr),
//             m, n, k,
//             total_batches,
//             stream
//         );
//     } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
//         cublas_matmul_bf16(
//             reinterpret_cast<const __nv_bfloat16*>(a_ptr),
//             reinterpret_cast<const __nv_bfloat16*>(b_ptr),
//             reinterpret_cast<__nv_bfloat16*>(out_ptr),
//             m, n, k,
//             total_batches,
//             stream
//         );
//     } else if constexpr (std::is_same<T, float>::value) {
//         cublas_matmul_fp32(a_ptr, b_ptr, out_ptr, m, n, k, total_batches, stream);
//     } else if constexpr (std::is_same<T, double>::value) {
//         cublas_matmul_fp64(a_ptr, b_ptr, out_ptr, m, n, k, total_batches, stream);
//     } else {
//         throw std::runtime_error("cuBLAS matmul: Unsupported data type");
//     }
// }

// // ============================================================================
// // PUBLIC API
// // ============================================================================

// void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream)
// {
//     dispatch_by_dtype(A.dtype(), [&](auto dummy) {
//         using T = decltype(dummy);
        
//         if constexpr (std::is_same<T, __half>::value ||
//                       std::is_same<T, __nv_bfloat16>::value ||
//                       std::is_same<T, float>::value ||
//                       std::is_same<T, double>::value) {
//             cublas_matmul_impl<T>(A, B, output, stream);
//         } else {
//             throw std::runtime_error("cuda_matmul: Unsupported dtype. Supported: FP16, BF16, FP32, FP64");
//         }
//     });
// }

// } // namespace OwnTensor

// #endif // WITH_CUDA












// #ifdef WITH_CUDA

// #include <cuda_runtime.h>
// #include <cuda_fp16.h>
// #include <cuda_bf16.h>
// #include <mma.h>
// #include <algorithm>
// #include <core/TensorDispatch.h>

// #include "ops/Matmul.cuh"
// #include "core/Tensor.h"

// namespace OwnTensor {

// using namespace nvcuda;

// // ============================================================================
// // KERNEL CONFIGURATION - Optimized 
// // ============================================================================

// // WMMA tile dimensions (hardware fixed)
// constexpr int WMMA_M = 16;
// constexpr int WMMA_N = 16;
// constexpr int WMMA_K = 16;

// // Block tile dimensions (tuned for Tensor Cores)
// constexpr int BM = 64;  // Block tile M dimension
// constexpr int BN = 64;  // Block tile N dimension
// constexpr int BK = 32;   // Block tile K dimension

// // Warp configuration
// constexpr int WARP_SIZE = 32;
// constexpr int WARPS_M = 2;  // 4 warps in M direction
// constexpr int WARPS_N = 2;  // 4 warps in N direction
// constexpr int NUM_WARPS = WARPS_M * WARPS_N;  // 16 warps per block
// constexpr int THREADS_PER_BLOCK = NUM_WARPS * WARP_SIZE;  // 512 threads

// // Warp tile dimensions
// constexpr int WM = BM / WARPS_M;  // 32
// constexpr int WN = BN / WARPS_N;  // 32

// // Shared memory padding to avoid bank conflicts
// constexpr int PAD = 8;

// // ============================================================================
// // HELPER DEVICE FUNCTIONS
// // ============================================================================

// // Compute batch offsets with broadcasting support
// __device__ void compute_batch_offset(
//     int batch_idx,
//     const int* shape, const int* strides, int ndim,
//     const int* out_shape, int out_ndim,
//     int& offset)
// {
//     offset = 0;
//     if (out_ndim <= 2) return;
    
//     int temp_batch = batch_idx;
//     for (int dim = out_ndim - 3; dim >= 0; --dim) {
//         int batch_dim_size = out_shape[dim];
//         int batch_coord = temp_batch % batch_dim_size;
//         temp_batch /= batch_dim_size;
        
//         // Right-aligned broadcasting
//         int corres_dim = dim - (out_ndim - ndim);
//         if (corres_dim >= 0 && corres_dim < ndim - 2) {
//             int dim_size = shape[corres_dim];
//             int idx = (dim_size > 1) ? batch_coord : 0;
//             offset += idx * strides[corres_dim];
//         }
//     }
// }

// // ============================================================================
// // FP16 TENSOR CORE KERNEL - CUDA-L2 Optimized
// // ============================================================================

// template<int BM, int BN, int BK, int WM, int WN>
// __global__ void matmul_fp16_optimized(
//     const __half* __restrict__ A,
//     const __half* __restrict__ B,
//     __half* __restrict__ C,
//     int M, int N, int K,
//     int total_batches,
//     const int* a_shape, const int* b_shape, const int* out_shape,
//     const int* a_strides, const int* b_strides, const int* out_strides,
//     int a_ndim, int b_ndim, int out_ndim)
// {
//     // Block and warp indices
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;
//     const int batch_idx = blockIdx.z;
    
//     if (batch_idx >= total_batches) return;
    
//     const int tid = threadIdx.x;
//     const int warp_id = tid / WARP_SIZE;
//     const int lane_id = tid % WARP_SIZE;
//     const int warp_row = warp_id / WARPS_N;
//     const int warp_col = warp_id % WARPS_N;
    
//     // Calculate batch offsets
//     int a_batch_offset = 0, b_batch_offset = 0, out_batch_offset = 0;
//     compute_batch_offset(batch_idx, a_shape, a_strides, a_ndim, out_shape, out_ndim, a_batch_offset);
//     compute_batch_offset(batch_idx, b_shape, b_strides, b_ndim, out_shape, out_ndim, b_batch_offset);
//     compute_batch_offset(batch_idx, out_shape, out_strides, out_ndim, out_shape, out_ndim, out_batch_offset);
    
//     // Adjust pointers for batch
//     A += a_batch_offset;
//     B += b_batch_offset;
//     C += out_batch_offset;
    
//     // Double-buffered shared memory with padding
//     __shared__ __half As[2][BM][BK + PAD];
//     __shared__ __half Bs[2][BK][BN + PAD];
    
//     // WMMA fragments
//     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
//     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_frag[2][2];
    
//     // Initialize accumulators
//     #pragma unroll
//     for (int i = 0; i < 2; i++) {
//         #pragma unroll
//         for (int j = 0; j < 2; j++) {
//             wmma::fill_fragment(acc_frag[i][j], __float2half(0.0f));
//         }
//     }
    
//     // Calculate global positions
//     const int block_row = by * BM;
//     const int block_col = bx * BN;
    
//     // Tile loading parameters
//     const int a_tile_stride = BM * BK;
//     const int b_tile_stride = BK * BN;
//     const int num_tiles = (K + BK - 1) / BK;
    
//     // Double buffering indices
//     int write_idx = 0;
//     int read_idx = 1;
    
//     // ========================================================================
//     // PREFETCH FIRST TILE
//     // ========================================================================
//     {
//         // Load A tile cooperatively
//         for (int i = tid; i < a_tile_stride; i += THREADS_PER_BLOCK) {
//             int row = i / BK;
//             int col = i % BK;
//             int global_row = block_row + row;
//             int global_col = col;
            
//             if (global_row < M && global_col < K) {
//                 As[write_idx][row][col] = A[global_row * K + global_col];
//             } else {
//                 As[write_idx][row][col] = __float2half(0.0f);
//             }
//         }
        
//         // Load B tile cooperatively
//         for (int i = tid; i < b_tile_stride; i += THREADS_PER_BLOCK) {
//             int row = i / BN;
//             int col = i % BN;
//             int global_row = row;
//             int global_col = block_col + col;
            
//             if (global_row < K && global_col < N) {
//                 Bs[write_idx][row][col] = B[global_row * N + global_col];
//             } else {
//                 Bs[write_idx][row][col] = __float2half(0.0f);
//             }
//         }
//     }
    
//     __syncthreads();
    
//     // ========================================================================
//     // MAIN COMPUTATION LOOP WITH DOUBLE BUFFERING
//     // ========================================================================
//     for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
//         // Swap buffers
//         read_idx = write_idx;
//         write_idx = 1 - write_idx;
        
//         // ====================================================================
//         // ASYNC PREFETCH NEXT TILE (overlap with compute)
//         // ====================================================================
//         if (tile_k + 1 < num_tiles) {
//             int next_k = (tile_k + 1) * BK;
            
//             // Prefetch A tile
//             for (int i = tid; i < a_tile_stride; i += THREADS_PER_BLOCK) {
//                 int row = i / BK;
//                 int col = i % BK;
//                 int global_row = block_row + row;
//                 int global_col = next_k + col;
                
//                 if (global_row < M && global_col < K) {
//                     As[write_idx][row][col] = A[global_row * K + global_col];
//                 } else {
//                     As[write_idx][row][col] = __float2half(0.0f);
//                 }
//             }
            
//             // Prefetch B tile
//             for (int i = tid; i < b_tile_stride; i += THREADS_PER_BLOCK) {
//                 int row = i / BN;
//                 int col = i % BN;
//                 int global_row = next_k + row;
//                 int global_col = block_col + col;
                
//                 if (global_row < K && global_col < N) {
//                     Bs[write_idx][row][col] = B[global_row * N + global_col];
//                 } else {
//                     Bs[write_idx][row][col] = __float2half(0.0f);
//                 }
//             }
//         }
        
//         // ====================================================================
//         // TENSOR CORE COMPUTATION (while prefetch happens)
//         // ====================================================================
        
//         // Process BK elements with WMMA (BK=32, WMMA_K=16 → 2 iterations)
//         #pragma unroll
//         for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
//             // Each warp computes 2x2 WMMA tiles (32x32 warp tile)
//             #pragma unroll
//             for (int wm = 0; wm < 2; wm++) {
//                 #pragma unroll
//                 for (int wn = 0; wn < 2; wn++) {
//                     int warp_m_base = warp_row * WM + wm * WMMA_M;
//                     int warp_n_base = warp_col * WN + wn * WMMA_N;
                    
//                     // Load A fragment
//                     wmma::load_matrix_sync(
//                         a_frag,
//                         &As[read_idx][warp_m_base][k_step],
//                         BK + PAD
//                     );
                    
//                     // Load B fragment
//                     wmma::load_matrix_sync(
//                         b_frag,
//                         &Bs[read_idx][k_step][warp_n_base],
//                         BN + PAD
//                     );
                    
//                     // Tensor Core MMA
//                     wmma::mma_sync(acc_frag[wm][wn], a_frag, b_frag, acc_frag[wm][wn]);
//                 }
//             }
//         }
        
//         __syncthreads();
//     }
    
//     // ========================================================================
//     // STORE RESULTS TO GLOBAL MEMORY
//     // ========================================================================
//     #pragma unroll
//     for (int wm = 0; wm < 2; wm++) {
//         #pragma unroll
//         for (int wn = 0; wn < 2; wn++) {
//             int c_row = block_row + warp_row * WM + wm * WMMA_M;
//             int c_col = block_col + warp_col * WN + wn * WMMA_N;
            
//             // Fast path: entire tile fits
//             if (c_row + WMMA_M <= M && c_col + WMMA_N <= N) {
//                 wmma::store_matrix_sync(
//                     &C[c_row * N + c_col],
//                     acc_frag[wm][wn],
//                     N,
//                     wmma::mem_row_major
//                 );
//             }
//             // Boundary case: element-wise store
//             else if (c_row < M && c_col < N) {
//                 __half temp[WMMA_M * WMMA_N];
//                 wmma::store_matrix_sync(temp, acc_frag[wm][wn], WMMA_N, wmma::mem_row_major);
                
//                 for (int i = 0; i < WMMA_M; i++) {
//                     for (int j = 0; j < WMMA_N; j++) {
//                         if (c_row + i < M && c_col + j < N) {
//                             C[(c_row + i) * N + (c_col + j)] = temp[i * WMMA_N + j];
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// // ============================================================================
// // BFLOAT16 TENSOR CORE KERNEL - CUDA-L2 Optimized
// // ============================================================================

// template<int BM, int BN, int BK, int WM, int WN>
// __global__ void matmul_bf16_optimized(
//     const __nv_bfloat16* __restrict__ A,
//     const __nv_bfloat16* __restrict__ B,
//     __nv_bfloat16* __restrict__ C,
//     int M, int N, int K,
//     int total_batches,
//     const int* a_shape, const int* b_shape, const int* out_shape,
//     const int* a_strides, const int* b_strides, const int* out_strides,
//     int a_ndim, int b_ndim, int out_ndim)
// {
//     const int bx = blockIdx.x;
//     const int by = blockIdx.y;
//     const int batch_idx = blockIdx.z;
    
//     if (batch_idx >= total_batches) return;
    
//     const int tid = threadIdx.x;
//     const int warp_id = tid / WARP_SIZE;
//     const int warp_row = warp_id / WARPS_N;
//     const int warp_col = warp_id % WARPS_N;
    
//     // Calculate batch offsets
//     int a_batch_offset = 0, b_batch_offset = 0, out_batch_offset = 0;
//     compute_batch_offset(batch_idx, a_shape, a_strides, a_ndim, out_shape, out_ndim, a_batch_offset);
//     compute_batch_offset(batch_idx, b_shape, b_strides, b_ndim, out_shape, out_ndim, b_batch_offset);
//     compute_batch_offset(batch_idx, out_shape, out_strides, out_ndim, out_shape, out_ndim, out_batch_offset);
    
//     A += a_batch_offset;
//     B += b_batch_offset;
//     C += out_batch_offset;
    
//     // Double-buffered shared memory
//     __shared__ __nv_bfloat16 As[2][BM][BK + PAD];
//     __shared__ __nv_bfloat16 Bs[2][BK][BN + PAD];
    
//     // WMMA fragments (FP32 accumulation for better precision)
//     wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> a_frag;
//     wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __nv_bfloat16, wmma::row_major> b_frag;
//     wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc_frag[2][2];
    
//     // Initialize accumulators
//     #pragma unroll
//     for (int i = 0; i < 2; i++) {
//         #pragma unroll
//         for (int j = 0; j < 2; j++) {
//             wmma::fill_fragment(acc_frag[i][j], 0.0f);
//         }
//     }
    
//     const int block_row = by * BM;
//     const int block_col = bx * BN;
//     const int a_tile_stride = BM * BK;
//     const int b_tile_stride = BK * BN;
//     const int num_tiles = (K + BK - 1) / BK;
    
//     int write_idx = 0;
//     int read_idx = 1;
    
//     // Prefetch first tile
//     {
//         for (int i = tid; i < a_tile_stride; i += THREADS_PER_BLOCK) {
//             int row = i / BK;
//             int col = i % BK;
//             int global_row = block_row + row;
//             int global_col = col;
            
//             if (global_row < M && global_col < K) {
//                 As[write_idx][row][col] = A[global_row * K + global_col];
//             } else {
//                 As[write_idx][row][col] = __float2bfloat16(0.0f);
//             }
//         }
        
//         for (int i = tid; i < b_tile_stride; i += THREADS_PER_BLOCK) {
//             int row = i / BN;
//             int col = i % BN;
//             int global_row = row;
//             int global_col = block_col + col;
            
//             if (global_row < K && global_col < N) {
//                 Bs[write_idx][row][col] = B[global_row * N + global_col];
//             } else {
//                 Bs[write_idx][row][col] = __float2bfloat16(0.0f);
//             }
//         }
//     }
    
//     __syncthreads();
    
//     // Main computation loop
//     for (int tile_k = 0; tile_k < num_tiles; tile_k++) {
//         read_idx = write_idx;
//         write_idx = 1 - write_idx;
        
//         // Prefetch next tile
//         if (tile_k + 1 < num_tiles) {
//             int next_k = (tile_k + 1) * BK;
            
//             for (int i = tid; i < a_tile_stride; i += THREADS_PER_BLOCK) {
//                 int row = i / BK;
//                 int col = i % BK;
//                 int global_row = block_row + row;
//                 int global_col = next_k + col;
                
//                 if (global_row < M && global_col < K) {
//                     As[write_idx][row][col] = A[global_row * K + global_col];
//                 } else {
//                     As[write_idx][row][col] = __float2bfloat16(0.0f);
//                 }
//             }
            
//             for (int i = tid; i < b_tile_stride; i += THREADS_PER_BLOCK) {
//                 int row = i / BN;
//                 int col = i % BN;
//                 int global_row = next_k + row;
//                 int global_col = block_col + col;
                
//                 if (global_row < K && global_col < N) {
//                     Bs[write_idx][row][col] = B[global_row * N + global_col];
//                 } else {
//                     Bs[write_idx][row][col] = __float2bfloat16(0.0f);
//                 }
//             }
//         }
        
//         // Tensor Core computation
//         #pragma unroll
//         for (int k_step = 0; k_step < BK; k_step += WMMA_K) {
//             #pragma unroll
//             for (int wm = 0; wm < 2; wm++) {
//                 #pragma unroll
//                 for (int wn = 0; wn < 2; wn++) {
//                     int warp_m_base = warp_row * WM + wm * WMMA_M;
//                     int warp_n_base = warp_col * WN + wn * WMMA_N;
                    
//                     wmma::load_matrix_sync(a_frag, &As[read_idx][warp_m_base][k_step], BK + PAD);
//                     wmma::load_matrix_sync(b_frag, &Bs[read_idx][k_step][warp_n_base], BN + PAD);
//                     wmma::mma_sync(acc_frag[wm][wn], a_frag, b_frag, acc_frag[wm][wn]);
//                 }
//             }
//         }
        
//         __syncthreads();
//     }
    
//     // Store results with FP32→BF16 conversion
//     __shared__ float temp_output[NUM_WARPS][2][2][WMMA_M * WMMA_N];
    
//     #pragma unroll
//     for (int wm = 0; wm < 2; wm++) {
//         #pragma unroll
//         for (int wn = 0; wn < 2; wn++) {
//             wmma::store_matrix_sync(temp_output[warp_id][wm][wn], acc_frag[wm][wn], WMMA_N, wmma::mem_row_major);
//         }
//     }
    
//     __syncthreads();
    
//     #pragma unroll
//     for (int wm = 0; wm < 2; wm++) {
//         #pragma unroll
//         for (int wn = 0; wn < 2; wn++) {
//             int c_row = block_row + warp_row * WM + wm * WMMA_M;
//             int c_col = block_col + warp_col * WN + wn * WMMA_N;
            
//             if (c_row < M && c_col < N) {
//                 for (int i = 0; i < WMMA_M; i++) {
//                     for (int j = 0; j < WMMA_N; j++) {
//                         if (c_row + i < M && c_col + j < N) {
//                             C[(c_row + i) * N + (c_col + j)] = 
//                                 __float2bfloat16(temp_output[warp_id][wm][wn][i * WMMA_N + j]);
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

// // ============================================================================
// // HOST FUNCTION - Optimized Dispatch
// // ============================================================================

// template<typename T>
// void cuda_matmul_optimized_impl(
//     const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream)
// {
//     const auto& a_shape = A.shape().dims;
//     const auto& b_shape = B.shape().dims;
//     const auto& out_shape = output.shape().dims;
    
//     size_t a_ndim = a_shape.size();
//     size_t b_ndim = b_shape.size();
//     size_t out_ndim = out_shape.size();
    
//     // Matrix dimensions
//     int m = static_cast<int>(a_shape[a_ndim - 2]);
//     int k = static_cast<int>(a_shape[a_ndim - 1]);
//     int n = static_cast<int>(b_shape[b_ndim - 1]);
    
//     // Calculate total batches
//     int total_batches = 1;
//     for (size_t i = 0; i < out_ndim - 2; ++i) {
//         total_batches *= static_cast<int>(out_shape[i]);
//     }
    
//     // Grid configuration
//     dim3 block(THREADS_PER_BLOCK);
//     dim3 grid(
//         (n + BN - 1) / BN,
//         (m + BM - 1) / BM,
//         total_batches
//     );
    
//     // Allocate device memory for shapes and strides
//     int *d_a_shape, *d_b_shape, *d_out_shape;
//     int *d_a_strides, *d_b_strides, *d_out_strides;
    
//     cudaMalloc(&d_a_shape, a_ndim * sizeof(int));
//     cudaMalloc(&d_b_shape, b_ndim * sizeof(int));
//     cudaMalloc(&d_out_shape, out_ndim * sizeof(int));
//     cudaMalloc(&d_a_strides, a_ndim * sizeof(int));
//     cudaMalloc(&d_b_strides, b_ndim * sizeof(int));
//     cudaMalloc(&d_out_strides, out_ndim * sizeof(int));
    
//     // Copy shapes and strides to device
//     std::vector<int> a_shape_int(a_shape.begin(), a_shape.end());
//     std::vector<int> b_shape_int(b_shape.begin(), b_shape.end());
//     std::vector<int> out_shape_int(out_shape.begin(), out_shape.end());
//     std::vector<int> a_strides_int(A.stride().strides.begin(), A.stride().strides.end());
//     std::vector<int> b_strides_int(B.stride().strides.begin(), B.stride().strides.end());
//     std::vector<int> out_strides_int(output.stride().strides.begin(), output.stride().strides.end());
    
//     cudaMemcpyAsync(d_a_shape, a_shape_int.data(), a_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
//     cudaMemcpyAsync(d_b_shape, b_shape_int.data(), b_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
//     cudaMemcpyAsync(d_out_shape, out_shape_int.data(), out_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
//     cudaMemcpyAsync(d_a_strides, a_strides_int.data(), a_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
//     cudaMemcpyAsync(d_b_strides, b_strides_int.data(), b_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
//     cudaMemcpyAsync(d_out_strides, out_strides_int.data(), out_ndim * sizeof(int), cudaMemcpyHostToDevice, stream);
    
//     const T* a_ptr = A.data<T>();
//     const T* b_ptr = B.data<T>();
//     T* out_ptr = output.data<T>();
    
//     // Launch appropriate kernel
//     if constexpr (std::is_same<T, __half>::value) {
//         matmul_fp16_optimized<BM, BN, BK, WM, WN><<<grid, block, 0, stream>>>(
//             reinterpret_cast<const __half*>(a_ptr),
//             reinterpret_cast<const __half*>(b_ptr),
//             reinterpret_cast<__half*>(out_ptr),
//             m, n, k, total_batches,
//             d_a_shape, d_b_shape, d_out_shape,
//             d_a_strides, d_b_strides, d_out_strides,
//             static_cast<int>(a_ndim), static_cast<int>(b_ndim), static_cast<int>(out_ndim)
//         );
//     } else if constexpr (std::is_same<T, __nv_bfloat16>::value) {
//         matmul_bf16_optimized<BM, BN, BK, WM, WN><<<grid, block, 0, stream>>>(
//             reinterpret_cast<const __nv_bfloat16*>(a_ptr),
//             reinterpret_cast<const __nv_bfloat16*>(b_ptr),
//             reinterpret_cast<__nv_bfloat16*>(out_ptr),
//             m, n, k, total_batches,
//             d_a_shape, d_b_shape, d_out_shape,
//             d_a_strides, d_b_strides, d_out_strides,
//             static_cast<int>(a_ndim), static_cast<int>(b_ndim), static_cast<int>(out_ndim)
//         );
//     } else {
//         cudaFree(d_a_shape); cudaFree(d_b_shape); cudaFree(d_out_shape);
//         cudaFree(d_a_strides); cudaFree(d_b_strides); cudaFree(d_out_strides);
//         throw std::runtime_error("Optimized matmul only supports __half and __nv_bfloat16");
//     }
    
//     // Check for errors
//     cudaError_t err = cudaGetLastError();
//     if (err != cudaSuccess) {
//         cudaFree(d_a_shape); cudaFree(d_b_shape); cudaFree(d_out_shape);
//         cudaFree(d_a_strides); cudaFree(d_b_strides); cudaFree(d_out_strides);
//         throw std::runtime_error("Optimized matmul kernel failed: " + std::string(cudaGetErrorString(err)));
//     }
    
//     // Free device memory
//     cudaFree(d_a_shape);
//     cudaFree(d_b_shape);
//     cudaFree(d_out_shape);
//     cudaFree(d_a_strides);
//     cudaFree(d_b_strides);
//     cudaFree(d_out_strides);
// }

// // ============================================================================
// // PUBLIC API
// // ============================================================================

// void cuda_matmul(const Tensor& A, const Tensor& B, Tensor& output, cudaStream_t stream)
// {
//     dispatch_by_dtype(A.dtype(), [&](auto dummy) {
//         using T = decltype(dummy);
        
//         if constexpr (std::is_same<T, __half>::value || std::is_same<T, __nv_bfloat16>::value) {
//             cuda_matmul_optimized_impl<T>(A, B, output, stream);
//         } else {
//             throw std::runtime_error("cuda_matmul: Only FP16 and BF16 are supported in optimized kernel. Use cuBLAS for other types.");
//         }
//     });
// }

// } // namespace OwnTensor

// #endif // WITH_CUDA