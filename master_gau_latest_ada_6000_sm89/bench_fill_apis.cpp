// =============================================================================
// bench_fill_apis.cpp — author: Gautam_Reddy_1926
//
// Goal: for every (dtype, size) pair, find the fastest way to set a buffer to
//   (a) zero, (b) ones.
//
// Methods compared
//   ZERO path:
//     - cudaMemsetAsync (runtime API, byte pattern)
//     - cuMemsetD8Async  (driver API, 1-byte pattern)
//     - cuMemsetD16Async (driver API, 2-byte pattern)
//     - cuMemsetD32Async (driver API, 4-byte pattern)
//     - our fill_cuda_launch<T>(0)
//   ONES path: same set, but the byte/half/word pattern is the bit-rep of 1.0
//     for floating types, or the integer 1.
//
// cudaMemsetAsync only writes byte patterns — for non-zero ONES on >1-byte
// types it produces WRONG VALUES; we still time it as a "raw bandwidth" lower
// bound, but mark it ❌ in correctness column.
//
// Build:
//   g++ -Iinclude -I/usr/local/cuda-13.0/include -DWITH_CUDA -std=c++2a -O3 \
//       -mavx2 -mfma -mf16c bench_fill_apis.cpp \
//       -Llib -ltensor -L/usr/local/cuda-13.0/lib64 \
//       -lcudart -lcuda -ltbb -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml \
//       -Wl,-rpath,'$ORIGIN/lib' -o bench_fill_apis
//
// Run:  CUDA_VISIBLE_DEVICES=6 ./bench_fill_apis
// =============================================================================

#include "core/Tensor.h"
#include "core/TensorDataManip.h"
#include "core/TensorDispatch.h"
#include "ops/helpers/FillKernels.h"
#include "device/DeviceCore.h"

#include <cuda_runtime.h>
#include <cuda.h>          // driver API: cuMemsetD8/D16/D32Async
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using namespace OwnTensor;
using Clock = std::chrono::high_resolution_clock;

static double now_ms() {
    auto t = Clock::now().time_since_epoch();
    return std::chrono::duration<double, std::milli>(t).count();
}

#define CK_CUDA(call) do { \
    cudaError_t e = (call); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA err %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        std::abort(); \
    } \
} while(0)

#define CK_DRV(call) do { \
    CUresult e = (call); \
    if (e != CUDA_SUCCESS) { \
        const char* s = nullptr; cuGetErrorString(e, &s); \
        fprintf(stderr, "DRV err %s:%d %s\n", __FILE__, __LINE__, s ? s : "?"); \
        std::abort(); \
    } \
} while(0)

// -------- bit pattern helpers (host side) -------------------------------------
template<typename T> static uint8_t  ones_byte()  { T v = T(1); uint8_t  o; std::memcpy(&o, &v, 1); return o; }
template<typename T> static uint16_t ones_word16(){ T v = T(1); uint16_t o; std::memcpy(&o, &v, 2); return o; }
template<typename T> static uint32_t ones_word32(){ T v = T(1); uint32_t o; std::memcpy(&o, &v, 4); return o; }

// -------- timing helpers ------------------------------------------------------
struct TimedResult {
    double ms;        // mean ms/op
    double gbs;       // GB/s
    bool   correct;   // value-correctness against reference
};

static double bytes_to_gb(int64_t b) { return double(b) / 1e9; }

// Verify a small prefix of GPU memory matches the expected byte pattern.
// `pat_bytes` describes the repeating unit (1, 2, or 4 bytes). For non-byte
// patterns we read back enough bytes to span ~64 elements.
static bool verify_pattern(void* dptr, const uint8_t* pat, int pat_bytes,
                           int64_t total_bytes, cudaStream_t s)
{
    int64_t check_bytes = std::min<int64_t>(total_bytes, 256);
    std::vector<uint8_t> host(check_bytes);
    CK_CUDA(cudaMemcpyAsync(host.data(), dptr, check_bytes, cudaMemcpyDeviceToHost, s));
    CK_CUDA(cudaStreamSynchronize(s));
    for (int64_t i = 0; i < check_bytes; ++i) {
        if (host[i] != pat[i % pat_bytes]) return false;
    }
    return true;
}

template<typename Fn>
static TimedResult time_op(int warm, int iters, cudaStream_t s, Fn&& op,
                           int64_t total_bytes, const uint8_t* expected_pat,
                           int pat_bytes, void* check_ptr)
{
    for (int i = 0; i < warm; ++i) op();
    CK_CUDA(cudaStreamSynchronize(s));
    double t0 = now_ms();
    for (int i = 0; i < iters; ++i) op();
    CK_CUDA(cudaStreamSynchronize(s));
    double ms = (now_ms() - t0) / iters;
    bool ok = expected_pat ? verify_pattern(check_ptr, expected_pat, pat_bytes,
                                            total_bytes, s) : true;
    return { ms, bytes_to_gb(total_bytes) / (ms / 1000.0), ok };
}

// =============================================================================
// Per-dtype runner
// =============================================================================
template<typename T>
static void run_dtype(const char* dtype_name, Dtype dt,
                      const std::vector<std::pair<const char*, int64_t>>& sizes,
                      int warm, int iters)
{
    auto opts = TensorOptions().with_dtype(dt).with_device(Device::CUDA);
    cudaStream_t s = cuda::getCurrentStream();
    CUstream cs = (CUstream)s;

    printf("\n==================== dtype = %s (sizeof=%zu) ====================\n",
           dtype_name, sizeof(T));
    printf("%-32s | %-7s | %18s | %18s | %18s | %18s | %18s\n",
           "size", "OP",
           "cudaMemsetAsync", "cuMemsetD8Async", "cuMemsetD16Async",
           "cuMemsetD32Async", "fill_cuda_launch");
    printf("%.*s\n", 170, "----------------------------------------------------------------"
                          "----------------------------------------------------------------"
                          "----------------------------------------");

    for (auto& [label, n] : sizes) {
        Tensor t(Shape{{n}}, opts);
        void* p = t.data();
        CUdeviceptr dp = (CUdeviceptr)p;
        int64_t nbytes = n * sizeof(T);

        // ----------- ZERO ----------------------------------------------------
        uint8_t zpat[4] = {0,0,0,0};

        TimedResult z_rt = time_op(warm, iters, s,
            [&]{ CK_CUDA(cudaMemsetAsync(p, 0, nbytes, s)); },
            nbytes, zpat, 1, p);

        TimedResult z_d8 = time_op(warm, iters, s,
            [&]{ CK_DRV(cuMemsetD8Async(dp, 0, nbytes, cs)); },
            nbytes, zpat, 1, p);

        TimedResult z_d16 { -1, -1, false };
        if (sizeof(T) >= 2 && nbytes % 2 == 0) {
            z_d16 = time_op(warm, iters, s,
                [&]{ CK_DRV(cuMemsetD16Async(dp, 0, nbytes / 2, cs)); },
                nbytes, zpat, 1, p);
        }

        TimedResult z_d32 { -1, -1, false };
        if (sizeof(T) >= 4 && nbytes % 4 == 0) {
            z_d32 = time_op(warm, iters, s,
                [&]{ CK_DRV(cuMemsetD32Async(dp, 0, nbytes / 4, cs)); },
                nbytes, zpat, 1, p);
        }

        TimedResult z_fk { -1, -1, false };
        if constexpr (detail::fill_cuda_supported<T>) {
            z_fk = time_op(warm, iters, s,
                [&]{ cuda::fill_cuda_launch<T>(reinterpret_cast<T*>(p), T(0), n, s); },
                nbytes, zpat, 1, p);
        }

        printf("%-32s | %-7s | %7.4f ms %5.1f GB/s | %7.4f ms %5.1f GB/s | ",
               label, "ZERO",
               z_rt.ms, z_rt.gbs, z_d8.ms, z_d8.gbs);
        if (z_d16.ms >= 0) printf("%7.4f ms %5.1f GB/s | ", z_d16.ms, z_d16.gbs);
        else               printf("%18s | ", "n/a");
        if (z_d32.ms >= 0) printf("%7.4f ms %5.1f GB/s | ", z_d32.ms, z_d32.gbs);
        else               printf("%18s | ", "n/a");
        if (z_fk.ms  >= 0) printf("%7.4f ms %5.1f GB/s\n", z_fk.ms, z_fk.gbs);
        else               printf("%18s\n", "n/a");

        // ----------- ONES ----------------------------------------------------
        // Ground-truth byte pattern for value 1 in this dtype.
        uint8_t opat[8] = {0};
        std::memset(opat, 0, sizeof(opat));
        { T v = T(1); std::memcpy(opat, &v, std::min(sizeof(T), sizeof(opat))); }

        // cudaMemsetAsync with byte 0x01 — only correct if all bytes of 1 are 0x01
        // (true for int8/uint8/bool ONLY). We still time it.
        TimedResult o_rt = time_op(warm, iters, s,
            [&]{ CK_CUDA(cudaMemsetAsync(p, 0x01, nbytes, s)); },
            nbytes, opat, (int)std::min(sizeof(T), (size_t)4), p);

        // cuMemsetD8Async with byte 0x01 — same caveat as cudaMemsetAsync.
        TimedResult o_d8 = time_op(warm, iters, s,
            [&]{ CK_DRV(cuMemsetD8Async(dp, 0x01, nbytes, cs)); },
            nbytes, opat, (int)std::min(sizeof(T), (size_t)4), p);

        // cuMemsetD16Async with the 2-byte rep of 1 in this dtype.
        TimedResult o_d16 { -1, -1, false };
        if (sizeof(T) == 2 && nbytes % 2 == 0) {
            uint16_t w = ones_word16<T>();
            o_d16 = time_op(warm, iters, s,
                [&]{ CK_DRV(cuMemsetD16Async(dp, w, nbytes / 2, cs)); },
                nbytes, opat, 2, p);
        }

        // cuMemsetD32Async with the 4-byte rep of 1 in this dtype.
        TimedResult o_d32 { -1, -1, false };
        if (sizeof(T) == 4 && nbytes % 4 == 0) {
            uint32_t w = ones_word32<T>();
            o_d32 = time_op(warm, iters, s,
                [&]{ CK_DRV(cuMemsetD32Async(dp, w, nbytes / 4, cs)); },
                nbytes, opat, 4, p);
        }

        TimedResult o_fk { -1, -1, false };
        if constexpr (detail::fill_cuda_supported<T>) {
            o_fk = time_op(warm, iters, s,
                [&]{ cuda::fill_cuda_launch<T>(reinterpret_cast<T*>(p), T(1), n, s); },
                nbytes, opat, (int)std::min(sizeof(T), (size_t)4), p);
        }

        auto mark = [](const TimedResult& r) { return r.correct ? "✓" : "✗"; };

        printf("%-32s | %-7s | %s%6.4f ms %5.1f GB/s | %s%6.4f ms %5.1f GB/s | ",
               label, "ONES",
               mark(o_rt),  o_rt.ms,  o_rt.gbs,
               mark(o_d8),  o_d8.ms,  o_d8.gbs);
        if (o_d16.ms >= 0) printf("%s%6.4f ms %5.1f GB/s | ", mark(o_d16), o_d16.ms, o_d16.gbs);
        else               printf("%18s | ", "n/a");
        if (o_d32.ms >= 0) printf("%s%6.4f ms %5.1f GB/s | ", mark(o_d32), o_d32.ms, o_d32.gbs);
        else               printf("%18s | ", "n/a");
        if (o_fk.ms  >= 0) printf("%s%6.4f ms %5.1f GB/s\n", mark(o_fk),  o_fk.ms,  o_fk.gbs);
        else               printf("%18s\n", "n/a");
    }
}

// =============================================================================
int main()
{
    // Init driver context (cudaFree(0) lazily creates a primary context that
    // the driver API can piggyback on — required before any cu* call).
    CK_CUDA(cudaFree(0));

    // Sizes spanning DL-relevant scales.
    std::vector<std::pair<const char*, int64_t>> sizes = {
        { "1K (4 KB f32)",                            1'000           },
        { "64K (256 KB f32)",                        64'000           },
        { "1M (4 MB f32)",                        1'000'000           },
        { "MLP weight 3072*768 (9 MB f32)",       3072LL * 768        },
        { "attn B*T*E 16*1024*768 (50 MB f32)",   16LL * 1024 * 768   },
        { "embed 50304*768 (154 MB f32)",         50304LL * 768       },
        { "256M (1 GB f32)",                  256'000'000             },
    };

    const int WARM  = 10;
    const int ITERS = 100;

    run_dtype<float>   ("Float32", Dtype::Float32, sizes, WARM, ITERS);
    run_dtype<double>  ("Float64", Dtype::Float64, sizes, WARM, ITERS);
    run_dtype<float16_t>("Float16", Dtype::Float16, sizes, WARM, ITERS);
    run_dtype<bfloat16_t>("BFloat16", Dtype::Bfloat16, sizes, WARM, ITERS);
    run_dtype<int8_t>  ("Int8",    Dtype::Int8,    sizes, WARM, ITERS);
    run_dtype<int16_t> ("Int16",   Dtype::Int16,   sizes, WARM, ITERS);
    run_dtype<int32_t> ("Int32",   Dtype::Int32,   sizes, WARM, ITERS);
    run_dtype<int64_t> ("Int64",   Dtype::Int64,   sizes, WARM, ITERS);
    run_dtype<bool>    ("Bool",    Dtype::Bool,    sizes, WARM, ITERS);

    printf("\nLegend: ✓ = result matches expected pattern, ✗ = WRONG values "
           "(byte-pattern API is being used on a multi-byte dtype where the "
           "per-byte rep of 1 isn't 0x01 — those rows are bandwidth-only).\n");

    printf("\nDecision rule (Gautam_Reddy_1926):\n"
           "  ZERO: byte pattern is always 0x00 → cudaMemsetAsync wins on every "
           "dtype/size (DMA engine, no SMs).\n"
           "  ONES: pick by sizeof(T):\n"
           "    1 B (int8/uint8/bool)   → cudaMemsetAsync(0x01)  (also cuMemsetD8Async)\n"
           "    2 B (half/bf16/int16)   → cuMemsetD16Async(<bit_rep_of_1>)\n"
           "    4 B (float/int32)       → cuMemsetD32Async(<bit_rep_of_1>)\n"
           "    8 B (double/int64)      → fill_cuda_launch<T>  (no D64 driver API)\n");

    return 0;
}
