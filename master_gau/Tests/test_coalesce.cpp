#include <cstdio>
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"
using namespace OwnTensor;

void fill(Tensor& t, std::mt19937& r) {
    float* d = t.data<float>();
    std::normal_distribution<float> n(0,1);
    for(size_t i=0;i<t.numel();++i) d[i]=n(r);
}

double np_mean(const float* d, const std::vector<int64_t>& shape, const std::vector<int64_t>& axes) {
    // Brute force reference: compute mean by iterating all elements
    int64_t ndim = shape.size();
    std::vector<int64_t> strides(ndim);
    strides[ndim-1] = 1;
    for(int i=ndim-2;i>=0;--i) strides[i] = strides[i+1]*shape[i+1];
    
    // For output position 0, compute mean
    bool is_reduced[8] = {};
    for(auto a:axes) is_reduced[a] = true;
    
    int64_t red_count = 1;
    for(auto a:axes) red_count *= shape[a];
    
    // Iterate all elements, check if non-reduced coords are all 0
    int64_t total = 1;
    for(auto s:shape) total *= s;
    
    double sum = 0;
    int64_t count = 0;
    for(int64_t idx=0; idx<total; ++idx) {
        // Decompose idx into coordinates
        int64_t rem = idx;
        bool all_zero = true;
        for(int i=0;i<ndim;++i) {
            int64_t coord = rem / strides[i];
            rem %= strides[i];
            if(!is_reduced[i] && coord != 0) { all_zero = false; break; }
        }
        if(all_zero) { sum += (double)d[idx]; count++; }
    }
    return sum / count;
}

void test(const char* label, std::vector<int64_t> shape, std::vector<int64_t> axes, std::mt19937& rng) {
    Tensor t({Shape(shape)}, TensorOptions().with_dtype(Dtype::Float32));
    fill(t, rng);
    
    // Our result
    Tensor result = reduce_mean(t, axes);
    float our_val = result.data<float>()[0];
    
    // Reference
    double ref_val = np_mean(t.data<float>(), shape, axes);
    
    double err = std::abs((double)our_val - ref_val) / std::abs(ref_val);
    
    // Check shape
    auto& out_shape = result.shape().dims;
    std::string ss="("; for(size_t i=0;i<shape.size();++i){if(i)ss+=",";ss+=std::to_string(shape[i]);}ss+=")";
    std::string ds="("; for(size_t i=0;i<axes.size();++i){if(i)ds+=",";ds+=std::to_string(axes[i]);}ds+=")";
    std::string os="("; for(size_t i=0;i<out_shape.size();++i){if(i)os+=",";os+=std::to_string(out_shape[i]);}os+=")";
    
    const char* status = (err < 1e-5) ? "PASS" : "FAIL";
    printf("  %-30s shape=%-20s axes=%-10s out=%-15s err=%.2e  [%s]\n",
           label, ss.c_str(), ds.c_str(), os.c_str(), err, status);
}

int main() {
    std::mt19937 rng(42);
    printf("COALESCE+REORDER CORRECTNESS TESTS\n");
    printf("===================================\n\n");
    
    printf("--- Case 1: InnerContiguous (rightmost dims) ---\n");
    test("Last dim 2D", {32,768}, {1}, rng);
    test("Last dim 3D", {32,128,768}, {2}, rng);
    test("Last 2 dims", {32,128,768}, {1,2}, rng);
    test("Last 3 dims (full)", {4,8,16}, {0,1,2}, rng);
    test("1D full reduction", {10000}, {0}, rng);
    
    printf("\n--- Case 2: OuterContiguous (leftmost dims) ---\n");
    test("First dim 2D", {1000,256}, {0}, rng);
    test("First dim 3D", {32,128,768}, {0}, rng);
    test("First 2 dims", {32,128,768}, {0,1}, rng);
    test("First dim 4D", {16,64,32,32}, {0}, rng);
    
    printf("\n--- Case 3: Middle consecutive (COALESCED → OuterContiguous) ---\n");
    test("Mid dim 3D", {32,512,768}, {1}, rng);
    test("Mid dim 4D", {32,256,56,56}, {1}, rng);
    test("Mid 2 dims 4D", {16,64,32,32}, {1,2}, rng);
    test("BERT-pool", {16,512,1024}, {1}, rng);
    test("ViT-patch", {32,197,768}, {1}, rng);
    test("Attn-head", {32,12,128,128}, {1}, rng);
    test("Channel-mean", {8,256,14,14}, {1}, rng);
    test("Mid dim 5D", {4,8,16,32,64}, {2}, rng);
    test("Mid 2 dims 5D", {4,8,16,32,64}, {1,2}, rng);
    test("Mid 3 dims 5D", {4,8,16,32,64}, {1,2,3}, rng);
    
    printf("\n--- Case 4: Non-consecutive axes (Generic carry-add) ---\n");
    test("Dims (0,2) 3D", {100,200,50}, {0,2}, rng);
    test("Dims (0,2) 4D", {16,64,32,32}, {0,2}, rng);
    test("Dims (1,3) 4D", {16,64,32,32}, {1,3}, rng);
    test("Dims (0,3) 4D", {8,16,32,64}, {0,3}, rng);
    test("Dims (0,2,4) 5D", {4,8,16,32,8}, {0,2,4}, rng);
    
    printf("\n--- Case 5: Edge cases ---\n");
    test("Single element", {1}, {0}, rng);
    test("Single row", {1,1000}, {1}, rng);
    test("Single col", {1000,1}, {0}, rng);
    test("Reduce all 4D", {4,8,16,32}, {0,1,2,3}, rng);
    test("Dim=0 of (1,X)", {1,10000}, {0}, rng);
    test("Large mid", {2,100000,3}, {1}, rng);
    
    printf("\n--- Case 6: NanMean correctness ---\n");
    {
        Tensor t({Shape({5})}, TensorOptions().with_dtype(Dtype::Float32));
        float* d = t.data<float>();
        d[0]=1; d[1]=std::nanf(""); d[2]=3; d[3]=std::nanf(""); d[4]=5;
        Tensor r = reduce_nanmean(t);
        printf("  nanmean([1,NaN,3,NaN,5]) = %.6f (expected 3.0) [%s]\n",
               (double)r.data<float>()[0], 
               std::abs(r.data<float>()[0]-3.0f)<1e-5 ? "PASS" : "FAIL");
    }
    {
        Tensor t({Shape({4,5})}, TensorOptions().with_dtype(Dtype::Float32));
        float* d = t.data<float>();
        for(int i=0;i<20;++i) d[i] = (float)(i+1);
        d[1]=std::nanf(""); d[7]=std::nanf(""); d[13]=std::nanf(""); d[19]=std::nanf("");
        // Row 0: [1,NaN,3,4,5] → mean=3.25
        Tensor r = reduce_nanmean(t, {1});
        printf("  nanmean([[1,NaN,3,4,5],...], dim=1)[0] = %.6f (expected 3.25) [%s]\n",
               (double)r.data<float>()[0],
               std::abs(r.data<float>()[0]-3.25f)<1e-5 ? "PASS" : "FAIL");
    }
    {
        // NanMean with middle dim reduction
        Tensor t({Shape({2,3,4})}, TensorOptions().with_dtype(Dtype::Float32));
        float* d = t.data<float>();
        for(int i=0;i<24;++i) d[i] = (float)(i+1);
        d[4]=std::nanf(""); d[8]=std::nanf(""); // positions in dim=1
        Tensor r = reduce_nanmean(t, {1});
        printf("  nanmean(2x3x4, dim=1)[0,0] = %.6f [%s]\n",
               (double)r.data<float>()[0],
               !std::isnan(r.data<float>()[0]) ? "PASS" : "FAIL");
    }
    
    printf("\nALL TESTS COMPLETE\n");
    return 0;
}
