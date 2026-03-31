// COMPREHENSIVE Mean Benchmark: Our Library
// All dtypes × All paths × All sizes × DL apps × Precision × Edge cases
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"

using namespace OwnTensor;

template<typename F>
double bench(F fn, int warmup=5, int iters=30) {
    for (int i=0;i<warmup;++i) fn();
    std::vector<double> t;
    for (int i=0;i<iters;++i) {
        auto s=std::chrono::high_resolution_clock::now();
        fn();
        auto e=std::chrono::high_resolution_clock::now();
        t.push_back(std::chrono::duration<double,std::micro>(e-s).count());
    }
    std::sort(t.begin(),t.end());
    return t[iters/2];
}

void fill_f32(Tensor& t, std::mt19937& r) {
    float* d=t.data<float>(); std::normal_distribution<float> n(0,1);
    for (size_t i=0;i<t.numel();++i) d[i]=n(r);
}
void fill_f64(Tensor& t, std::mt19937& r) {
    double* d=t.data<double>(); std::normal_distribution<double> n(0,1);
    for (size_t i=0;i<t.numel();++i) d[i]=n(r);
}
double gt_mean_f32(const float* d, size_t n) {
    double s=0; for(size_t i=0;i<n;++i) s+=(double)d[i]; return s/n;
}
double gt_mean_f64(const double* d, size_t n) {
    double s=0; for(size_t i=0;i<n;++i) s+=d[i]; return s/n;
}
double rel_err(double got, double expected) {
    if(expected==0) return got==0?0:1e99;
    if(std::isinf(got)||std::isnan(got)) return 1e99;
    return std::abs(got-expected)/std::abs(expected);
}

#define W 120
void header(const char* t) { printf("\n%.*s\n  %s\n%.*s\n",W,"========================================================================================================================================================================",t,W,"========================================================================================================================================================================"); }
void sub(const char* t) { printf("\n--- %s ---\n",t); }

int main() {
    printf("%.*s\n",W,"========================================================================================================================================================================");
    printf("  COMPREHENSIVE REGULAR MEAN BENCHMARK — Our Library (master_gau)\n");
    printf("  CPU: %d threads (i7-14700K)\n", omp_get_max_threads());
    printf("%.*s\n",W,"========================================================================================================================================================================");
    std::mt19937 rng(42);

    // ============================================================
    header("1. INNERCONTIGUOUS (reduce last dim) — fp32 + fp64 × ALL SIZES");
    // ============================================================
    printf("%22s %6s %10s %12s %12s\n","Shape","Dtype","Time(μs)","rel_err","Strategy");
    printf("----------------------------------------------------------------------\n");

    struct S { int64_t r,c; const char* lbl; };
    S inner[] = {
        {8,16,"tiny"},{32,64,"tiny"},{10,100,"small"},
        {32,768,"layernorm"},{32,4096,"large-LN"},{256,4096,"batch-feat"},
        {4096,768,"seq-LN"},{1000,10000,"medium"},
        {2048,50176,"spatial"},{49152,128,"many-out"},
        {10,1000000,"wide"},{1000000,10,"tall"},
    };

    for (auto& s : inner) {
        if(s.r*s.c>200000000LL) continue;
        int64_t numel=s.r*s.c;
        const char* strat = numel<32768?"Sequential":(s.r>=28?"Strat1":"Strat2");
        // fp32
        { Tensor t({Shape({s.r,s.c})},TensorOptions().with_dtype(Dtype::Float32));
          fill_f32(t,rng);
          double gt=gt_mean_f32(t.data<float>(),s.r*s.c); // row 0 only for speed
          double us=bench([&](){return reduce_mean(t,{1});});
          Tensor res=reduce_mean(t,{1});
          double err=rel_err((double)res.data<float>()[0],gt_mean_f32(t.data<float>()+0,s.c));
          printf("  %20s %6s %10.0f %12.2e %12s\n",
                 (std::string("(")+std::to_string(s.r)+","+std::to_string(s.c)+")").c_str(),
                 "fp32",us,err,strat); }
        // fp64
        { Tensor t({Shape({s.r,s.c})},TensorOptions().with_dtype(Dtype::Float64));
          fill_f64(t,rng);
          double us=bench([&](){return reduce_mean(t,{1});});
          Tensor res=reduce_mean(t,{1});
          double err=rel_err(res.data<double>()[0],gt_mean_f64(t.data<double>()+0,s.c));
          printf("  %20s %6s %10.0f %12.2e %12s\n",
                 (std::string("(")+std::to_string(s.r)+","+std::to_string(s.c)+")").c_str(),
                 "fp64",us,err,strat); }
    }

    // ============================================================
    header("2. OUTERCONTIGUOUS (reduce first dim) — fp32 + fp64");
    // ============================================================
    printf("%22s %6s %10s %12s\n","Shape","Dtype","Time(μs)","rel_err");
    printf("------------------------------------------------------------\n");

    S outer[] = {
        {16,64,"tiny"},{100,256,"small"},{1000,256,"medium"},
        {10000,100,"tall"},{100,10000,"wide"},{50176,2048,"spatial"},
    };
    for (auto& s : outer) {
        if(s.r*s.c>200000000LL) continue;
        { Tensor t({Shape({s.r,s.c})},TensorOptions().with_dtype(Dtype::Float32));
          fill_f32(t,rng);
          double us=bench([&](){return reduce_mean(t,{0});});
          printf("  %20s %6s %10.0f\n",
                 (std::string("(")+std::to_string(s.r)+","+std::to_string(s.c)+")").c_str(),"fp32",us); }
        { Tensor t({Shape({s.r,s.c})},TensorOptions().with_dtype(Dtype::Float64));
          fill_f64(t,rng);
          double us=bench([&](){return reduce_mean(t,{0});});
          printf("  %20s %6s %10.0f\n",
                 (std::string("(")+std::to_string(s.r)+","+std::to_string(s.c)+")").c_str(),"fp64",us); }
    }

    // ============================================================
    header("3. GENERIC (reduce mixed dims) — fp32 + fp64");
    // ============================================================
    printf("%22s %12s %6s %10s\n","Shape","Dims","Dtype","Time(μs)");
    printf("----------------------------------------------------------------------\n");

    auto run_generic = [&](std::vector<int64_t> shape, std::vector<int64_t> dims, const char* slbl, const char* dlbl) {
        Tensor t({Shape(shape)},TensorOptions().with_dtype(Dtype::Float32));
        fill_f32(t,rng);
        double us=bench([&](){return reduce_mean(t,dims);});
        std::string ss="("; for(size_t i=0;i<shape.size();++i){if(i)ss+=",";ss+=std::to_string(shape[i]);}ss+=")";
        std::string ds="("; for(size_t i=0;i<dims.size();++i){if(i)ds+=",";ds+=std::to_string(dims[i]);}ds+=")";
        printf("  %22s %12s %6s %10.0f\n",ss.c_str(),ds.c_str(),"fp32",us);
    };
    auto run_generic64 = [&](std::vector<int64_t> shape, std::vector<int64_t> dims) {
        Tensor t({Shape(shape)},TensorOptions().with_dtype(Dtype::Float64));
        fill_f64(t,rng);
        double us=bench([&](){return reduce_mean(t,dims);});
        std::string ss="("; for(size_t i=0;i<shape.size();++i){if(i)ss+=",";ss+=std::to_string(shape[i]);}ss+=")";
        std::string ds="("; for(size_t i=0;i<dims.size();++i){if(i)ds+=",";ds+=std::to_string(dims[i]);}ds+=")";
        printf("  %22s %12s %6s %10.0f\n",ss.c_str(),ds.c_str(),"fp64",us);
    };

    run_generic({100,200,50},{0,2},"3D XZ","fp32");
    run_generic64({100,200,50},{0,2});
    run_generic({100,200,50},{0},"3D X","fp32");
    run_generic64({100,200,50},{0});
    run_generic({100,200,50},{1,2},"3D YZ","fp32");
    run_generic64({100,200,50},{1,2});
    run_generic({32,128,768},{0,2},"transformer","fp32");
    run_generic64({32,128,768},{0,2});
    run_generic({32,128,768},{1,2},"batch","fp32");
    run_generic64({32,128,768},{1,2});
    run_generic({16,64,32,32},{0,2},"4D XH","fp32");
    run_generic64({16,64,32,32},{0,2});
    run_generic({16,64,32,32},{2,3},"4D spatial","fp32");
    run_generic64({16,64,32,32},{2,3});

    // ============================================================
    header("4. FULL REDUCTION — ALL SIZES × fp32 + fp64");
    // ============================================================
    printf("%12s %6s %10s %12s %10s\n","Size","Dtype","Time(μs)","rel_err","Strategy");
    printf("-------------------------------------------------------\n");

    for (int64_t sz : {100LL,1000LL,10000LL,100000LL,1000000LL,10000000LL,50000000LL}) {
        const char* strat=sz<32768?"Sequential":"Strat2";
        { Tensor t({Shape({sz})},TensorOptions().with_dtype(Dtype::Float32));
          fill_f32(t,rng);
          double gt=gt_mean_f32(t.data<float>(),sz);
          double us=bench([&](){return reduce_mean(t);});
          double val=(double)reduce_mean(t).data<float>()[0];
          printf("  %10lld %6s %10.0f %12.2e %10s\n",(long long)sz,"fp32",us,rel_err(val,gt),strat); }
        { Tensor t({Shape({sz})},TensorOptions().with_dtype(Dtype::Float64));
          fill_f64(t,rng);
          double gt=gt_mean_f64(t.data<double>(),sz);
          double us=bench([&](){return reduce_mean(t);});
          double val=reduce_mean(t).data<double>()[0];
          printf("  %10lld %6s %10.0f %12.2e %10s\n",(long long)sz,"fp64",us,rel_err(val,gt),strat); }
    }

    // ============================================================
    header("5. DL APPLICATION SHAPES — fp32");
    // ============================================================
    printf("%30s %28s %12s %10s\n","Application","Shape","Dim","Time(μs)");
    printf("-------------------------------------------------------------------------------------\n");

    auto dl = [&](const char* name, std::vector<int64_t> shape, std::vector<int64_t> dims) {
        Tensor t({Shape(shape)},TensorOptions().with_dtype(Dtype::Float32));
        fill_f32(t,rng);
        double us;
        if(dims.empty()) us=bench([&](){return reduce_mean(t);});
        else us=bench([&](){return reduce_mean(t,dims);});
        std::string ss="("; for(size_t i=0;i<shape.size();++i){if(i)ss+=",";ss+=std::to_string(shape[i]);}ss+=")";
        std::string ds; if(dims.empty()) ds="None"; else {ds="(";for(size_t i=0;i<dims.size();++i){if(i)ds+=",";ds+=std::to_string(dims[i]);}ds+=")";}
        printf("  %28s %28s %12s %10.0f\n",name,ss.c_str(),ds.c_str(),us);
    };

    dl("LayerNorm-small",{32,768},{-1});
    dl("LayerNorm-large",{32,4096},{-1});
    dl("BatchNorm-2D",{32,256,14,14},{0,2,3});
    dl("BatchNorm-1D",{32,256,100},{0,2});
    dl("Attention-QK",{32,12,128,128},{-1});
    dl("Attention-head-mean",{32,12,128,128},{1});
    dl("Seq-LayerNorm",{32,512,768},{-1});
    dl("Feature-mean",{64,2048},{0});
    dl("Spatial-pool",{32,512,7,7},{2,3});
    dl("Global-avg-pool",{32,2048,7,7},{2,3});
    dl("Loss-reduction",{32,10000},{-1});
    dl("Loss-full",{32,10000},{});
    dl("Token-mean",{32,512,768},{1});
    dl("Channel-mean",{32,256,56,56},{1});
    dl("Embedding-mean",{32,128,300},{-1});
    dl("BERT-pool",{16,512,1024},{1});
    dl("ViT-patch",{32,197,768},{1});
    dl("ResNet-feat",{64,2048},{-1});

    // ============================================================
    header("6. PRECISION STRESS TESTS — fp32");
    // ============================================================
    printf("%25s %10s %12s %20s\n","Dataset","N","rel_err","Note");
    printf("--------------------------------------------------------------------------------\n");

    auto prec = [&](const char* desc, auto fill_fn, int64_t N, const char* note="") {
        Tensor t({Shape({N})},TensorOptions().with_dtype(Dtype::Float32));
        float* d=t.data<float>(); fill_fn(d,N,rng);
        double gt=gt_mean_f32(d,N);
        double val=(double)reduce_mean(t).data<float>()[0];
        printf("  %23s %10lld %12.2e %20s\n",desc,(long long)N,rel_err(val,gt),note);
    };

    for (int64_t N : {1000LL,100000LL,10000000LL}) {
        prec("Uniform[-1,1]",[](float*d,int64_t n,std::mt19937&r){std::uniform_real_distribution<float>u(-1,1);for(int64_t i=0;i<n;++i)d[i]=u(r);},N);
        prec("Gaussian N(0,1)",[](float*d,int64_t n,std::mt19937&r){std::normal_distribution<float>u(0,1);for(int64_t i=0;i<n;++i)d[i]=u(r);},N);
        prec("Large mean+tiny var",[](float*d,int64_t n,std::mt19937&r){std::normal_distribution<float>u(0,1e-3f);for(int64_t i=0;i<n;++i)d[i]=1e6f+u(r);},N,"catastrophic?");
        prec("Mixed [1e-6,1e6]",[](float*d,int64_t n,std::mt19937&r){std::uniform_real_distribution<float>u(1e-6f,1e6f);for(int64_t i=0;i<n;++i)d[i]=u(r);},N);
        prec("Near FLT_MAX",[](float*d,int64_t n,std::mt19937&r){std::uniform_real_distribution<float>u(1e37f,3e38f);for(int64_t i=0;i<n;++i)d[i]=u(r);},N,"overflow?");
        prec("Subnormals",[](float*d,int64_t n,std::mt19937&r){std::uniform_real_distribution<float>u(1e-45f,1e-38f);for(int64_t i=0;i<n;++i)d[i]=u(r);},N,"denorm trap?");
        prec("Alternating ±1e30",[](float*d,int64_t n,std::mt19937&){for(int64_t i=0;i<n;++i)d[i]=(i%2==0)?1e30f:-1e30f;},N,"cancellation?");
        prec("One outlier",[](float*d,int64_t n,std::mt19937&){d[0]=1e10f;for(int64_t i=1;i<n;++i)d[i]=1.0f;},N,"outlier?");
        printf("\n");
    }

    // ============================================================
    header("7. EDGE CASES");
    // ============================================================
    { Tensor t({Shape({1})},TensorOptions().with_dtype(Dtype::Float32));
      t.data<float>()[0]=42.0f;
      printf("  Single element: mean([42.0]) = %f\n",(double)reduce_mean(t).data<float>()[0]); }
    { Tensor t({Shape({10000})},TensorOptions().with_dtype(Dtype::Float32));
      for(int i=0;i<10000;++i) t.data<float>()[i]=3.14f;
      printf("  All same: mean([3.14]*10K) = %.10f\n",(double)reduce_mean(t).data<float>()[0]); }
    { Tensor t({Shape({10000})},TensorOptions().with_dtype(Dtype::Float32));
      for(int i=0;i<10000;++i) t.data<float>()[i]=0.0f;
      printf("  All zeros: mean([0]*10K) = %f\n",(double)reduce_mean(t).data<float>()[0]); }
    { Tensor t({Shape({100})},TensorOptions().with_dtype(Dtype::Float32));
      for(int i=0;i<100;++i) t.data<float>()[i]=std::nanf("");
      printf("  All NaN: mean([NaN]*100) = %f\n",(double)reduce_mean(t).data<float>()[0]); }
    { Tensor t({Shape({100000000LL})},TensorOptions().with_dtype(Dtype::Float32));
      for(int64_t i=0;i<100000000LL;++i) t.data<float>()[i]=1.0f;
      printf("  ones(100M): mean = %.10f (expected 1.0)\n",(double)reduce_mean(t).data<float>()[0]);
      t.data<float>()[0]=1e8f;
      printf("  ones(100M)+[1e8]: mean = %.10f (expected ~2.0)\n",(double)reduce_mean(t).data<float>()[0]); }

    printf("\n%.*s\nBENCHMARK COMPLETE\n%.*s\n",W,"========================================================================================================================================================================",W,"========================================================================================================================================================================");
    return 0;
}
