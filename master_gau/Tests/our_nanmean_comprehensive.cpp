// Comprehensive NanMean benchmark — Our Library
#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>
#include <algorithm>
#include <omp.h>
#include "core/Tensor.h"
#include "ops/UnaryOps/Reduction.h"
using namespace OwnTensor;

template<typename F>
double bench(F fn, int w=10, int it=50) {
    for(int i=0;i<w;++i)fn();
    std::vector<double>t;
    for(int i=0;i<it;++i){auto s=std::chrono::high_resolution_clock::now();fn();
    auto e=std::chrono::high_resolution_clock::now();
    t.push_back(std::chrono::duration<double,std::micro>(e-s).count());}
    std::sort(t.begin(),t.end());return t[it/2];
}
void fill_nan_f32(Tensor&t,double pct,std::mt19937&r){
    float*d=t.data<float>();std::normal_distribution<float>n(0,1);
    std::uniform_real_distribution<float>u(0,1);
    for(size_t i=0;i<t.numel();++i){d[i]=n(r);if(u(r)<pct)d[i]=std::nanf("");}
}
void fill_nan_f64(Tensor&t,double pct,std::mt19937&r){
    double*d=t.data<double>();std::normal_distribution<double>n(0,1);
    std::uniform_real_distribution<double>u(0,1);
    for(size_t i=0;i<t.numel();++i){d[i]=n(r);if(u(r)<pct)d[i]=std::nan("");}
}

int main(){
    printf("OUR NANMEAN — ALL PATHS × fp32 + fp64 (28 threads)\n\n");
    std::mt19937 rng(42);
    double NP=0.1;

    printf("%20s %22s %5s %18s | %8s %8s\n","Label","Shape","Dim","Path","fp32","fp64");
    printf("%s\n",std::string(95,'-').c_str());

    auto run=[&](const char*lbl,std::vector<int64_t>sh,std::vector<int64_t>dm,const char*path){
        printf("%20s %22s",lbl,("("+[&]{std::string s;for(size_t i=0;i<sh.size();++i){if(i)s+=",";s+=std::to_string(sh[i]);}return s;}()+")").c_str());
        std::string ds;for(size_t i=0;i<dm.size();++i){if(i)ds+=",";ds+=std::to_string(dm[i]);}
        printf(" %5s %18s |",dm.empty()?"all":ds.c_str(),path);
        // fp32
        {Tensor t({Shape(sh)},TensorOptions().with_dtype(Dtype::Float32));
         fill_nan_f32(t,NP,rng);
         double us=bench([&](){return reduce_nanmean(t,dm);});
         printf(" %7.0fμ",us);}
        // fp64
        {Tensor t({Shape(sh)},TensorOptions().with_dtype(Dtype::Float64));
         fill_nan_f64(t,NP,rng);
         double us=bench([&](){return reduce_nanmean(t,dm);});
         printf(" %7.0fμ",us);}
        printf("\n");
    };

    // Inner
    run("Tiny",{8,16},{-1},"Inner");
    run("LayerNorm",{32,768},{-1},"Inner");
    run("Batch-feat",{256,4096},{-1},"Inner");
    run("Seq-LN",{4096,768},{-1},"Inner");
    run("Many-out",{49152,128},{-1},"Inner");
    run("Medium",{1000,10000},{-1},"Inner");
    run("Spatial",{2048,50176},{-1},"Inner");
    run("Wide",{10,1000000},{-1},"Inner");
    run("Tall",{1000000,10},{-1},"Inner");
    printf("\n");

    // Outer
    run("Outer-med",{1000,256},{0},"Outer");
    run("Outer-tall",{10000,100},{0},"Outer");
    run("Outer-wide",{100,10000},{0},"Outer");
    printf("\n");

    // Middle (coalesced)
    run("Token-mean",{32,512,768},{1},"Middle->Coalesced");
    run("Channel-mean",{32,256,56,56},{1},"Middle->Coalesced");
    run("BERT-pool",{16,512,1024},{1},"Middle->Coalesced");
    run("ViT-patch",{32,197,768},{1},"Middle->Coalesced");
    run("Attn-head",{32,12,128,128},{1},"Middle->Coalesced");
    printf("\n");

    // Generic (non-consecutive)
    run("3D XZ",{100,200,50},{0,2},"Generic");
    run("Transformer",{32,128,768},{0,2},"Generic");
    run("4D XH",{16,64,32,32},{0,2},"Generic");
    printf("\n");

    // Full reduction
    run("Full",{1000},{},"Full-red");
    run("Full",{10000},{},"Full-red");
    run("Full",{100000},{},"Full-red");
    run("Full",{1000000},{},"Full-red");
    run("Full",{10000000},{},"Full-red");
    run("Full",{50000000},{},"Full-red");
    printf("\n");

    // Varying NaN%
    printf("--- Varying NaN%% (1000, 10000) fp32 ---\n");
    for(double pct:{0.0,0.01,0.1,0.5,0.9,0.99}){
        Tensor t({Shape({1000,10000})},TensorOptions().with_dtype(Dtype::Float32));
        fill_nan_f32(t,pct,rng);
        double us=bench([&](){return reduce_nanmean(t,{-1});});
        printf("  %5.1f%% NaN: %8.0f μs\n",pct*100,us);
    }

    printf("\nBENCHMARK COMPLETE\n");
    return 0;
}
