CUDA_VISIBLE_DEVICES=6 make run-snippet FILE=gpt2_attn_navin.cpp
--- BLAS backend: cublas ---
--- Compiling snippet: gpt2_attn_navin.cpp ---
g++ -Iinclude -I/usr/local/cuda-13.0/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include -I/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nvtx/include -DWITH_CUDA -std=c++2a -fPIC -Wall -Wextra -O3 -fopenmp -mavx2 -mavx512f -mfma -mf16c  -o snippet_runner gpt2_attn_navin.cpp -L/usr/local/cuda-13.0/lib64 -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Llib -Xlinker -rpath -Xlinker '$ORIGIN/lib' -ltensor -lcudart -lcuda -ltbb -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml
In file included from gpt2_attn_navin.cpp:25:
include/checkpointing/Checkpointing.h: In static member function ‘static void OwnTensor::CheckpointManager::signal_handler(int)’:
include/checkpointing/Checkpointing.h:513:36: warning: unused parameter ‘sig’ [-Wunused-parameter]
  513 |     static void signal_handler(int sig) {
      |                                ~~~~^~~
gpt2_attn_navin.cpp: In function ‘int main()’:
gpt2_attn_navin.cpp:424:19: warning: unused variable ‘TOK_GEN_FREQ’ [-Wunused-variable]
  424 |         const int TOK_GEN_FREQ = 1000;
      |                   ^~~~~~~~~~~~
In file included from gpt2_attn_navin.cpp:29:
DataLoader.h: At global scope:
DataLoader.h:26:12: warning: ‘int getenv_int(const char*, int)’ defined but not used [-Wunused-function]
   26 | static int getenv_int(const char* key, int def) {
      |            ^~~~~~~~~~

--- Running snippet_runner ---
./snippet_runner
=== GPT-2 Training Script (Fixed Attention) ===
Configuration:
  vocab_size: 50304
  context_length: 1024
  n_embd: 768
  n_heads: 12
  n_layers: 12
  head_dim: 64
  B=16, T=1024
  GLOBAL_BATCH: 524288
  GRAD_ACCUM_STEPS: 32
  Weight Tying: ENABLED

Initializing model on CUDA device 0...
Number of parameters: 124475904
found 15 shards for split train
found 1 shards for split val

Starting training...
validation loss: 11.0437
step     0 | loss: 11.041492 | lr 6.0000e-04 | norm: 15.6641 | dt: 14386.42ms | tok/sec: 36443.26
step     1 | loss: 10.007134 | lr 6.0000e-04 | norm: 3.9044 | dt: 9096.09ms | tok/sec: 57638.81
step     2 | loss: 9.625859 | lr 5.8372e-04 | norm: 2.0835 | dt: 9239.36ms | tok/sec: 56745.04
step     3 | loss: 9.404709 | lr 5.3683e-04 | norm: 1.9586 | dt: 9369.22ms | tok/sec: 55958.54
step     4 | loss: 9.237324 | lr 4.6500e-04 | norm: 1.8173 | dt: 9493.14ms | tok/sec: 55228.06
step     5 | loss: 9.091610 | lr 3.7689e-04 | norm: 1.7000 | dt: 9592.72ms | tok/sec: 54654.76
step     6 | loss: 8.964387 | lr 2.8311e-04 | norm: 1.5786 | dt: 9649.67ms | tok/sec: 54332.23
step     7 | loss: 8.883045 | lr 1.9500e-04 | norm: 1.4531 | dt: 9701.52ms | tok/sec: 54041.86
[CheckpointManager] Saved: checkpoints/gpt2_step_8.ckpt (safely)
step     8 | loss: 8.794813 | lr 1.2317e-04 | norm: 1.4109 | dt: 17947.18ms | tok/sec: 29212.83
validation loss: 8.7633
step     9 | loss: 8.758547 | lr 7.6283e-05 | norm: 1.3769 | dt: 11617.43ms | tok/sec: 45129.42

=== Training Complete ===

blubridge@blubridge:/mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1$ rm checkpoints/gpt2_step_8.ckpt
blubridge@blubridge:/mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1$ CUDA_VISIBLE_DEVICES=6 nsys profile --stats=true ./snippet_runner
WARNING: The version of the system or its configuration does not allow enabling CPU profiling:

- CPU IP/backtrace sampling will be disabled.
- CPU context switch tracing will be disabled.
Try the 'nsys status --environment' command to learn more.

Collecting data...
=== GPT-2 Training Script (Fixed Attention) ===
Configuration:
  vocab_size: 50304
  context_length: 1024
  n_embd: 768
  n_heads: 12
  n_layers: 12
  head_dim: 64
  B=16, T=1024
  GLOBAL_BATCH: 524288
  GRAD_ACCUM_STEPS: 32
  Weight Tying: ENABLED

Initializing model on CUDA device 0...
Number of parameters: 124475904
found 15 shards for split train
found 1 shards for split val

Starting training...
validation loss: 11.0437
step     0 | loss: 11.041492 | lr 6.0000e-04 | norm: 15.6641 | dt: 15376.22ms | tok/sec: 34097.33
step     1 | loss: 10.007134 | lr 6.0000e-04 | norm: 3.9044 | dt: 9295.28ms | tok/sec: 56403.65
step     2 | loss: 9.625859 | lr 5.8372e-04 | norm: 2.0835 | dt: 9463.17ms | tok/sec: 55403.00
step     3 | loss: 9.404708 | lr 5.3683e-04 | norm: 1.9586 | dt: 9580.33ms | tok/sec: 54725.44
step     4 | loss: 9.237324 | lr 4.6500e-04 | norm: 1.8173 | dt: 9692.44ms | tok/sec: 54092.44
step     5 | loss: 9.091610 | lr 3.7689e-04 | norm: 1.7000 | dt: 9766.04ms | tok/sec: 53684.81
step     6 | loss: 8.964387 | lr 2.8311e-04 | norm: 1.5786 | dt: 9832.74ms | tok/sec: 53320.65
step     7 | loss: 8.883045 | lr 1.9500e-04 | norm: 1.4531 | dt: 9873.31ms | tok/sec: 53101.54
[CheckpointManager] Saved: checkpoints/gpt2_step_8.ckpt (safely)
step     8 | loss: 8.794812 | lr 1.2317e-04 | norm: 1.4109 | dt: 21710.73ms | tok/sec: 24148.79
validation loss: 8.7633
step     9 | loss: 8.758546 | lr 7.6283e-05 | norm: 1.3769 | dt: 11629.93ms | tok/sec: 45080.94

=== Training Complete ===
Generating '/tmp/nsys-report-676d.qdstrm'
[1/8] [========================100%] report31.nsys-rep
[2/8] [========================100%] report31.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1/report31.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)      Max (ns)       StdDev (ns)             Name
 --------  ---------------  ---------  -------------  -------------  -----------  -------------  ---------------  ----------------------
     49.0  248,255,045,246      2,463  100,793,765.8  100,217,058.0       10,570  2,265,343,244     44,068,883.0  poll                  
     48.1  243,584,121,769        277  879,365,060.5  500,142,439.0  500,041,879  4,001,565,540  1,089,660,363.7  pthread_cond_timedwait
      1.5    7,797,672,602        445   17,522,859.8       58,290.0        1,900  2,371,800,998    121,439,050.0  writev                
      1.3    6,785,909,611      2,958    2,294,087.1    1,061,917.5        1,151    340,917,457      7,566,725.1  ioctl                 
      0.0      219,142,028        134    1,635,388.3      164,999.5        6,770     70,194,631      6,146,583.8  mmap                  
      0.0       12,233,431         54      226,545.0       64,534.5       21,710      6,319,416        853,819.9  mmap64                
      0.0       10,860,411         57      190,533.5      103,800.0        1,260      2,466,205        339,908.0  pthread_mutex_lock    
      0.0       10,431,685         18      579,538.1      434,279.0       12,450      1,595,696        595,376.7  pthread_rwlock_wrlock 
      0.0        8,424,103         71      118,649.3       12,650.0        1,050      1,745,287        327,200.3  close                 
      0.0        6,978,385          4    1,744,596.3    1,503,862.0      568,069      3,402,592      1,209,754.1  pthread_create        
      0.0        5,279,731         20      263,986.5       57,560.0       22,370      2,389,015        564,126.5  pthread_rwlock_rdlock 
      0.0        4,853,639         86       56,437.7       32,685.0        8,631        602,408         84,210.2  open64                
      0.0        4,264,770         37      115,264.1       41,830.0        9,650      2,473,755        401,661.9  munmap                
      0.0        3,864,269         56       69,004.8       32,170.0        2,300        455,619         87,358.9  fopen                 
      0.0        2,208,135         10      220,813.5       62,580.0        2,010      1,568,767        475,864.3  sem_timedwait         
      0.0        2,031,007         22       92,318.5       32,965.0        1,000        842,868        189,387.0  putc                  
      0.0        1,475,559         23       64,154.7       18,710.0        1,170        534,738        112,215.4  write                 
      0.0          637,789          2      318,894.5      318,894.5      213,430        424,359        149,149.3  fopen64               
      0.0          607,802         52       11,688.5        6,960.0        2,260         74,060         12,624.9  fclose                
      0.0          576,517         29       19,879.9        5,690.0        1,029        203,010         44,278.2  fwrite                
      0.0          256,389         10       25,638.9       25,130.0        7,080         50,330         14,096.6  open                  
      0.0          172,770          1      172,770.0      172,770.0      172,770        172,770              0.0  fgets                 
      0.0          157,980          2       78,990.0       78,990.0       72,770         85,210          8,796.4  fread                 
      0.0          153,329         82        1,869.9        1,429.5        1,000          9,770          1,254.9  fileno                
      0.0          137,408          4       34,352.0       35,879.5       11,220         54,429         18,505.2  pipe2                 
      0.0           76,430          2       38,215.0       38,215.0       33,330         43,100          6,908.4  socket                
      0.0           70,470         14        5,033.6        4,234.5        1,490          9,980          3,111.2  read                  
      0.0           64,238         14        4,588.4        3,159.5        1,240         13,400          3,527.5  fcntl                 
      0.0           44,480          1       44,480.0       44,480.0       44,480         44,480              0.0  connect               
      0.0           34,860          3       11,620.0        9,270.0        4,010         21,580          9,017.7  stat                  
      0.0           33,370         15        2,224.7        1,870.0        1,050          4,811          1,132.6  fflush                
      0.0           11,440          3        3,813.3        3,260.0        2,570          5,610          1,593.7  fstat                 
      0.0            8,690          1        8,690.0        8,690.0        8,690          8,690              0.0  bind                  
      0.0            8,330          3        2,776.7        2,700.0        2,300          3,330            519.3  pthread_cond_broadcast
      0.0            6,309          4        1,577.3        1,550.0        1,029          2,180            475.5  dup                   
      0.0            2,320          1        2,320.0        2,320.0        2,320          2,320              0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)     Min (ns)    Max (ns)     StdDev (ns)                    Name
 --------  ---------------  ---------  -------------  -------------  ---------  -----------  -------------  ---------------------------------------
     43.9   43,255,787,191    149,272      289,778.3       29,609.5      3,230  365,014,385    1,434,320.5  cudaLaunchKernel                       
     26.5   26,130,074,665     36,845      709,189.2       50,840.0      2,170   18,594,881    2,173,731.2  cudaMemsetAsync                        
     13.7   13,537,897,422     49,000      276,283.6       23,305.0      3,120   19,805,658    1,396,814.2  cuLaunchKernel                         
      5.5    5,410,434,413      1,741    3,107,659.1       54,720.0      4,320  560,320,341   21,560,461.1  cudaMemcpyAsync                        
      3.5    3,425,128,433         10  342,512,843.3  383,620,551.5  1,159,307  395,643,270  121,032,052.5  cudaMemcpy                             
      3.2    3,143,067,654        345    9,110,341.0       40,489.0      1,160  567,173,907   49,607,720.7  cudaMallocAsync_v11020                 
      1.7    1,689,565,309        157   10,761,562.5       64,710.0      2,589  355,678,465   31,128,367.3  cudaHostAlloc                          
      0.9      930,123,188         17   54,713,128.7    4,071,342.0    406,189  693,735,958  167,299,222.3  cuLibraryLoadData                      
      0.5      506,786,071        152    3,334,118.9       37,219.0      4,769   91,199,086    9,289,430.9  cudaFreeHost                           
      0.1      139,992,221     49,000        2,857.0        1,100.0        270   25,344,016      115,516.1  cuKernelGetFunction                    
      0.1      129,274,175     11,840       10,918.4        5,700.0        820      916,118       20,450.5  cudaEventRecord                        
      0.1      112,913,045    149,272          756.4          280.0        100      545,969        3,394.8  cuKernelGetName                        
      0.1       81,006,375         16    5,062,898.4       49,100.0      2,569   43,810,997   11,008,864.5  cudaMalloc                             
      0.0       39,318,183     59,200          664.2          230.0         89      225,960        2,536.0  cuStreamGetCaptureInfo_v2              
      0.0       20,484,157      8,640        2,370.9          980.0        300      404,899        7,264.0  cudaStreamSetAttribute_v11000          
      0.0       10,692,556         12      891,046.3        3,659.5        440    5,498,299    2,042,061.5  cudaFree                               
      0.0        2,212,284         13      170,175.7      169,229.0     86,400      338,419       72,241.1  cuMemsetD32Async                       
      0.0        1,461,107          1    1,461,107.0    1,461,107.0  1,461,107    1,461,107            0.0  cudaEventQuery                         
      0.0        1,350,320      1,320        1,023.0          669.0        110       67,120        2,249.6  cuGetProcAddress_v2                    
      0.0          315,752         35        9,021.5        3,580.0        570       46,991       11,766.2  cuLibraryGetKernel                     
      0.0          268,640          1      268,640.0      268,640.0    268,640      268,640            0.0  cudaStreamDestroy                      
      0.0          230,740         36        6,409.4          650.0        300       69,330       14,158.7  cudaEventCreateWithFlags               
      0.0          230,379          1      230,379.0      230,379.0    230,379      230,379            0.0  cudaStreamCreateWithFlags              
      0.0          181,999          2       90,999.5       90,999.5     47,160      134,839       61,998.4  cudaDeviceSynchronize                  
      0.0           74,280          5       14,856.0       17,470.0      1,930       25,670        9,462.4  cuInit                                 
      0.0           60,150          1       60,150.0       60,150.0     60,150       60,150            0.0  cudaStreamSynchronize                  
      0.0           30,660          4        7,665.0        3,115.0      1,080       23,350       10,536.5  cuModuleGetLoadingMode                 
      0.0           12,000          3        4,000.0        2,460.0      2,090        7,450        2,993.5  cudaGetDriverEntryPointByVersion_v12050
      0.0            2,130          2        1,065.0        1,065.0      1,030        1,100           49.5  cudaGetDeviceProperties_v12000         

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)                                                  Name
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------------------------------------------------------------------------------------
     16.9   16,791,869,021      3,840   4,372,882.6   4,239,647.0   2,785,910   5,811,201    416,780.9  void OwnTensor::mem_efficient_bwd_unified_kernel_exp12<(int)64, (bool)1>(OwnTensor::MemEfficientBwd…
     13.5   13,411,603,106      8,000   1,676,450.4   1,029,334.0     700,845  18,692,994  3,050,301.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nt_align4>(T1::Params)             
     12.8   12,740,445,110     17,280     737,294.3     851,089.5     213,285   1,253,436    289,519.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(T1::Params)              
      9.2    9,174,093,625      4,200   2,184,308.0     989,494.0     713,262  15,514,171  3,836,272.3  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_tn_align4>(T1::Params)             
      8.7    8,609,890,359      4,320   1,993,030.2   2,001,388.5   1,088,916   2,607,002    128,142.4  void OwnTensor::fused_attn_forward_kernel_tc_sm89<(int)64, (int)64, (int)64, (int)1>(OwnTensor::Mem…
      5.6    5,561,711,766        320  17,380,349.3  17,254,389.5  11,571,966  20,028,702  1,365,198.9  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(T1::Params)             
      4.6    4,579,036,100      7,680     596,228.7     487,546.5     202,276   1,162,330    292,558.9  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_nt_align4>(T1::Params)             
      4.0    3,936,066,982      7,680     512,508.7     448,889.0     173,123   1,001,879    245,245.0  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_tn_align4>(T1::Params)             
      3.9    3,905,257,915      3,840   1,016,994.2     977,814.0     753,519   1,351,870    102,700.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(T1::Params)             
      2.6    2,616,228,391        320   8,175,713.7   8,172,548.0   8,119,595   8,267,594     24,694.5  void OwnTensor::cuda::sparseCENormalize_from_stats<float, unsigned short>(const T1 *, const T2 *, T…
      2.6    2,614,931,653      3,840     680,971.8     680,910.0     670,381     693,168      3,153.2  void OwnTensor::cuda::gelu_backward_sm89_kernel<float>(const T1 *, const T1 *, T1 *, long)          
      2.1    2,087,341,385      4,320     483,180.9     483,081.0     476,043     492,651      2,237.0  void OwnTensor::cuda::gelu_forward_sm89_kernel<float>(const T1 *, T1 *, long)                       
      2.0    2,010,136,990     55,221      36,401.7       3,233.0       1,888     845,683     72,749.0  void OwnTensor::cuda::vectorized_kernel_impl<(int)4, OwnTensor::Add_inplace_Functor, float, float, …
      1.8    1,804,087,924     15,840     113,894.4     119,363.0      91,681     482,282     10,981.7  void OwnTensor::generic_strided_copy_kernel<(int)10>(const unsigned char *, unsigned char *, long, …
      1.5    1,481,250,313      3,840     385,742.3     385,576.0     376,040     397,384      3,253.5  void OwnTensor::cat_batched_kernel<unsigned int>(T1 *, long, long, OwnTensor::CatBatchMeta)         
      1.4    1,366,668,138      8,960     152,529.9     157,060.0       1,216     171,684     29,149.3  void OwnTensor::cuda::vectorized_kernel_impl<(int)4, OwnTensor::AddFunctor, float, float, float>(in…
      1.4    1,353,662,291        360   3,760,173.0   3,760,116.0   3,756,455   3,765,009      1,224.0  void OwnTensor::cuda::sparse_ce_forward_kernel_vec_save_stats<float, unsigned short>(const T1 *, co…
      1.2    1,157,364,013      9,000     128,596.0     129,154.0      78,466     160,100      6,086.4  void OwnTensor::cuda::layer_norm_forward_sm89_kernel<float, float>(const T1 *, const T1 *, const T1…
      1.1    1,132,798,521      8,000     141,599.8     138,947.5     127,714     175,524      8,170.4  void OwnTensor::cuda::ln_backward_input_sm89_kernel<float>(const T1 *, const T1 *, const float *, c…
      1.1    1,119,772,391     15,680      71,414.1      38,753.0      19,521     167,108     46,677.4  void OwnTensor::cuda::unified_reduce_kernel<float, float, OwnTensor::detail::SumOp<float>, unsigned…
      1.0      996,967,566      8,000     124,620.9     123,171.0     119,778     432,298      9,010.4  void OwnTensor::cuda::ln_backward_gamma_beta_sm89_kernel<float>(const T1 *, const T1 *, const float…
      0.6      622,340,622      3,840     162,067.9     162,020.0     160,291     165,027        526.8  void OwnTensor::mem_efficient_bwd_precompute_D_sm89<(int)64>(OwnTensor::MemEfficientBwdParams)      
      0.0       44,890,160        130     345,308.9     368,440.5      59,777     387,176     82,408.8  OwnTensor::cuda::multi_tensor_adam_sm89_kernel(OwnTensor::cuda::AdamLaunchMetadata, float, float, f…
      0.0       40,331,752        640      63,018.4      60,065.5       7,713     121,635     54,855.5  void OwnTensor::cuda::embedding_backward_kernel_optimized<float>(const unsigned short *, const T1 *…
      0.0       34,429,931        360      95,638.7      95,186.0      68,993     117,699      5,134.2  void OwnTensor::add_kernel_nd_broadcast<float>(const T1 *, const T1 *, T1 *, OwnTensor::SimplifiedB…
      0.0       29,340,694        720      40,751.0      18,928.5       3,392      85,986     31,514.6  OwnTensor::cuda::embedding_forward_kernel_vectorized(const unsigned short *, const float *, float *…
      0.0       12,183,433        130      93,718.7     106,322.0      23,521     111,683     26,476.9  OwnTensor::cuda::multi_tensor_scale_sm89_kernel(OwnTensor::cuda::ScaleLaunchMetadata, const float *)
      0.0       11,539,964         10   1,153,996.4   1,166,857.5   1,098,358   1,190,714     29,756.6  OwnTensor::cuda::multi_tensor_grad_norm_kernel(const OwnTensor::cuda::TensorInfo *, const long *, i…
      0.0        5,053,744        117      43,194.4      47,169.0       4,800      50,977     11,723.0  OwnTensor::cuda::multi_tensor_zero_sm89_kernel(OwnTensor::cuda::ZeroLaunchMetadata)                 
      0.0        4,292,863         10     429,286.3     429,817.5     425,257     432,521      2,306.7  void OwnTensor::transpose_2d_tiled_kernel<float, (int)32, (int)8>(const T1 *, T1 *, int, int)       
      0.0        1,960,324        720       2,722.7       2,752.0       1,600       3,393        259.7  void OwnTensor::cuda::sum_reduction_kernel<float>(const T1 *, T1 *, long)                           
      0.0        1,146,302        680       1,685.7       1,664.0       1,088       2,112        155.6  void OwnTensor::convert_type_kernel<long, unsigned short>(const T1 *, T2 *, long)                   
      0.0          827,224        360       2,297.8       2,240.0       1,536       2,656        144.8  void OwnTensor::scalar_div_copy<float, float>(const T1 *, double, T2 *, unsigned long)              
      0.0          462,568          2     231,284.0     231,284.0      10,656     451,912    312,015.1  void gen_sequenced<curandStateXORWOW, float2, normal_args_st, &curand_normal_scaled2<curandStateXOR…
      0.0          235,525          2     117,762.5     117,762.5      94,786     140,739     32,493.7  void generate_seed_pseudo<rng_config<curandStateXORWOW, (curandOrdering)101>>(unsigned long long, u…
      0.0           19,777         10       1,977.7       1,984.0       1,792       2,112         78.1  OwnTensor::cuda::compute_clip_coef_kernel(float *, float *, float, bool)                            

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count    Avg (ns)    Med (ns)  Min (ns)   Max (ns)    StdDev (ns)             Operation
 --------  ---------------  ------  -----------  --------  --------  -----------  ------------  ------------------------------
     72.7    1,445,230,261     504  2,867,520.4   1,856.0       512  554,021,857  31,790,363.6  [CUDA memcpy Device-to-Host]  
     16.6      330,945,088  36,858      8,978.9   1,600.0       352      179,075      18,658.7  [CUDA memset]                 
     10.7      211,949,386     887    238,950.8   3,776.0       448   10,519,783   1,153,119.1  [CUDA memcpy Host-to-Device]  
      0.0          623,823     360      1,732.8   1,728.0     1,024        2,176         104.2  [CUDA memcpy Device-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)   Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
 -----------  ------  --------  --------  --------  --------  -----------  ------------------------------
 439,443.556  36,858    11.923     0.003     0.000   154.534       24.503  [CUDA memset]
   1,493.711     504     2.964     0.003     0.000   154.534       12.265  [CUDA memcpy Device-to-Host]  
     363.861     887     0.410     0.033     0.001     9.437        1.732  [CUDA memcpy Host-to-Device]  
       2.949     360     0.008     0.008     0.008     0.008        0.000  [CUDA memcpy Device-to-Device]

Generated:
        /mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1/report31.nsys-rep
        /mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1/report31.sqlite
blubridge@blubridge:/mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1$
