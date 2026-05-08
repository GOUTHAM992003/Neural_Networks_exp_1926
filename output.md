CUDA_VISIBLE_DEVICES=6 USE_PACKED_SDPA=1 make run-snippet FILE=gpt2_attn_navin.cpp
--- BLAS backend: cublas ---
--- Compiling snippet: gpt2_attn_navin.cpp ---
g++ -Iinclude -I/usr/local/cuda-13.0/include -I/usr/local/cuda-13.0/targets/x86_64-linux/include -I/opt/nvidia/nsight-systems/2025.6.3/target-linux-x64/nvtx/include -DWITH_CUDA -std=c++2a -fPIC -Wall -Wextra -O3 -fopenmp -mavx2 -mavx512f -mfma -mf16c  -o snippet_runner gpt2_attn_navin.cpp -L/usr/local/cuda-13.0/lib64 -L/usr/local/cuda-13.0/targets/x86_64-linux/lib -Llib -Xlinker -rpath -Xlinker '$ORIGIN/lib' -ltensor -lcudart -lcuda -ltbb -lcurand -lcublas -lcublasLt -lgomp -lnvidia-ml
In file included from gpt2_attn_navin.cpp:26:
include/checkpointing/Checkpointing.h: In static member function ‘static void OwnTensor::CheckpointManager::signal_handler(int)’:
include/checkpointing/Checkpointing.h:513:36: warning: unused parameter ‘sig’ [-Wunused-parameter]
  513 |     static void signal_handler(int sig) {
      |                                ~~~~^~~
gpt2_attn_navin.cpp: In function ‘int main()’:
gpt2_attn_navin.cpp:435:19: warning: unused variable ‘TOK_GEN_FREQ’ [-Wunused-variable]
  435 |         const int TOK_GEN_FREQ = 1000;
      |                   ^~~~~~~~~~~~
In file included from gpt2_attn_navin.cpp:30:
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
step     0 | loss: 11.041492 | lr 6.0000e-04 | norm: 15.6641 | dt: 13004.28ms | tok/sec: 40316.58
step     1 | loss: 10.007135 | lr 6.0000e-04 | norm: 3.9044 | dt: 9187.98ms | tok/sec: 57062.38
step     2 | loss: 9.625859 | lr 5.8372e-04 | norm: 2.0835 | dt: 9314.19ms | tok/sec: 56289.19
step     3 | loss: 9.404708 | lr 5.3683e-04 | norm: 1.9586 | dt: 9440.77ms | tok/sec: 55534.46
step     4 | loss: 9.237324 | lr 4.6500e-04 | norm: 1.8173 | dt: 9528.55ms | tok/sec: 55022.86
step     5 | loss: 9.091610 | lr 3.7689e-04 | norm: 1.7000 | dt: 9586.76ms | tok/sec: 54688.73
step     6 | loss: 8.964387 | lr 2.8311e-04 | norm: 1.5786 | dt: 9639.72ms | tok/sec: 54388.30
step     7 | loss: 8.883045 | lr 1.9500e-04 | norm: 1.4531 | dt: 9693.99ms | tok/sec: 54083.80
[CheckpointManager] Saved: checkpoints/gpt2_step_8.ckpt (safely)
step     8 | loss: 8.794812 | lr 1.2317e-04 | norm: 1.4109 | dt: 13020.29ms | tok/sec: 40267.01
validation loss: 8.7634
step     9 | loss: 8.758547 | lr 7.6283e-05 | norm: 1.3769 | dt: 11694.33ms | tok/sec: 44832.67

=== Training Complete ===

blubridge@blubridge:/mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1$ CUDA_VISIBLE_DEVICES=6 USE_PACKED_SDPA=1 nsys profile --stats=true ./snippet_runner
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
step     0 | loss: 11.041492 | lr 6.0000e-04 | norm: 15.6641 | dt: 13745.67ms | tok/sec: 38142.05
step     1 | loss: 10.007134 | lr 6.0000e-04 | norm: 3.9044 | dt: 9328.52ms | tok/sec: 56202.70
step     2 | loss: 9.625859 | lr 5.8372e-04 | norm: 2.0835 | dt: 9468.61ms | tok/sec: 55371.15
step     3 | loss: 9.404708 | lr 5.3683e-04 | norm: 1.9586 | dt: 9562.58ms | tok/sec: 54827.03
step     4 | loss: 9.237324 | lr 4.6500e-04 | norm: 1.8173 | dt: 9623.44ms | tok/sec: 54480.30
step     5 | loss: 9.091610 | lr 3.7689e-04 | norm: 1.7000 | dt: 9693.32ms | tok/sec: 54087.58
step     6 | loss: 8.964387 | lr 2.8311e-04 | norm: 1.5786 | dt: 9745.28ms | tok/sec: 53799.17
step     7 | loss: 8.883044 | lr 1.9500e-04 | norm: 1.4531 | dt: 9789.72ms | tok/sec: 53554.96
[CheckpointManager] Saved: checkpoints/gpt2_step_8.ckpt (safely)
step     8 | loss: 8.794813 | lr 1.2317e-04 | norm: 1.4109 | dt: 12841.46ms | tok/sec: 40827.77
validation loss: 8.7633
step     9 | loss: 8.758547 | lr 7.6283e-05 | norm: 1.3769 | dt: 11813.16ms | tok/sec: 44381.71

=== Training Complete ===
Generating '/tmp/nsys-report-0ba0.qdstrm'
[1/8] [========================100%] report42.nsys-rep
[2/8] [========================100%] report42.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1/report42.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)      Max (ns)       StdDev (ns)             Name
 --------  ---------------  ---------  -------------  -------------  -----------  -------------  ---------------  ----------------------
     49.4  225,464,027,813      2,250  100,206,234.6  100,121,275.5        3,340  1,395,752,896     28,078,414.8  poll                  
     49.1  224,016,821,774        252  888,955,642.0  500,065,337.0  500,024,552  4,000,142,042  1,102,133,665.2  pthread_cond_timedwait
      1.0    4,645,411,689      2,921    1,590,349.8      977,764.0        1,380    140,837,054      3,486,235.7  ioctl                 
      0.4    1,658,908,703        445    3,727,884.7       29,489.0        5,449    173,643,317     14,079,810.4  writev                
      0.0      146,895,320        134    1,096,233.7       34,085.0        5,979     47,428,053      4,170,817.4  mmap                  
      0.0       11,587,240         74      156,584.3        3,909.5        1,060      3,784,768        558,195.7  close                 
      0.0        6,302,194         54      116,707.3       32,339.5       12,410      3,625,870        490,465.5  mmap64                
      0.0        6,032,311         21      287,252.9       43,019.0       24,680      1,028,154        352,849.6  pthread_rwlock_rdlock 
      0.0        1,756,772         36       48,799.2       15,079.5        7,400      1,002,824        165,372.4  munmap                
      0.0        1,467,257         17       86,309.2       70,299.0       12,460        201,517         65,846.0  pthread_rwlock_wrlock 
      0.0        1,368,947         86       15,918.0       15,120.0        7,390         32,359          5,272.7  open64                
      0.0        1,148,119         10      114,811.9       44,749.0       29,889        751,958        224,137.1  sem_timedwait         
      0.0        1,020,253          4      255,063.3      256,600.5      241,296        265,756         11,370.5  pthread_create        
      0.0          889,508         56       15,884.1       11,810.0        2,540         47,779         12,411.6  fopen                 
      0.0          693,120         23       30,135.7        7,540.0        1,380         94,149         31,719.1  write                 
      0.0          482,853          2      241,426.5      241,426.5      135,338        347,515        150,031.8  fopen64               
      0.0          251,756         11       22,886.9       21,599.0        2,651         44,489         13,235.6  pthread_mutex_lock    
      0.0          241,054         29        8,312.2        2,980.0        1,180         20,229          7,346.9  putc                  
      0.0          232,365         52        4,468.6        4,000.0        1,970         14,410          2,126.9  fclose                
      0.0          157,927          1      157,927.0      157,927.0      157,927        157,927              0.0  pthread_cond_wait     
      0.0          139,558          1      139,558.0      139,558.0      139,558        139,558              0.0  fgets                 
      0.0          105,837         10       10,583.7       10,469.5        5,000         16,450          4,196.4  open                  
      0.0           97,787         22        4,444.9        2,869.5        1,250         26,960          5,635.8  fwrite                
      0.0           49,870         15        3,324.7        3,310.0        1,200          6,730          1,754.0  read                  
      0.0           46,520          2       23,260.0       23,260.0       18,600         27,920          6,590.2  fread                 
      0.0           42,249          4       10,562.3       10,449.5        6,660         14,690          3,864.2  pipe2                 
      0.0           37,250          2       18,625.0       18,625.0       14,790         22,460          5,423.5  socket                
      0.0           21,949          1       21,949.0       21,949.0       21,949         21,949              0.0  connect               
      0.0           17,219         11        1,565.4        1,630.0        1,040          2,040            386.6  fcntl                 
      0.0           16,341          3        5,447.0        5,871.0        2,590          7,880          2,670.4  stat                  
      0.0            8,630          4        2,157.5        1,160.0        1,000          5,310          2,104.0  pthread_cond_broadcast
      0.0            8,110          4        2,027.5        2,025.0        1,380          2,680            714.0  fstat                 
      0.0            4,999          4        1,249.8        1,274.5        1,000          1,450            204.9  dup                   
      0.0            3,580          1        3,580.0        3,580.0        3,580          3,580              0.0  bind                  
      0.0            1,759          1        1,759.0        1,759.0        1,759          1,759              0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)     Min (ns)    Max (ns)     StdDev (ns)                    Name
 --------  ---------------  ---------  -------------  -------------  ---------  -----------  -------------  ---------------------------------------
     43.8   43,966,358,401    129,592      339,267.5       62,648.5      5,260  403,442,634    1,413,845.8  cudaLaunchKernel                       
     27.3   27,419,964,863     33,005      830,782.2      121,628.0      3,140   19,140,107    2,357,140.1  cudaMemsetAsync                        
     17.1   17,178,373,486     49,000      350,579.1       10,729.5      4,789   20,251,178    1,474,710.4  cuLaunchKernel                         
      4.1    4,105,240,650      1,381    2,972,658.0       20,080.0      4,990  126,050,196   16,463,021.4  cudaMemcpyAsync                        
      3.8    3,763,821,113         10  376,382,111.3  418,747,248.5  1,130,302  425,376,495  132,003,863.8  cudaMemcpy                             
      2.3    2,279,922,288        332    6,867,235.8       11,280.0      1,720  332,713,902   33,360,168.4  cudaMallocAsync_v11020                 
      0.7      713,083,032        157    4,541,930.1       14,329.0      3,240  154,370,642   13,439,325.9  cudaHostAlloc                          
      0.4      405,698,614         16   25,356,163.4    1,834,259.5    187,307  298,362,955   74,428,873.5  cuLibraryLoadData                      
      0.3      297,986,562        152    1,960,437.9       13,980.0      3,730   58,899,915    5,450,440.5  cudaFreeHost                           
      0.0       45,139,048     49,000          921.2          560.0        449   10,228,873       46,226.0  cuKernelGetFunction                    
      0.0       39,849,700     11,840        3,365.7        2,970.0      1,590      227,536        3,064.1  cudaEventRecord                        
      0.0       34,132,822    129,592          263.4          211.0        180       11,060          180.8  cuKernelGetName                        
      0.0       27,709,326         16    1,731,832.9       13,854.5      3,160    9,406,246    2,783,995.7  cudaMalloc                             
      0.0       16,204,180     59,200          273.7          210.0        169       19,670          215.5  cuStreamGetCaptureInfo_v2              
      0.0        8,096,306      8,640          937.1          690.0        580       15,280          822.3  cudaStreamSetAttribute_v11000          
      0.0        4,885,651         12      407,137.6          985.5        360    2,452,140      937,467.7  cudaFree                               
      0.0          631,979      1,320          478.8          415.5        170       13,279          560.5  cuGetProcAddress_v2                    
      0.0          499,762         13       38,443.2       39,109.0     26,959       42,469        3,893.8  cuMemsetD32Async                       
      0.0          222,927          2      111,463.5      111,463.5     81,899      141,028       41,810.5  cudaDeviceSynchronize                  
      0.0          117,290         33        3,554.2        1,400.0        609       17,340        4,927.2  cuLibraryGetKernel                     
      0.0           62,641         36        1,740.0          735.0        530        6,840        1,886.5  cudaEventCreateWithFlags               
      0.0           43,360          1       43,360.0       43,360.0     43,360       43,360            0.0  cudaStreamCreateWithFlags              
      0.0           29,090          1       29,090.0       29,090.0     29,090       29,090            0.0  cudaStreamSynchronize                  
      0.0           24,719          1       24,719.0       24,719.0     24,719       24,719            0.0  cudaStreamDestroy                      
      0.0           19,320          5        3,864.0        3,970.0      2,930        4,560          732.4  cuInit                                 
      0.0           11,150          1       11,150.0       11,150.0     11,150       11,150            0.0  cudaEventQuery                         
      0.0            6,570          4        1,642.5          480.0        360        5,250        2,405.7  cuModuleGetLoadingMode                 
      0.0            3,130          3        1,043.3          880.0        600        1,650          543.7  cudaGetDriverEntryPointByVersion_v12050
      0.0            1,100          2          550.0          550.0        530          570           28.3  cudaGetDeviceProperties_v12000         

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)                                                  Name
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------------------------------------------------------------------------------------
     17.5   17,372,988,733      3,840   4,524,215.8   4,385,763.0   2,382,287   5,844,847    402,948.1  void OwnTensor::mem_efficient_bwd_unified_kernel_exp12<(int)64, (bool)1>(OwnTensor::MemEfficientBwd…
     14.1   13,957,364,134      8,000   1,744,670.5   1,065,960.0     708,196  19,167,841  3,190,713.0  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nt_align4>(T1::Params)             
     13.4   13,247,186,912     17,280     766,619.6     889,767.0     213,537   1,441,388    300,105.2  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(T1::Params)              
      9.6    9,525,855,309      4,200   2,268,060.8   1,025,639.5     715,908  16,255,529  3,988,430.2  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_tn_align4>(T1::Params)             
      9.1    9,006,972,550      4,320   2,084,947.3   2,093,489.0   1,090,663   2,997,753    123,621.9  void OwnTensor::fused_attn_forward_kernel_tc_sm89<(int)64, (int)64, (int)64, (int)1>(OwnTensor::Mem…
      5.8    5,724,458,713        320  17,888,933.5  17,847,504.5  11,845,898  20,309,709  1,290,574.5  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(T1::Params)             
      4.8    4,749,834,118      7,680     618,468.0     498,451.5     194,881   1,361,706    302,204.2  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_nt_align4>(T1::Params)             
      4.1    4,081,612,567      7,680     531,460.0     442,643.0     173,953   1,018,889    253,406.4  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_tn_align4>(T1::Params)             
      4.1    4,039,048,523      3,840   1,051,835.6   1,017,672.0     735,717   1,374,251    100,644.0  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(T1::Params)             
      2.6    2,618,376,115        320   8,182,425.4   8,177,316.0   8,136,165   8,570,441     32,371.5  void OwnTensor::cuda::sparseCENormalize_from_stats<float, unsigned short>(const T1 *, const T2 *, T…
      2.6    2,617,602,790      3,840     681,667.4     681,669.0     668,102   1,036,456      6,886.8  void OwnTensor::cuda::gelu_backward_sm89_kernel<float>(const T1 *, const T1 *, T1 *, long)          
      2.1    2,089,157,891      4,320     483,601.4     483,524.0     475,780     492,964      2,160.8  void OwnTensor::cuda::gelu_forward_sm89_kernel<float>(const T1 *, T1 *, long)                       
      2.1    2,032,931,176     55,221      36,814.5       3,360.0       2,016     547,876     72,682.1  void OwnTensor::cuda::vectorized_kernel_impl<(int)4, OwnTensor::Add_inplace_Functor, float, float, …
      1.4    1,359,051,232      8,960     151,679.8     157,345.0       1,569     167,745     28,942.6  void OwnTensor::cuda::vectorized_kernel_impl<(int)4, OwnTensor::AddFunctor, float, float, float>(in…
      1.4    1,353,861,053        360   3,760,725.1   3,760,606.5   3,758,359   3,765,216      1,389.6  void OwnTensor::cuda::sparse_ce_forward_kernel_vec_save_stats<float, unsigned short>(const T1 *, co…
      1.2    1,221,530,996     15,680      77,903.8      39,488.0      19,104     168,225     49,317.4  void OwnTensor::cuda::unified_reduce_kernel<float, float, OwnTensor::detail::SumOp<float>, unsigned…
      1.2    1,195,552,078      9,000     132,839.1     132,961.0      78,912     180,705      6,022.1  void OwnTensor::cuda::layer_norm_forward_sm89_kernel<float, float>(const T1 *, const T1 *, const T1…
      1.2    1,147,347,583      8,000     143,418.4     140,545.0     124,096     176,417      8,614.4  void OwnTensor::cuda::ln_backward_input_sm89_kernel<float>(const T1 *, const T1 *, const float *, c…
      1.0      997,797,686      8,000     124,724.7     123,168.5     120,417     166,913      8,369.4  void OwnTensor::cuda::ln_backward_gamma_beta_sm89_kernel<float>(const T1 *, const T1 *, const float…
      0.6      627,027,616      3,840     163,288.4     163,202.0     160,161     166,753        796.4  void OwnTensor::mem_efficient_bwd_precompute_D_sm89<(int)64>(OwnTensor::MemEfficientBwdParams)      
      0.0       44,616,199        130     343,201.5     365,299.0      63,584     388,451     80,950.0  OwnTensor::cuda::multi_tensor_adam_sm89_kernel(OwnTensor::cuda::AdamLaunchMetadata, float, float, f…
      0.0       40,781,002        640      63,720.3      60,752.5       7,776     122,113     55,072.4  void OwnTensor::cuda::embedding_backward_kernel_optimized<float>(const unsigned short *, const T1 *…
      0.0       35,095,153        360      97,486.5      97,584.5      69,152     121,249      3,502.1  void OwnTensor::add_kernel_nd_broadcast<float>(const T1 *, const T1 *, T1 *, OwnTensor::SimplifiedB…
      0.0       29,386,191        720      40,814.2      18,016.0       3,360      88,385     33,940.3  OwnTensor::cuda::embedding_forward_kernel_vectorized(const unsigned short *, const float *, float *…
      0.0       12,194,879        130      93,806.8     105,969.0      23,584     110,529     26,095.8  OwnTensor::cuda::multi_tensor_scale_sm89_kernel(OwnTensor::cuda::ScaleLaunchMetadata, const float *)
      0.0       11,949,763         10   1,194,976.3   1,197,178.0   1,142,568   1,242,410     30,814.2  OwnTensor::cuda::multi_tensor_grad_norm_kernel(const OwnTensor::cuda::TensorInfo *, const long *, i…
      0.0        5,049,354        117      43,156.9      47,104.0       4,928      49,984     11,594.8  OwnTensor::cuda::multi_tensor_zero_sm89_kernel(OwnTensor::cuda::ZeroLaunchMetadata)                 
      0.0        4,325,795         10     432,579.5     431,779.0     427,268     438,404      3,036.7  void OwnTensor::transpose_2d_tiled_kernel<float, (int)32, (int)8>(const T1 *, T1 *, int, int)       
      0.0        1,993,452        720       2,768.7       2,784.0       1,504       3,392        249.3  void OwnTensor::cuda::sum_reduction_kernel<float>(const T1 *, T1 *, long)                           
      0.0        1,238,822        680       1,821.8       1,824.0       1,088       2,208         80.0  void OwnTensor::convert_type_kernel<long, unsigned short>(const T1 *, T2 *, long)                   
      0.0          850,823        360       2,363.4       2,336.0       1,440       2,784        133.2  void OwnTensor::scalar_div_copy<float, float>(const T1 *, double, T2 *, unsigned long)              
      0.0          469,827          2     234,913.5     234,913.5      10,752     459,075    317,012.2  void gen_sequenced<curandStateXORWOW, float2, normal_args_st, &curand_normal_scaled2<curandStateXOR…
      0.0          234,977          2     117,488.5     117,488.5      94,912     140,065     31,928.0  void generate_seed_pseudo<rng_config<curandStateXORWOW, (curandOrdering)101>>(unsigned long long, u…
      0.0           20,064         10       2,006.4       2,000.0       1,824       2,144         81.3  OwnTensor::cuda::compute_clip_coef_kernel(float *, float *, float, bool)                            

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)   Med (ns)  Min (ns)   Max (ns)    StdDev (ns)           Operation
 --------  ---------------  ------  ---------  --------  --------  -----------  -----------  ----------------------------
     61.4      530,565,174  33,018   16,069.0   1,472.0       352      180,481     38,455.8  [CUDA memset]               
     35.5      306,740,947     504  608,613.0   1,632.0       480  124,859,548  7,801,208.3  [CUDA memcpy Device-to-Host]
      3.1       26,593,608     887   29,981.5   3,040.0       480    1,062,951    126,612.6  [CUDA memcpy Host-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)   Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)           Operation
 -----------  ------  --------  --------  --------  --------  -----------  ----------------------------
 632,717.084  33,018    19.163     0.003     0.000   154.534       50.241  [CUDA memset]
   1,493.711     504     2.964     0.003     0.000   154.534       12.265  [CUDA memcpy Device-to-Host]
     363.861     887     0.410     0.033     0.001     9.437        1.732  [CUDA memcpy Host-to-Device]

Generated:
        /mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1/report42.nsys-rep
        /mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1/report42.sqlite
blubridge@blubridge:/mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1$
