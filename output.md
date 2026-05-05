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
gpt2_attn_navin.cpp:430:19: warning: unused variable ‘TOK_GEN_FREQ’ [-Wunused-variable]
  430 |         const int TOK_GEN_FREQ = 1000;
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
step     0 | loss: 11.041492 | lr 6.0000e-03 | norm: 15.6641 | dt: 13521.59ms | tok/sec: 38774.13
step     1 | loss: 9.946155 | lr 5.8679e-03 | norm: 2.6345 | dt: 9049.08ms | tok/sec: 57938.26
step     2 | loss: 8.721649 | lr 5.4843e-03 | norm: 6.0898 | dt: 9164.97ms | tok/sec: 57205.64
step     3 | loss: 8.154962 | lr 4.8870e-03 | norm: 3.4455 | dt: 9310.51ms | tok/sec: 56311.43
step     4 | loss: 8.199133 | lr 4.1343e-03 | norm: 2.0733 | dt: 9425.55ms | tok/sec: 55624.16
step     5 | loss: 8.073243 | lr 3.3000e-03 | norm: 1.5846 | dt: 9534.75ms | tok/sec: 54987.08
step     6 | loss: 15.133972 | lr 2.4657e-03 | norm: 2.7074 | dt: 9620.77ms | tok/sec: 54495.43
step     7 | loss: 7.969420 | lr 1.7130e-03 | norm: 1.4004 | dt: 9648.38ms | tok/sec: 54339.47
[CheckpointManager] Saved: checkpoints/gpt2_step_8.ckpt (safely)
step     8 | loss: 7.893339 | lr 1.1157e-03 | norm: 1.1773 | dt: 13372.42ms | tok/sec: 39206.66
validation loss: 7.8482
step     9 | loss: 7.829244 | lr 7.3215e-04 | norm: 0.7920 | dt: 11685.74ms | tok/sec: 44865.62

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
step     0 | loss: 11.041492 | lr 6.0000e-03 | norm: 15.6641 | dt: 14040.42ms | tok/sec: 37341.33
step     1 | loss: 9.946153 | lr 5.8679e-03 | norm: 2.6345 | dt: 9224.50ms | tok/sec: 56836.45
step     2 | loss: 8.721652 | lr 5.4843e-03 | norm: 6.0898 | dt: 9298.86ms | tok/sec: 56381.98
step     3 | loss: 8.154966 | lr 4.8870e-03 | norm: 3.4456 | dt: 9380.38ms | tok/sec: 55892.01
step     4 | loss: 8.199138 | lr 4.1343e-03 | norm: 2.0733 | dt: 9444.54ms | tok/sec: 55512.30
step     5 | loss: 8.073242 | lr 3.3000e-03 | norm: 1.5846 | dt: 9493.58ms | tok/sec: 55225.52
step     6 | loss: 15.133974 | lr 2.4657e-03 | norm: 2.7074 | dt: 9527.98ms | tok/sec: 55026.12
step     7 | loss: 7.969419 | lr 1.7130e-03 | norm: 1.4004 | dt: 9549.89ms | tok/sec: 54899.88
[CheckpointManager] Saved: checkpoints/gpt2_step_8.ckpt (safely)
step     8 | loss: 7.893341 | lr 1.1157e-03 | norm: 1.1774 | dt: 12580.06ms | tok/sec: 41676.12
validation loss: 7.8482
step     9 | loss: 7.829247 | lr 7.3215e-04 | norm: 0.7921 | dt: 11489.75ms | tok/sec: 45630.94

=== Training Complete ===
Generating '/tmp/nsys-report-7446.qdstrm'
[1/8] [========================100%] report39.nsys-rep
[2/8] [========================100%] report39.sqlite
[3/8] Executing 'nvtx_sum' stats report
SKIPPED: /mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1/report39.sqlite does not contain NV Tools Extension (NVTX) data.
[4/8] Executing 'osrt_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)      Min (ns)      Max (ns)       StdDev (ns)             Name
 --------  ---------------  ---------  -------------  -------------  -----------  -------------  ---------------  ----------------------
     49.6  222,276,774,324      2,217  100,260,159.8  100,121,442.0        5,450  1,512,199,475     30,714,076.6  poll                  
     48.8  218,517,813,326        248  881,120,215.0  500,069,702.0  500,030,652  4,000,135,693  1,092,376,792.1  pthread_cond_timedwait
      1.2    5,277,230,266      2,920    1,807,270.6    1,252,908.0        1,390    146,743,384      3,500,131.1  ioctl                 
      0.4    1,584,150,584        445    3,559,889.0       29,490.0        7,160    165,564,239     13,706,611.5  writev                
      0.0      149,355,483        134    1,114,593.2       37,424.5        6,660     48,008,819      4,223,084.5  mmap                  
      0.0       13,249,639         75      176,661.9        3,720.0        1,010      3,032,561        573,593.4  close                 
      0.0        8,143,052         21      387,764.4       43,369.0       24,100      4,307,710      1,007,797.4  pthread_rwlock_rdlock 
      0.0        6,510,465         54      120,564.2       36,844.5       12,550      3,664,216        495,176.5  mmap64                
      0.0        1,760,452         36       48,901.4       15,650.0        6,230      1,003,381        165,428.6  munmap                
      0.0        1,726,204         86       20,072.1       19,555.0        6,550         48,639          8,031.0  open64                
      0.0        1,620,393         23       70,451.9        8,860.0        1,500        926,991        189,676.7  write                 
      0.0        1,438,575         20       71,928.8       43,794.5       10,820        169,949         60,750.6  pthread_rwlock_wrlock 
      0.0        1,320,733         56       23,584.5       17,415.0        2,600         76,919         19,799.7  fopen                 
      0.0        1,058,421         10      105,842.1       42,550.0       26,799        672,783        199,874.0  sem_timedwait         
      0.0          996,931          4      249,232.8      246,997.5      186,599        316,337         55,938.3  pthread_create        
      0.0          275,038         29        9,484.1        3,060.0        1,150         20,860          8,366.9  putc                  
      0.0          262,146         52        5,041.3        4,564.5        2,040         13,800          2,273.3  fclose                
      0.0          239,418          2      119,709.0      119,709.0      107,859        131,559         16,758.4  fopen64               
      0.0          191,318          1      191,318.0      191,318.0      191,318        191,318              0.0  pthread_cond_wait     
      0.0          171,989          8       21,498.6       22,630.0        1,410         30,079          8,954.3  pthread_mutex_lock    
      0.0          145,248         10       14,524.8       14,869.5        4,870         24,770          7,241.7  open                  
      0.0          135,408          1      135,408.0      135,408.0      135,408        135,408              0.0  fgets                 
      0.0           98,059         21        4,669.5        2,820.0        1,280         28,960          6,414.6  fwrite                
      0.0           49,061         15        3,270.7        3,260.0        1,000          6,720          1,699.1  read                  
      0.0           48,609          2       24,304.5       24,304.5       17,960         30,649          8,972.5  fread                 
      0.0           47,750          2       23,875.0       23,875.0       21,800         25,950          2,934.5  socket                
      0.0           44,899          4       11,224.8       11,280.0        5,620         16,719          5,414.8  pipe2                 
      0.0           24,999          1       24,999.0       24,999.0       24,999         24,999              0.0  connect               
      0.0           20,090         12        1,674.2        1,860.0        1,010          2,340            478.3  fcntl                 
      0.0           19,860          3        6,620.0        7,720.0        3,080          9,060          3,138.1  stat                  
      0.0           11,540          4        2,885.0        1,220.0        1,200          7,900          3,343.3  pthread_cond_broadcast
      0.0            7,800          4        1,950.0        1,850.0        1,400          2,700            550.5  fstat                 
      0.0            7,160          1        7,160.0        7,160.0        7,160          7,160              0.0  bind                  
      0.0            4,660          4        1,165.0        1,190.0        1,010          1,270            120.4  dup                   
      0.0            2,100          1        2,100.0        2,100.0        2,100          2,100              0.0  listen                

[5/8] Executing 'cuda_api_sum' stats report

 Time (%)  Total Time (ns)  Num Calls    Avg (ns)       Med (ns)     Min (ns)    Max (ns)     StdDev (ns)                    Name
 --------  ---------------  ---------  -------------  -------------  ---------  -----------  -------------  ---------------------------------------
     59.3   58,570,390,660    129,592      451,959.9       94,979.0      5,370  397,540,633    1,814,721.3  cudaLaunchKernel                       
     19.3   19,070,918,023     49,000      389,202.4       56,659.5      4,870   18,598,196    1,401,185.7  cuLaunchKernel                         
      9.2    9,129,658,354     33,005      276,614.4       11,810.0      3,150    3,770,814      559,565.2  cudaMemsetAsync                        
      4.1    4,043,223,465      1,741    2,322,357.0       20,770.0      4,880  124,525,842   14,482,415.9  cudaMemcpyAsync                        
      3.7    3,680,067,716         10  368,006,771.6  409,412,288.0  1,130,819  413,596,383  128,966,847.4  cudaMemcpy                             
      2.7    2,640,447,135        333    7,929,270.7        8,520.0      1,800  381,712,962   40,943,822.8  cudaMallocAsync_v11020                 
      0.8      762,560,754        157    4,857,074.9       14,200.0      3,650  152,741,288   13,524,518.5  cudaHostAlloc                          
      0.4      398,525,493         16   24,907,843.3    1,638,775.0    189,888  292,461,128   73,125,735.2  cuLibraryLoadData                      
      0.3      321,890,679        152    2,117,701.8       14,525.0      4,030   58,175,504    5,453,924.7  cudaFreeHost                           
      0.0       46,909,204     49,000          957.3          631.0        450    8,950,206       40,454.3  cuKernelGetFunction                    
      0.0       39,341,742     11,840        3,322.8        2,920.0      1,520       20,030        1,514.2  cudaEventRecord                        
      0.0       34,636,620    129,592          267.3          220.0        180       21,489          188.6  cuKernelGetName                        
      0.0       30,733,352         16    1,920,834.5       13,865.0      3,200    7,921,096    2,781,305.4  cudaMalloc                             
      0.0       17,545,864     59,200          296.4          230.0        169      101,699          609.6  cuStreamGetCaptureInfo_v2              
      0.0        8,987,006         12      748,917.2        1,140.0        451    4,671,136    1,737,954.3  cudaFree                               
      0.0        8,326,796      8,640          963.7          710.0        580      158,998        1,878.9  cudaStreamSetAttribute_v11000          
      0.0          525,169      1,320          397.9          350.0        170        9,920          313.8  cuGetProcAddress_v2                    
      0.0          396,285         13       30,483.5       31,040.0     25,230       35,320        2,656.6  cuMemsetD32Async                       
      0.0          230,228          2      115,114.0      115,114.0     89,119      141,109       36,762.5  cudaDeviceSynchronize                  
      0.0          120,511         33        3,651.8        1,550.0        520       18,239        5,320.6  cuLibraryGetKernel                     
      0.0           61,849          1       61,849.0       61,849.0     61,849       61,849            0.0  cudaStreamSynchronize                  
      0.0           57,271         36        1,590.9          664.5        530       10,390        2,201.6  cudaEventCreateWithFlags               
      0.0           41,340          1       41,340.0       41,340.0     41,340       41,340            0.0  cudaStreamCreateWithFlags              
      0.0           24,210          1       24,210.0       24,210.0     24,210       24,210            0.0  cudaStreamDestroy                      
      0.0           18,361          5        3,672.2        3,851.0      2,430        4,160          710.3  cuInit                                 
      0.0           10,670          1       10,670.0       10,670.0     10,670       10,670            0.0  cudaEventQuery                         
      0.0            6,410          4        1,602.5          535.0        270        5,070        2,315.6  cuModuleGetLoadingMode                 
      0.0            3,469          3        1,156.3          769.0        720        1,980          713.7  cudaGetDriverEntryPointByVersion_v12050
      0.0            1,170          2          585.0          585.0        440          730          205.1  cudaGetDeviceProperties_v12000         

[6/8] Executing 'cuda_gpu_kern_sum' stats report

 Time (%)  Total Time (ns)  Instances    Avg (ns)      Med (ns)     Min (ns)    Max (ns)   StdDev (ns)                                                  Name
 --------  ---------------  ---------  ------------  ------------  ----------  ----------  -----------  ----------------------------------------------------------------------------------------------------
     17.4   16,943,089,639      3,840   4,412,262.9   4,281,742.0   2,375,485   5,845,364    369,896.4  void OwnTensor::mem_efficient_bwd_unified_kernel_exp12<(int)64, (bool)1>(OwnTensor::MemEfficientBwd…
     13.9   13,552,616,006      8,000   1,694,077.0   1,040,847.0     703,753  18,663,312  3,083,293.6  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nt_align4>(T1::Params)             
     13.4   13,009,216,828     17,280     752,848.2     877,340.0     213,955   1,274,738    294,891.2  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x64_16x4_nn_align4>(T1::Params)              
      9.6    9,312,022,162      4,200   2,217,148.1   1,001,582.0     714,249  15,903,589  3,911,835.6  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_256x128_16x3_tn_align4>(T1::Params)             
      9.1    8,803,772,896      4,320   2,037,910.4   2,024,412.5   1,089,389   2,666,854    117,018.2  void OwnTensor::fused_attn_forward_kernel_tc_sm89<(int)64, (int)64, (int)64, (int)1>(OwnTensor::Mem…
      5.8    5,660,702,933        320  17,689,696.7  17,880,015.5  11,812,785  19,712,382  1,216,589.7  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_nn_align4>(T1::Params)             
      4.8    4,623,786,764      7,680     602,055.6     487,910.5     199,075   1,398,004    293,576.7  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_32x3_nt_align4>(T1::Params)             
      4.1    3,973,652,183      7,680     517,402.6     512,006.0     174,786     985,198    245,856.4  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x256_16x3_tn_align4>(T1::Params)             
      4.0    3,935,231,017      3,840   1,024,799.7     986,414.0     736,233   1,309,171     92,076.9  void cutlass::Kernel2<cutlass_80_tensorop_s1688gemm_128x128_16x5_tn_align4>(T1::Params)             
      2.7    2,619,358,480        320   8,185,495.3   8,178,487.0   8,129,110   8,592,156     39,795.9  void OwnTensor::cuda::sparseCENormalize_from_stats<float, unsigned short>(const T1 *, const T2 *, T…
      2.7    2,616,634,753      3,840     681,415.3     681,545.0     668,169     692,777      3,794.1  void OwnTensor::cuda::gelu_backward_sm89_kernel<float>(const T1 *, const T1 *, T1 *, long)          
      2.1    2,089,027,592      4,320     483,571.2     483,463.0     476,998     494,727      2,195.0  void OwnTensor::cuda::gelu_forward_sm89_kernel<float>(const T1 *, T1 *, long)                       
      2.1    2,030,611,055     55,221      36,772.4       3,296.0       1,984     550,503     72,689.6  void OwnTensor::cuda::vectorized_kernel_impl<(int)4, OwnTensor::Add_inplace_Functor, float, float, …
      1.4    1,358,373,533      8,960     151,604.2     157,282.0       1,152     168,995     28,940.5  void OwnTensor::cuda::vectorized_kernel_impl<(int)4, OwnTensor::AddFunctor, float, float, float>(in…
      1.4    1,353,812,337        360   3,760,589.8   3,760,694.0   3,757,581   3,764,372      1,328.2  void OwnTensor::cuda::sparse_ce_forward_kernel_vec_save_stats<float, unsigned short>(const T1 *, co…
      1.2    1,203,319,872     15,680      76,742.3      38,352.5      19,456     165,058     48,779.0  void OwnTensor::cuda::unified_reduce_kernel<float, float, OwnTensor::detail::SumOp<float>, unsigned…
      1.2    1,176,035,355      9,000     130,670.6     130,562.0      78,273     161,666      5,685.4  void OwnTensor::cuda::layer_norm_forward_sm89_kernel<float, float>(const T1 *, const T1 *, const T1…
      1.2    1,132,341,220      8,000     141,542.7     138,978.0     123,681     171,427      7,790.6  void OwnTensor::cuda::ln_backward_input_sm89_kernel<float>(const T1 *, const T1 *, const float *, c…
      1.0      997,217,288      8,000     124,652.2     123,073.0     120,226     166,690      8,358.3  void OwnTensor::cuda::ln_backward_gamma_beta_sm89_kernel<float>(const T1 *, const T1 *, const float…
      0.6      626,852,723      3,840     163,242.9     163,170.0     160,834     167,330        808.5  void OwnTensor::mem_efficient_bwd_precompute_D_sm89<(int)64>(OwnTensor::MemEfficientBwdParams)      
      0.0       44,852,374        130     345,018.3     368,885.0      60,257     390,917     82,252.4  OwnTensor::cuda::multi_tensor_adam_sm89_kernel(OwnTensor::cuda::AdamLaunchMetadata, float, float, f…
      0.0       40,670,066        640      63,547.0      60,720.5       7,872     122,562     54,881.7  void OwnTensor::cuda::embedding_backward_kernel_optimized<float>(const unsigned short *, const T1 *…
      0.0       34,600,271        360      96,111.9      95,457.0      69,729     117,537      3,978.2  void OwnTensor::add_kernel_nd_broadcast<float>(const T1 *, const T1 *, T1 *, OwnTensor::SimplifiedB…
      0.0       29,358,792        720      40,776.1      17,392.0       3,648      87,105     33,894.4  OwnTensor::cuda::embedding_forward_kernel_vectorized(const unsigned short *, const float *, float *…
      0.0       11,644,612         10   1,164,461.2   1,167,504.5   1,125,999   1,204,497     20,683.1  OwnTensor::cuda::multi_tensor_grad_norm_kernel(const OwnTensor::cuda::TensorInfo *, const long *, i…
      0.0       10,979,163        130      84,455.1     105,393.0       1,504     112,066     37,188.2  OwnTensor::cuda::multi_tensor_scale_sm89_kernel(OwnTensor::cuda::ScaleLaunchMetadata, const float *)
      0.0        5,068,870        117      43,323.7      47,072.0       5,088      49,985     11,547.2  OwnTensor::cuda::multi_tensor_zero_sm89_kernel(OwnTensor::cuda::ZeroLaunchMetadata)                 
      0.0        4,328,252         10     432,825.2     432,806.0     426,406     436,454      3,117.5  void OwnTensor::transpose_2d_tiled_kernel<float, (int)32, (int)8>(const T1 *, T1 *, int, int)       
      0.0        1,975,393        720       2,743.6       2,784.0       1,472       3,488        248.1  void OwnTensor::cuda::sum_reduction_kernel<float>(const T1 *, T1 *, long)                           
      0.0        1,156,945        680       1,701.4       1,712.0       1,088       1,984        126.0  void OwnTensor::convert_type_kernel<long, unsigned short>(const T1 *, T2 *, long)                   
      0.0          842,093        360       2,339.1       2,304.0       1,440       2,784        132.5  void OwnTensor::scalar_div_copy<float, float>(const T1 *, double, T2 *, unsigned long)              
      0.0          469,927          2     234,963.5     234,963.5      10,849     459,078    316,945.8  void gen_sequenced<curandStateXORWOW, float2, normal_args_st, &curand_normal_scaled2<curandStateXOR…
      0.0          234,916          2     117,458.0     117,458.0      94,882     140,034     31,927.3  void generate_seed_pseudo<rng_config<curandStateXORWOW, (curandOrdering)101>>(unsigned long long, u…
      0.0           19,937         10       1,993.7       1,984.0       1,792       2,113         84.2  OwnTensor::cuda::compute_clip_coef_kernel(float *, float *, float, bool)                            

[7/8] Executing 'cuda_gpu_mem_time_sum' stats report

 Time (%)  Total Time (ns)  Count   Avg (ns)   Med (ns)  Min (ns)   Max (ns)    StdDev (ns)            Operation
 --------  ---------------  ------  ---------  --------  --------  -----------  -----------  ------------------------------
     61.9      524,324,752  33,018   15,880.0   1,472.0       352      181,219     38,381.7  [CUDA memset]                 
     35.8      303,045,968     504  601,281.7   1,664.0       672  123,420,017  7,707,619.0  [CUDA memcpy Device-to-Host]  
      2.3       19,268,566     887   21,723.3   3,072.0       480      517,927     86,843.5  [CUDA memcpy Host-to-Device]  
      0.1          631,432     360    1,754.0   1,728.0     1,056        2,208        102.2  [CUDA memcpy Device-to-Device]

[8/8] Executing 'cuda_gpu_mem_size_sum' stats report

 Total (MB)   Count   Avg (MB)  Med (MB)  Min (MB)  Max (MB)  StdDev (MB)            Operation
 -----------  ------  --------  --------  --------  --------  -----------  ------------------------------
 632,717.084  33,018    19.163     0.003     0.000   154.534       50.241  [CUDA memset]
   1,493.711     504     2.964     0.003     0.000   154.534       12.265  [CUDA memcpy Device-to-Host]  
     363.861     887     0.410     0.033     0.001     9.437        1.732  [CUDA memcpy Host-to-Device]  
       2.949     360     0.008     0.008     0.008     0.008        0.000  [CUDA memcpy Device-to-Device]

Generated:
        /mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1/report39.nsys-rep
        /mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1/report39.sqlite
blubridge@blubridge:/mnt/volgrp03/3rd_floor/Gautam/master_gau_latest_1$
