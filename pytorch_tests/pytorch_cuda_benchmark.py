import torch
import pandas as pd
import time
import numpy as np
import os

# Check CUDA availability
if not torch.cuda.is_available():
    print("CUDA not available! This script requires a GPU.")
    print("   PyTorch CUDA support:", torch.cuda.is_available())
    print("   Install CUDA-enabled PyTorch to run GPU benchmarks.")
    exit(1)

print("="*60)
print("PyTorch CUDA (GPU) Benchmark")
print("="*60)
print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
print(f"✓ CUDA version: {torch.version.cuda}")
print()


D = 10
R = 10
C = 10
size = D * R * C

print(f"Using LARGE tensors: [{D}, {R}, {C}] -> {size} elements")

device = torch.device('cuda')

# Read inputs from CSV
input_csv_path = "/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/inputs/benchmark_all_3d_inputs.csv"
print(f"Reading inputs from {input_csv_path}...")

try:
    df_inputs = pd.read_csv(input_csv_path)
    
    # Validate Shape matches strictly
    max_d = df_inputs['depth'].max()
    max_r = df_inputs['row'].max()
    max_c = df_inputs['col'].max()
    
    if (max_d != D-1) or (max_r != R-1) or (max_c != C-1):
        raise ValueError(f"Shape mismatch! CSV has [{max_d+1}, {max_r+1}, {max_c+1}], expected [{D}, {R}, {C}]")
    
    # Extract input columns
    input_a = df_inputs['input_a'].values
    input_b = df_inputs['input_b'].values
    
    # Reshape to (D, R, C)
    if len(input_a) != size:
        raise ValueError(f"CSV has {len(input_a)} elements, expected {size}")
        
    input_a = input_a.reshape(D, R, C)
    input_b = input_b.reshape(D, R, C)
    
    # Move to GPU tensor
    a = torch.tensor(input_a, dtype=torch.float32, device=device)
    b = torch.tensor(input_b, dtype=torch.float32, device=device)
    
    print("✓ Inputs loaded from CSV")

except (FileNotFoundError, ValueError, Exception) as e:
    print(f"⚠ Could not read inputs from CSV: {e}")
    print("   Falling back to RANDOM generation...")
    a = torch.randn(D, R, C, dtype=torch.float32, device=device)
    b = torch.randn(D, R, C, dtype=torch.float32, device=device)
    
    # Save these generated inputs so we know what was used
    try:
        os.makedirs("benchmark_results/pytorch_cuda", exist_ok=True)
        save_path = "benchmark_results/pytorch_cuda/generated_inputs.csv"
        print(f"   Saving generated inputs to {save_path}...")
        
        # Create DataFrame
        a_cpu = a.cpu().numpy().flatten()
        b_cpu = b.cpu().numpy().flatten()
        indices = np.arange(size)
        
        df_gen = pd.DataFrame({
            'index': indices,
            'depth': indices // (R*C),
            'row': (indices // C) % R,
            'col': indices % C,
            'input_a': a_cpu,
            'input_b': b_cpu
        })
        df_gen.to_csv(save_path, index=False)
        print("   ✓ Inputs saved.")
    except Exception as save_err:
        print(f"   ⚠ Failed to save generated inputs: {save_err}")

a_pos = torch.abs(a) + 0.1
b_pos = torch.abs(b) + 0.1

# Timing setup
NUM_RUNS = 50
WARMUP_RUNS = 5
timings = {}

def time_operation(name, func, num_runs=NUM_RUNS, warmup=WARMUP_RUNS, num_elements=size, 
                   dtype_size=4, op_type='binary'):
    """
    Time a CUDA operation with GPU synchronization.
    
    Args:
        name: Operation name
        func: Lambda function to run
        num_runs: Number of timing iterations
        warmup: Number of warmup iterations
        num_elements: Total number of elements processed
        dtype_size: Size of data type in bytes (4 for float32)
        op_type: 'binary' (3x), 'unary' (2x), 'reduction' (1x)
    """
    
    # Calculate Theoretical Bytes Accessed based on Op Type
    if op_type == 'binary':
        bytes_per_access = dtype_size * 3
    elif op_type == 'unary':
        bytes_per_access = dtype_size * 2
    elif op_type == 'reduction':
        bytes_per_access = dtype_size * 1
    elif op_type == 'matmul':
        bytes_per_access = dtype_size * 3 
    else:
        bytes_per_access = dtype_size * 3
        
    # Warmup
    for _ in range(warmup):
        _ = func()
        torch.cuda.synchronize()  # Important for GPU timing!
    
    # Actual timing
    times = []
    for _ in range(num_runs):
        torch.cuda.synchronize()  # Ensure previous ops finished
        start = time.perf_counter()
        result = func()
        torch.cuda.synchronize()  # Wait for GPU to finish
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_time_ms = np.mean(times)
    mean_time_sec = mean_time_ms / 1000.0
    
    # Debug print
    if name == 'add':
        print(f"DEBUG: op={name}, mean_ms={mean_time_ms:.5f}, size={num_elements}, throughput={num_elements/mean_time_sec:.2e}")

    # Calculate throughput and bandwidth
    throughput_elements_per_sec = num_elements / mean_time_sec if mean_time_sec > 0 else 0
    memory_bandwidth_gb_per_sec = (num_elements * bytes_per_access) / mean_time_sec / 1e9 if mean_time_sec > 0 else 0
    
    timings[name] = {
        'mean_ms': mean_time_ms,
        'min_ms': np.min(times),
        'max_ms': np.max(times),
        'std_ms': np.std(times),
        'throughput_elem_per_sec': throughput_elements_per_sec,
        'memory_bandwidth_gb_per_sec': memory_bandwidth_gb_per_sec
    }
    return result

print("Running PyTorch CUDA benchmark with timing...")

# All operations with timing (same as CPU version)
# Binary ops (Default op_type='binary' -> 3 * 4 = 12 bytes/elem)
add_res = time_operation("add", lambda: a + b, op_type='binary')
sub_result = time_operation("sub", lambda: a - b, op_type='binary')
mul_result = time_operation("mul", lambda: a * b, op_type='binary')
div_result = time_operation("div", lambda: a / (b + 0.1), op_type='binary')

# Unary ops (op_type='unary' -> 2 * 4 = 8 bytes/elem)
square_result = time_operation("square", lambda: torch.square(a), op_type='unary')
sqrt_result = time_operation("sqrt", lambda: torch.sqrt(a), op_type='unary')
neg_result = time_operation("neg", lambda: torch.neg(a), op_type='unary')
abs_result = time_operation("abs", lambda: torch.abs(a), op_type='unary')
sign_result = time_operation("sign", lambda: torch.sign(a), op_type='unary')
reciprocal_result = time_operation("reciprocal", lambda: torch.reciprocal(a), op_type='unary')
pow2_result = time_operation("pow2", lambda: torch.pow(a, 2), op_type='unary')

sin_result = time_operation("sin", lambda: torch.sin(a), op_type='unary')
cos_result = time_operation("cos", lambda: torch.cos(a), op_type='unary')
tan_result = time_operation("tan", lambda: torch.tan(a), op_type='unary')

sinh_result = time_operation("sinh", lambda: torch.sinh(a), op_type='unary')
cosh_result = time_operation("cosh", lambda: torch.cosh(a), op_type='unary')
tanh_result = time_operation("tanh", lambda: torch.tanh(a), op_type='unary')

asin_result = time_operation("asin", lambda: torch.asin(a), op_type='unary')
acos_result = time_operation("acos", lambda: torch.acos(a), op_type='unary')
atan_result = time_operation("atan", lambda: torch.atan(a), op_type='unary')

asinh_result = time_operation("asinh", lambda: torch.asinh(a), op_type='unary')
acosh_result = time_operation("acosh", lambda: torch.acosh(a), op_type='unary')
atanh_result = time_operation("atanh", lambda: torch.atanh(a), op_type='unary')

exp_result = time_operation("exp", lambda: torch.exp(a), op_type='unary')
log_result = time_operation("log", lambda: torch.log(a), op_type='unary')
log2_result = time_operation("log2", lambda: torch.log2(a), op_type='unary')
log10_result = time_operation("log10", lambda: torch.log10(a), op_type='unary')

matmul_result = time_operation("matmul", lambda: torch.matmul(a, b.transpose(1, 2)), op_type='matmul')

# Reductions
# Reductions (op_type='reduction')
sum_result = time_operation("sum_all", lambda: torch.sum(a), op_type='reduction')
mean_result = time_operation("mean_all", lambda: torch.mean(a), op_type='reduction')
max_result = time_operation("max_all", lambda: torch.max(a), op_type='reduction')
min_result = time_operation("min_all", lambda: torch.min(a), op_type='reduction')
var_result = time_operation("var_all", lambda: torch.var(a), op_type='reduction')
std_result = time_operation("std_all", lambda: torch.std(a), op_type='reduction')

sum_all = sum_result
mean_all = mean_result  
max_all = max_result
min_all = min_result
var_all = var_result
std_all = std_result

scalar_add = time_operation("scalar_add", lambda: a + 2.5)
scalar_mul = time_operation("scalar_mul", lambda: a * 3.0)
scalar_div = time_operation("scalar_div", lambda: a / 2.0)
reverse_sub = time_operation("reverse_sub", lambda: 5.0 - a)

# Chains (op_type='unary' because effective result is Tensor -> Tensor, intermediate steps are fused/cached)
chain1 = time_operation("chain1", lambda: torch.sin(torch.cos(torch.sqrt(torch.square(a)))), op_type='unary')
chain2 = time_operation("chain2", lambda: torch.exp(torch.log(torch.log2(torch.log10(a_pos)))), op_type='unary')
chain3 = time_operation("chain3", lambda: torch.sin(torch.cos(torch.tan(matmul_result))), op_type='unary')
chain4 = time_operation("chain4", lambda: torch.square(torch.log(torch.tan(matmul_result + 0.5))), op_type='unary')
chain5 = time_operation("chain5", lambda: torch.tanh(torch.sin(torch.exp(a))), op_type='unary')
chain6 = time_operation("chain6", lambda: torch.log(torch.exp(torch.sqrt(a))), op_type='unary')
chain7 = time_operation("chain7", lambda: torch.cos(torch.sin(torch.tanh(torch.log(a_pos)))), op_type='unary')
chain8 = time_operation("chain8", lambda: torch.sqrt(torch.square(torch.exp(torch.log(a_pos)))), op_type='unary')
chain9 = time_operation("chain9", lambda: torch.log(torch.reciprocal(torch.sqrt(abs(torch.sin(a) + 0.1)))), op_type='unary')
chain10 = time_operation("chain10", lambda: torch.atan(torch.sinh(torch.tan(torch.cos(torch.sqrt(a_pos))))), op_type='unary')
chain11 = time_operation("chain11", lambda: torch.sin(a + b), op_type='binary')
chain12 = time_operation("chain12", lambda: torch.log(a_pos + b_pos), op_type='binary')
chain13 = time_operation("chain13", lambda: torch.tanh(torch.exp(a) + torch.log(b_pos)), op_type='binary')
chain14 = time_operation("chain14", lambda: torch.sin(torch.cos(torch.tan(torch.exp(torch.log(torch.log10(a_pos)))))), op_type='unary')
chain15 = time_operation("chain15", lambda: torch.exp(torch.sin(torch.cos(torch.tanh(torch.exp(torch.log(a_pos)))))), op_type='unary')

print(f"✓ Completed {len(timings)} operations")

# Move results to CPU for saving
columns = ['Idx','Batch', 'Row', 'Column','TENSOR a', 'TENSOR b', 
           'Addition', 'Sub', 'Mul','Div',
           'Square','Sqrt','neg','abs','sign','reciprocal','pow',
           'sin', 'cos', 'tan', 'sinh','cosh','tanh', 'asin','acos','atan',
           'asinh','acosh','atanh', 
           'exp','log', 'log2','log10',
           'scalar_add','scalar_mul','scalar_div','reverse_sub',
           'sum_all','mean_all','max_all','min_all','var_all','std_all',
           'chain1','chain2','chain3','chain4','chain5','chain6','chain7','chain8',
            'chain9','chain10','chain11','chain12','chain13','chain14','chain15'
           ]

# Optimized saving: only save first 1000 elements to avoid massive slowdown
limit = 1000
indices = range(limit)

data_dict = {
    'Idx': indices,
    'Batch': [i // (R*C) for i in indices],
    'Row': [(i // C) % R for i in indices],
    'Column': [i % C for i in indices],
    'TENSOR a': a.flatten()[:limit].cpu().numpy(),
    'TENSOR b': b.flatten()[:limit].cpu().numpy(),
    'Addition': add_res.flatten()[:limit].cpu().numpy(),
    'Sub': sub_result.flatten()[:limit].cpu().numpy(),
    'Mul': mul_result.flatten()[:limit].cpu().numpy(),
    'Div': div_result.flatten()[:limit].cpu().numpy(),
    'Square': square_result.flatten()[:limit].cpu().numpy(),
    'Sqrt': sqrt_result.flatten()[:limit].cpu().numpy(),
    'neg': neg_result.flatten()[:limit].cpu().numpy(),
    'abs': abs_result.flatten()[:limit].cpu().numpy(),
    'sign': sign_result.flatten()[:limit].cpu().numpy(),
    'reciprocal': reciprocal_result.flatten()[:limit].cpu().numpy(),
    'pow': pow2_result.flatten()[:limit].cpu().numpy(),
    'sin': sin_result.flatten()[:limit].cpu().numpy(),
    'cos': cos_result.flatten()[:limit].cpu().numpy(),
    'tan': tan_result.flatten()[:limit].cpu().numpy(),
    'sinh': sinh_result.flatten()[:limit].cpu().numpy(),
    'cosh': cosh_result.flatten()[:limit].cpu().numpy(),
    'tanh': tanh_result.flatten()[:limit].cpu().numpy(),
    'asin': asin_result.flatten()[:limit].cpu().numpy(),
    'acos': acos_result.flatten()[:limit].cpu().numpy(),
    'atan': atan_result.flatten()[:limit].cpu().numpy(),
    'asinh': asinh_result.flatten()[:limit].cpu().numpy(),
    'acosh': acosh_result.flatten()[:limit].cpu().numpy(),
    'atanh': atanh_result.flatten()[:limit].cpu().numpy(),
    'exp': exp_result.flatten()[:limit].cpu().numpy(),
    'log': log_result.flatten()[:limit].cpu().numpy(),
    'log2': log2_result.flatten()[:limit].cpu().numpy(),
    'log10': log10_result.flatten()[:limit].cpu().numpy(),
    'scalar_add': scalar_add.flatten()[:limit].cpu().numpy(),
    'scalar_mul': scalar_mul.flatten()[:limit].cpu().numpy(),
    'scalar_div': scalar_div.flatten()[:limit].cpu().numpy(),
    'reverse_sub': reverse_sub.flatten()[:limit].cpu().numpy(),
    'sum_all': [sum_all.item()] * limit,
    'mean_all': [mean_all.item()] * limit,
    'max_all': [max_all.item()] * limit,
    'min_all': [min_all.item()] * limit,
    'var_all': [var_all.item()] * limit,
    'std_all': [std_all.item()] * limit,
    'chain1': chain1.flatten()[:limit].cpu().numpy(),
    'chain2': chain2.flatten()[:limit].cpu().numpy(),
    'chain3': chain3.flatten()[:limit].cpu().numpy(),
    'chain4': chain4.flatten()[:limit].cpu().numpy(),
    'chain5': chain5.flatten()[:limit].cpu().numpy(),
    'chain6': chain6.flatten()[:limit].cpu().numpy(),
    'chain7': chain7.flatten()[:limit].cpu().numpy(),
    'chain8': chain8.flatten()[:limit].cpu().numpy(),
    'chain9': chain9.flatten()[:limit].cpu().numpy(),
    'chain10': chain10.flatten()[:limit].cpu().numpy(),
    'chain11': chain11.flatten()[:limit].cpu().numpy(),
    'chain12': chain12.flatten()[:limit].cpu().numpy(),
    'chain13': chain13.flatten()[:limit].cpu().numpy(),
    'chain14': chain14.flatten()[:limit].cpu().numpy(),
    'chain15': chain15.flatten()[:limit].cpu().numpy()
}

df = pd.DataFrame(data_dict)

# Create pytorch_cuda directory
import os
os.makedirs("benchmark_results/pytorch_cuda", exist_ok=True)

# 1. Save VALUES CSV
df.to_csv("benchmark_results/pytorch_cuda/pytorch_cuda_values.csv", index=False)
print("✓ Values saved to: benchmark_results/pytorch_cuda/pytorch_cuda_values.csv")

# Create metrics DataFrames
all_metrics_df = pd.DataFrame([
    {'operation': op, **metrics}
    for op, metrics in timings.items()
])

# 2. Save TIMINGS CSV
timings_df = all_metrics_df[['operation', 'mean_ms', 'min_ms', 'max_ms', 'std_ms']].copy()
timings_df.to_csv("benchmark_results/pytorch_cuda/pytorch_cuda_timings.csv", index=False)
print("✓ Timings saved to: benchmark_results/pytorch_cuda/pytorch_cuda_timings.csv")

# 3. Save THROUGHPUT CSV
throughput_df = all_metrics_df[['operation', 'throughput_elem_per_sec']].copy()
throughput_df.to_csv("benchmark_results/pytorch_cuda/pytorch_cuda_throughput.csv", index=False)
print("✓ Throughput saved to: benchmark_results/pytorch_cuda/pytorch_cuda_throughput.csv")

# 4. Save BANDWIDTH CSV
bandwidth_df = all_metrics_df[['operation', 'memory_bandwidth_gb_per_sec']].copy()
bandwidth_df.to_csv("benchmark_results/pytorch_cuda/pytorch_cuda_bandwidth.csv", index=False)
print("✓ Bandwidth saved to: benchmark_results/pytorch_cuda/pytorch_cuda_bandwidth.csv")

# 5. Save FLOPS CSV
flops_data = []
for op, metrics in timings.items():
    mean_time_sec = metrics['mean_ms'] / 1000.0
    
    if op in ['add', 'sub', 'mul', 'div', 'square', 'sqrt', 'neg', 'abs', 'sign', 'reciprocal', 'pow2',
              'scalar_add', 'scalar_mul', 'scalar_div', 'reverse_sub']:
        flops_per_elem = 1
    elif op in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh',
                'exp', 'log', 'log2', 'log10']:
        flops_per_elem = 5
    elif op == 'matmul':
        flops_per_elem = 2 * C
    elif 'chain' in op:
        flops_per_elem = 10
    else:
        flops_per_elem = 1
    
    total_flops = flops_per_elem * size
    gflops = (total_flops / mean_time_sec / 1e9) if mean_time_sec > 0 else 0
    
    flops_data.append({
        'operation': op,
        'gflops': gflops
    })

flops_df = pd.DataFrame(flops_data)
flops_df.to_csv("benchmark_results/pytorch_cuda/pytorch_cuda_flops.csv", index=False)
print("✓ FLOPS saved to: benchmark_results/pytorch_cuda/pytorch_cuda_flops.csv")

# Print summary
print(f"\n{'='*60}")
print(f"PyTorch CUDA Benchmark Summary")
print(f"{'='*60}")
print(f"  Device: {torch.cuda.get_device_name(0)}")
print(f"  Total operations: {len(timings)}")
print(f"  Fastest: {timings_df.loc[timings_df['mean_ms'].idxmin(), 'operation']} ({timings_df['mean_ms'].min():.4f} ms)")
print(f"  Slowest: {timings_df.loc[timings_df['mean_ms'].idxmax(), 'operation']} ({timings_df['mean_ms'].max():.4f} ms)")
print(f"  Peak throughput: {throughput_df['throughput_elem_per_sec'].max():.2e} elem/s")
print(f"  Peak bandwidth: {bandwidth_df['memory_bandwidth_gb_per_sec'].max():.2f} GB/s")
print(f"  Peak GFLOPS: {flops_df['gflops'].max():.2f}")
print(f"{'='*60}\n")
