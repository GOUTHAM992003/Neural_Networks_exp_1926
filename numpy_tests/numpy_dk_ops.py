import numpy as np
import pandas as pd
import os
import warnings
import time
warnings.filterwarnings('ignore')


D=10
R=10
C=10
size = D * R * C

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

    a = input_a.reshape(D, R, C).astype(np.float32)
    b = input_b.reshape(D, R, C).astype(np.float32)
    
    print("✓ Inputs loaded from CSV")

except (FileNotFoundError, ValueError, Exception) as e:
    print(f"⚠ Could not read inputs from CSV: {e}")
    print("   Generating RANDOM inputs instead...")
    np.random.seed(1926)
    a = np.random.randn(D, R, C).astype(np.float32)
    b = np.random.randn(D, R, C).astype(np.float32)
    print(f"✓ Generated random tensors [{D}, {R}, {C}]")
    
    # Save these generated inputs
    try:
        os.makedirs("benchmark_results/numpy", exist_ok=True)
        save_path = "benchmark_results/numpy/generated_inputs.csv"
        print(f"   Saving generated inputs to {save_path}...")
        
        a_flat = a.flatten()
        b_flat = b.flatten()
        indices = np.arange(size)
        
        df_gen = pd.DataFrame({
            'index': indices,
            'depth': indices // (R*C),
            'row': (indices // C) % R,
            'col': indices % C,
            'input_a': a_flat,
            'input_b': b_flat
        })
        df_gen.to_csv(save_path, index=False)
        print("   ✓ Inputs saved.")
    except Exception as save_err:
        print(f"   ⚠ Failed to save generated inputs: {save_err}")



a_pos = a + 0.1
b_pos = b + 0.1

# Timing setup
NUM_RUNS = 50
WARMUP_RUNS = 5
timings = {}

def time_operation(name, func, num_runs=NUM_RUNS, warmup=WARMUP_RUNS, num_elements=size, 
                   dtype_size=4, op_type='binary'):
    """
    Time an operation with multiple runs and calculate throughput metrics.
    
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
    
    # Actual timing
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func()
        end = time.perf_counter()
        times.append((end - start) * 1000)  # Convert to ms
    
    mean_time_ms = np.mean(times)
    mean_time_sec = mean_time_ms / 1000.0
    
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

print("Running NumPy benchmark with timing...")

# All operations with timing
# Binary ops (Default op_type='binary')
add_res = time_operation("add", lambda: a + b, op_type='binary')
sub_result = time_operation("sub", lambda: a - b, op_type='binary')
mul_result = time_operation("mul", lambda: a * b, op_type='binary')
div_result = time_operation("div", lambda: a / (b + 0.1), op_type='binary')

# Unary ops (op_type='unary')
square_result = time_operation("square", lambda: np.square(a), op_type='unary')
sqrt_result = time_operation("sqrt", lambda: np.sqrt(a_pos), op_type='unary')
neg_result = time_operation("neg", lambda: np.negative(a_pos), op_type='unary')
abs_result = time_operation("abs", lambda: np.abs(a), op_type='unary')
sign_result = time_operation("sign", lambda: np.sign(a), op_type='unary')
reciprocal_result = time_operation("reciprocal", lambda: np.reciprocal(a), op_type='unary')
pow2_result = time_operation("pow2", lambda: np.pow(a, 2), op_type='unary')

sin_result = time_operation("sin", lambda: np.sin(a), op_type='unary')
cos_result = time_operation("cos", lambda: np.cos(a), op_type='unary')
tan_result = time_operation("tan", lambda: np.tan(a), op_type='unary')

sinh_result = time_operation("sinh", lambda: np.sinh(a), op_type='unary')
cosh_result = time_operation("cosh", lambda: np.cosh(a), op_type='unary')
tanh_result = time_operation("tanh", lambda: np.tanh(a), op_type='unary')

asin_result = time_operation("asin", lambda: np.asin(a), op_type='unary')
acos_result = time_operation("acos", lambda: np.acos(a), op_type='unary')
atan_result = time_operation("atan", lambda: np.atan(a), op_type='unary')

asinh_result = time_operation("asinh", lambda: np.asinh(a), op_type='unary')
acosh_result = time_operation("acosh", lambda: np.acosh(a_pos + 0.9), op_type='unary')
atanh_result = time_operation("atanh", lambda: np.atanh(a), op_type='unary')

exp_result = time_operation("exp", lambda: np.exp(a), op_type='unary')
log_result = time_operation("log", lambda: np.log(a_pos), op_type='unary')
log2_result = time_operation("log2", lambda: np.log2(a_pos), op_type='unary')
log10_result = time_operation("log10", lambda: np.log10(a_pos), op_type='unary')

matmul_result = time_operation("matmul", lambda: np.matmul(a, np.transpose(b, (0, 2, 1))), op_type='matmul')

# Reductions (return scalars)
# Reductions (op_type='reduction')
sum_result = time_operation("sum_all", lambda: np.sum(a), op_type='reduction')
mean_result = time_operation("mean_all", lambda: np.mean(a), op_type='reduction')
max_result = time_operation("max_all", lambda: np.max(a), op_type='reduction')
min_result = time_operation("min_all", lambda: np.min(a), op_type='reduction')
var_result = time_operation("var_all", lambda: np.var(a), op_type='reduction')
std_result = time_operation("std_all", lambda: np.std(a), op_type='reduction')

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

# Chains (op_type='unary')
chain1 = time_operation("chain1", lambda: np.sin(np.cos(np.sqrt(square_result))), op_type='unary')
chain2 = time_operation("chain2", lambda: np.exp(np.log(np.log2(np.log10(a_pos) + 0.9) + 0.9)), op_type='unary')
chain3 = time_operation("chain3", lambda: np.sin(np.cos(np.tan(matmul_result))), op_type='unary')
chain4 = time_operation("chain4", lambda: np.square(np.log(np.tan(matmul_result + 0.5) + 0.9)), op_type='unary')
chain5 = time_operation("chain5", lambda: np.tanh(np.sin(np.exp(a))), op_type='unary')
chain6 = time_operation("chain6", lambda: np.log(np.exp(np.sqrt(a_pos))), op_type='unary')
chain7 = time_operation("chain7", lambda: np.cos(np.sin(np.tanh(np.log(a_pos)))), op_type='unary')
chain8 = time_operation("chain8", lambda: np.sqrt(np.square(np.exp(np.log(a_pos)))), op_type='unary')
chain9 = time_operation("chain9", lambda: np.log(np.reciprocal(np.sqrt(np.abs(np.sin(a) + 0.1)))), op_type='unary')
chain10 = time_operation("chain10", lambda: np.atan(np.sinh(np.tan(np.cos(np.sqrt(a_pos))))), op_type='unary')
chain11 = time_operation("chain11", lambda: np.sin(a + b), op_type='binary')
chain12 = time_operation("chain12", lambda: np.log(a_pos + b_pos), op_type='binary')
chain13 = time_operation("chain13", lambda: np.tanh(np.exp(a) + np.log(b_pos)), op_type='binary')
chain14 = time_operation("chain14", lambda: np.sin(np.cos(np.tan(np.exp(np.log(np.log10(a_pos)))))), op_type='unary')
chain15 = time_operation("chain15", lambda: np.exp(np.sin(np.cos(np.tanh(np.exp(np.log(a_pos + 0.9)))))), op_type='unary')

print(f"✓ Completed {len(timings)} operations")

# Save results CSV (same as before)
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

df = pd.DataFrame(columns=columns)


# Optimized DataFrame creation (Vectorized) - limit to 1000 rows max
limit = min(size, 1000)
indices = np.arange(limit)

data_dict = {
    'Idx': indices,
    'Batch': indices // (R*C),
    'Row': (indices // C) % R,
    'Column': indices % C,
    'TENSOR a': a.flatten()[:limit],
    'TENSOR b': b.flatten()[:limit],
    'Addition': add_res.flatten()[:limit],
    'Sub': sub_result.flatten()[:limit],
    'Mul': mul_result.flatten()[:limit],
    'Div': div_result.flatten()[:limit],
    'Square': square_result.flatten()[:limit],
    'Sqrt': sqrt_result.flatten()[:limit],
    'neg': neg_result.flatten()[:limit],
    'abs': abs_result.flatten()[:limit],
    'sign': sign_result.flatten()[:limit],
    'reciprocal': reciprocal_result.flatten()[:limit],
    'pow': pow2_result.flatten()[:limit],
    'sin': sin_result.flatten()[:limit],
    'cos': cos_result.flatten()[:limit],
    'tan': tan_result.flatten()[:limit],
    'sinh': sinh_result.flatten()[:limit],
    'cosh': cosh_result.flatten()[:limit],
    'tanh': tanh_result.flatten()[:limit],
    'asin': asin_result.flatten()[:limit],
    'acos': acos_result.flatten()[:limit],
    'atan': atan_result.flatten()[:limit],
    'asinh': asinh_result.flatten()[:limit],
    'acosh': acosh_result.flatten()[:limit],
    'atanh': atanh_result.flatten()[:limit],
    'exp': exp_result.flatten()[:limit],
    'log': log_result.flatten()[:limit],
    'log2': log2_result.flatten()[:limit],
    'log10': log10_result.flatten()[:limit],
    'scalar_add': scalar_add.flatten()[:limit],
    'scalar_mul': scalar_mul.flatten()[:limit],
    'scalar_div': scalar_div.flatten()[:limit],
    'reverse_sub': reverse_sub.flatten()[:limit],
    'sum_all': np.full(limit, sum_all),
    'mean_all': np.full(limit, mean_all),
    'max_all': np.full(limit, max_all),
    'min_all': np.full(limit, min_all),
    'var_all': np.full(limit, var_all),
    'std_all': np.full(limit, std_all),
    'chain1': chain1.flatten()[:limit],
    'chain2': chain2.flatten()[:limit],
    'chain3': chain3.flatten()[:limit],
    'chain4': chain4.flatten()[:limit],
    'chain5': chain5.flatten()[:limit],
    'chain6': chain6.flatten()[:limit],
    'chain7': chain7.flatten()[:limit],
    'chain8': chain8.flatten()[:limit],
    'chain9': chain9.flatten()[:limit],
    'chain10': chain10.flatten()[:limit],
    'chain11': chain11.flatten()[:limit],
    'chain12': chain12.flatten()[:limit],
    'chain13': chain13.flatten()[:limit],
    'chain14': chain14.flatten()[:limit],
    'chain15': chain15.flatten()[:limit]
}

df = pd.DataFrame(data_dict)

# Create output directory if it doesn't exist
os.makedirs("benchmark_results/numpy", exist_ok=True)

# 1. Save VALUES CSV (actual computation results - for correctness check)
df.to_csv("benchmark_results/numpy/numpy_values.csv", index=False)
print("✓ Values saved to: benchmark_results/numpy/numpy_values.csv")

# Create metrics DataFrames
all_metrics_df = pd.DataFrame([
    {'operation': op, **metrics}
    for op, metrics in timings.items()
])

# 2. Save TIMINGS CSV (timing only)
timings_df = all_metrics_df[['operation', 'mean_ms', 'min_ms', 'max_ms', 'std_ms']].copy()
timings_df.to_csv("benchmark_results/numpy/numpy_timings.csv", index=False)
print("✓ Timings saved to: benchmark_results/numpy/numpy_timings.csv")

# 3. Save THROUGHPUT CSV (throughput only)
throughput_df = all_metrics_df[['operation', 'throughput_elem_per_sec']].copy()
throughput_df.to_csv("benchmark_results/numpy/numpy_throughput.csv", index=False)
print("✓ Throughput saved to: benchmark_results/numpy/numpy_throughput.csv")

# 4. Save BANDWIDTH CSV (memory bandwidth only)
bandwidth_df = all_metrics_df[['operation', 'memory_bandwidth_gb_per_sec']].copy()
bandwidth_df.to_csv("benchmark_results/numpy/numpy_bandwidth.csv", index=False)
print("✓ Bandwidth saved to: benchmark_results/numpy/numpy_bandwidth.csv")

# 5. Save FLOPS CSV (computational performance - estimated)
# For simple ops: 1 FLOP per element, for complex ops: estimate based on operation type
flops_data = []
for op, metrics in timings.items():
    mean_time_sec = metrics['mean_ms'] / 1000.0
    
    # Estimate FLOPs based on operation type
    if op in ['add', 'sub', 'mul', 'div', 'square', 'sqrt', 'neg', 'abs', 'sign', 'reciprocal', 'pow2',
              'scalar_add', 'scalar_mul', 'scalar_div', 'reverse_sub']:
        flops_per_elem = 1
    elif op in ['sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh',
                'exp', 'log', 'log2', 'log10']:
        flops_per_elem = 5  # Trig/exp/log are expensive
    elif op == 'matmul':
        # Matrix multiply: 2*D*R*C FLOPs (each output element: C multiply-adds)
        flops_per_elem = 2 * C  # Approximately
    elif 'chain' in op:
        flops_per_elem = 10  # Chain operations have many ops
    else:
        flops_per_elem = 1
    
    total_flops = flops_per_elem * size
    gflops = (total_flops / mean_time_sec / 1e9) if mean_time_sec > 0 else 0
    
    flops_data.append({
        'operation': op,
        'gflops': gflops
    })

flops_df = pd.DataFrame(flops_data)
flops_df.to_csv("benchmark_results/numpy/numpy_flops.csv", index=False)
print("✓ FLOPS saved to: benchmark_results/numpy/numpy_flops.csv")

# Print summary
print(f"\n{'='*60}")
print(f"NumPy Benchmark Summary")
print(f"{'='*60}")
print(f"  Total operations: {len(timings)}")
print(f"  Fastest: {timings_df.loc[timings_df['mean_ms'].idxmin(), 'operation']} ({timings_df['mean_ms'].min():.4f} ms)")
print(f"  Slowest: {timings_df.loc[timings_df['mean_ms'].idxmax(), 'operation']} ({timings_df['mean_ms'].max():.4f} ms)")
print(f"  Peak throughput: {throughput_df['throughput_elem_per_sec'].max():.2e} elem/s")
print(f"  Peak bandwidth: {bandwidth_df['memory_bandwidth_gb_per_sec'].max():.2f} GB/s")
print(f"  Peak GFLOPS: {flops_df['gflops'].max():.2f}")
print(f"{'='*60}\n")
