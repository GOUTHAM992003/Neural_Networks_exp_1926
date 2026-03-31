#!/usr/bin/env python3
"""
COMPREHENSIVE Mean Benchmark: PyTorch baseline
All dtypes × All reduction paths × All sizes × DL apps × Edge cases
"""
import torch
import numpy as np
import time

torch.set_num_threads(28)

def bench(fn, warmup=5, iters=30):
    for _ in range(warmup): fn()
    times = []
    for _ in range(iters):
        s = time.perf_counter_ns(); r = fn(); e = time.perf_counter_ns()
        times.append((e-s)/1000)
    times.sort()
    return times[len(times)//2], r

def rel_err(got, expected):
    if expected == 0: return 0 if got == 0 else float('inf')
    if np.isinf(got) or np.isnan(got): return float('inf')
    return abs(got - expected) / abs(expected)

def gt_mean(data_np, axis=None):
    return float(np.mean(data_np.astype(np.float64), axis=axis).flatten()[0] if axis is not None
                 else np.mean(data_np.astype(np.float64)))

W = 120
def header(t): print(f"\n{'='*W}\n  {t}\n{'='*W}")
def sub(t): print(f"\n--- {t} ---")

print(f"{'='*W}")
print(f"  COMPREHENSIVE REGULAR MEAN BENCHMARK — PyTorch {torch.__version__}")
print(f"  CPU: {torch.get_num_threads()} threads")
print(f"{'='*W}")

DTYPES = [
    (torch.float32, np.float32, "fp32"),
    (torch.float64, np.float64, "fp64"),
    (torch.float16, np.float16, "fp16"),
    (torch.bfloat16, np.float32, "bf16"),  # np doesn't have bf16
]

# ============================================================
header("1. INNERCONTIGUOUS (reduce last dim) — ALL DTYPES × ALL SIZES")
# ============================================================
print(f"{'Shape':>22s} {'Dtype':>6s} {'Time(μs)':>10s} {'rel_err':>12s} {'Strategy':>12s}")
print("-"*70)

shapes_inner = [
    # Tiny (< GRAIN_SIZE) → Sequential
    ((8, 16), "tiny"),
    ((32, 64), "tiny"),
    ((10, 100), "small"),
    # Medium → Strategy 1
    ((32, 768), "layernorm"),
    ((32, 4096), "large-LN"),
    ((256, 4096), "batch-feat"),
    ((4096, 768), "seq-LN"),
    ((1000, 10000), "medium"),
    # Large → bandwidth-bound
    ((2048, 50176), "spatial"),
    ((49152, 128), "many-out"),
    ((10, 1000000), "wide"),
    ((1000000, 10), "tall"),
]

for (r, c), label in shapes_inner:
    numel = r * c
    strategy = "Sequential" if numel < 32768 else ("Strat1" if r >= 28 else "Strat2")
    for tdt, npdt, dname in DTYPES:
        np.random.seed(42)
        data = np.random.randn(r, c).astype(npdt)
        gt = gt_mean(data, axis=-1)
        t = torch.from_numpy(data.astype(npdt)).to(tdt)
        us, result = bench(lambda: torch.mean(t, dim=-1))
        err = rel_err(float(result.flatten()[0]), gt)
        print(f"  {f'({r},{c})':>20s} {dname:>6s} {us:>10.0f} {err:>12.2e} {strategy:>12s}")

# ============================================================
header("2. OUTERCONTIGUOUS (reduce first dim) — ALL DTYPES × ALL SIZES")
# ============================================================
print(f"{'Shape':>22s} {'Dtype':>6s} {'Time(μs)':>10s} {'rel_err':>12s}")
print("-"*60)

shapes_outer = [
    ((16, 64), "tiny"),
    ((100, 256), "small"),
    ((1000, 256), "medium"),
    ((10000, 100), "tall"),
    ((100, 10000), "wide"),
    ((50176, 2048), "spatial"),
    ((1000, 50000), "large"),
]

for (r, c), label in shapes_outer:
    for tdt, npdt, dname in DTYPES:
        np.random.seed(42)
        data = np.random.randn(r, c).astype(npdt)
        gt = gt_mean(data, axis=0)
        t = torch.from_numpy(data.astype(npdt)).to(tdt)
        us, result = bench(lambda: torch.mean(t, dim=0))
        err = rel_err(float(result.flatten()[0]), gt)
        print(f"  {f'({r},{c})':>20s} {dname:>6s} {us:>10.0f} {err:>12.2e}")

# ============================================================
header("3. GENERIC (reduce mixed dims) — ALL DTYPES")
# ============================================================
print(f"{'Shape':>22s} {'Dims':>12s} {'Dtype':>6s} {'Time(μs)':>10s} {'rel_err':>12s}")
print("-"*70)

generic_cases = [
    ((100, 200, 50), (0, 2), "3D XZ"),
    ((100, 200, 50), (0,), "3D X"),
    ((100, 200, 50), (1, 2), "3D YZ"),
    ((32, 128, 768), (0, 2), "transformer"),
    ((32, 128, 768), (1, 2), "batch-reduce"),
    ((16, 64, 32, 32), (0, 2), "4D conv-XH"),
    ((16, 64, 32, 32), (2, 3), "4D spatial"),
    ((16, 64, 32, 32), (1, 3), "4D CW"),
    ((8, 32, 16, 16, 8), (0, 2, 4), "5D mixed"),
]

for shape, dims, label in generic_cases:
    for tdt, npdt, dname in [DTYPES[0], DTYPES[1]]:  # fp32 + fp64
        np.random.seed(42)
        data = np.random.randn(*shape).astype(npdt)
        gt_val = float(np.mean(data.astype(np.float64), axis=dims).flatten()[0])
        t = torch.from_numpy(data.astype(npdt)).to(tdt)
        us, result = bench(lambda: torch.mean(t, dim=dims))
        err = rel_err(float(result.flatten()[0]), gt_val)
        print(f"  {str(shape):>22s} {str(dims):>12s} {dname:>6s} {us:>10.0f} {err:>12.2e}")

# ============================================================
header("4. FULL REDUCTION (scalar output) — ALL DTYPES × ALL SIZES")
# ============================================================
print(f"{'Size':>12s} {'Dtype':>6s} {'Time(μs)':>10s} {'rel_err':>12s} {'Strategy':>10s}")
print("-"*55)

for size in [100, 1000, 10000, 100000, 1000000, 10000000, 50000000]:
    strategy = "Sequential" if size < 32768 else "Strat2"
    for tdt, npdt, dname in DTYPES:
        np.random.seed(42)
        data = np.random.randn(size).astype(npdt)
        gt = float(np.mean(data.astype(np.float64)))
        t = torch.from_numpy(data.astype(npdt)).to(tdt)
        us, result = bench(lambda: torch.mean(t))
        err = rel_err(float(result), gt)
        print(f"  {size:>10d} {dname:>6s} {us:>10.0f} {err:>12.2e} {strategy:>10s}")

# ============================================================
header("5. DEEP LEARNING APPLICATION SHAPES")
# ============================================================
print(f"{'Application':>30s} {'Shape':>28s} {'Dim':>12s} {'Time(μs)':>10s}")
print("-"*85)

dl_apps = [
    ("LayerNorm-small", (32, 768), (-1,)),
    ("LayerNorm-large", (32, 4096), (-1,)),
    ("BatchNorm-2D", (32, 256, 14, 14), (0, 2, 3)),
    ("BatchNorm-1D", (32, 256, 100), (0, 2)),
    ("Attention-QK", (32, 12, 128, 128), (-1,)),
    ("Attention-head-mean", (32, 12, 128, 128), (1,)),
    ("Seq-LayerNorm", (32, 512, 768), (-1,)),
    ("Feature-mean", (64, 2048), (0,)),
    ("Spatial-pool", (32, 512, 7, 7), (2, 3)),
    ("Global-avg-pool", (32, 2048, 7, 7), (2, 3)),
    ("Loss-reduction", (32, 10000), (-1,)),
    ("Loss-full", (32, 10000), None),
    ("Token-mean", (32, 512, 768), (1,)),
    ("Channel-mean", (32, 256, 56, 56), (1,)),
    ("Embedding-mean", (32, 128, 300), (-1,)),
    ("BERT-pool", (16, 512, 1024), (1,)),
    ("GPT-logits", (8, 2048, 50257), (-1,)),
    ("ViT-patch", (32, 197, 768), (1,)),
    ("ResNet-feat", (64, 2048), (-1,)),
]

for name, shape, dim in dl_apps:
    t = torch.randn(*shape, dtype=torch.float32)
    if dim is None:
        us, _ = bench(lambda: torch.mean(t))
    else:
        us, _ = bench(lambda: torch.mean(t, dim=dim))
    print(f"  {name:>28s} {str(shape):>28s} {str(dim):>12s} {us:>10.0f}")

# ============================================================
header("6. PRECISION — STRESS TESTS")
# ============================================================
print(f"{'Dataset':>25s} {'N':>10s} {'Dtype':>6s} {'rel_err':>12s} {'Note':>20s}")
print("-"*80)

def prec_test(desc, data_fn, N, note=""):
    for tdt, npdt, dname in [(torch.float32, np.float32, "fp32"), (torch.float64, np.float64, "fp64")]:
        data = data_fn(N, npdt)
        gt = float(np.mean(data.astype(np.float64)))
        t = torch.from_numpy(data).to(tdt)
        r = float(torch.mean(t))
        err = rel_err(r, gt)
        print(f"  {desc:>23s} {N:>10d} {dname:>6s} {err:>12.2e} {note:>20s}")

np.random.seed(42)
for N in [1000, 100000, 10000000]:
    prec_test("Uniform[-1,1]", lambda n,d: np.random.uniform(-1,1,n).astype(d), N)
    prec_test("Gaussian N(0,1)", lambda n,d: np.random.randn(n).astype(d), N)
    prec_test("Large mean+tiny var", lambda n,d: (1e6+np.random.randn(n)*1e-3).astype(d), N, "catastrophic?")
    prec_test("Mixed [1e-6,1e6]", lambda n,d: np.random.uniform(1e-6,1e6,n).astype(d), N)
    prec_test("Near FLT_MAX", lambda n,d: np.random.uniform(1e37,3e38,n).astype(d), N, "overflow?")
    prec_test("Subnormals", lambda n,d: np.random.uniform(1e-45,1e-38,n).astype(d), N, "denorm trap?")
    prec_test("Alternating ±1e30", lambda n,d: np.array([1e30 if i%2==0 else -1e30 for i in range(n)]).astype(d), N, "cancellation?")
    prec_test("One outlier", lambda n,d: np.concatenate([[1e10], np.ones(n-1)]).astype(d), N, "outlier?")
    print()

# ============================================================
header("7. EDGE CASES")
# ============================================================
cases = [
    ("Empty-like (1,0)", lambda: torch.mean(torch.randn(1,0,dtype=torch.float32), dim=-1)),
    ("Single element", lambda: torch.mean(torch.tensor([42.0]))),
    ("All same value", lambda: torch.mean(torch.full((10000,), 3.14, dtype=torch.float32))),
    ("All zeros", lambda: torch.mean(torch.zeros(10000, dtype=torch.float32))),
    ("All NaN", lambda: torch.mean(torch.full((100,), float('nan'), dtype=torch.float32))),
    ("Inf values", lambda: torch.mean(torch.tensor([1.0, float('inf'), 2.0]))),
    ("-Inf values", lambda: torch.mean(torch.tensor([1.0, float('-inf'), 2.0]))),
    ("Inf + -Inf", lambda: torch.mean(torch.tensor([float('inf'), float('-inf')]))),
    ("Keepdim=True", lambda: torch.mean(torch.randn(3,4), dim=-1, keepdim=True)),
]

for name, fn in cases:
    try:
        r = fn()
        print(f"  {name:>25s} → {r}")
    except Exception as e:
        print(f"  {name:>25s} → ERROR: {e}")

print(f"\n{'='*W}\nBENCHMARK COMPLETE\n{'='*W}")
