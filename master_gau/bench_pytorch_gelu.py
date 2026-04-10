import torch, torch.nn.functional as F, time

torch.manual_seed(1337)
WARMUP, N = 5, 100

def bench(fn):
    for _ in range(WARMUP): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    s = time.perf_counter()
    for _ in range(N): fn()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    return (time.perf_counter() - s) / N * 1000

print(f"PyTorch {torch.__version__} | Threads: {torch.get_num_threads()} | CUDA: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")

for dev in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
    x = torch.randn(8, 1024, 384, dtype=torch.float32, device=dev)
    bias = torch.randn(384, dtype=torch.float32, device=dev)
    grad = torch.randn_like(x)
    bx = x.clone().requires_grad_(True)

    label = "CPU" if dev == 'cpu' else "GPU"
    print(f"\n=== {label} Forward+Backward GeLU ===")
    print(f"Shape: [8, 1024, 384] | Warmup: {WARMUP}, Timed: {N}")
    print("-" * 60)

    print(f"gelu_forward (fp32):                    {bench(lambda: F.gelu(x, approximate='tanh')):.4f} ms")
    print(f"fused_bias_gelu_forward (fp32):         {bench(lambda: F.gelu(x + bias, approximate='tanh')):.4f} ms  (not fused)")

    # Backward: compute gelu, then backward
    def gelu_bwd():
        bx_ = x.clone().requires_grad_(True)
        y = F.gelu(bx_, approximate='tanh')
        y.backward(grad)
    def bias_gelu_bwd():
        bx_ = x.clone().requires_grad_(True)
        y = F.gelu(bx_ + bias, approximate='tanh')
        y.backward(grad)

    print(f"gelu_backward (fp32):                   {bench(gelu_bwd):.4f} ms  (includes forward)")
    print(f"fused_bias_gelu_backward (fp32):        {bench(bias_gelu_bwd):.4f} ms  (includes forward)")
