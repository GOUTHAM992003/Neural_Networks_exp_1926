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

# Training size from gpt2_attn_fixed.cpp: batch=8, seq=1024, n_embd=384
B, T, C = 8, 1024, 384

for dev in ['cpu'] + (['cuda'] if torch.cuda.is_available() else []):
    x = torch.randn(B, T, C, dtype=torch.float32, device=dev)
    gamma = torch.ones(C, dtype=torch.float32, device=dev)
    beta = torch.zeros(C, dtype=torch.float32, device=dev)
    grad = torch.randn_like(x)

    label = "CPU" if dev == 'cpu' else "GPU"
    print(f"\n=== {label} LayerNorm + RMSNorm — [{B}, {T}, {C}] ===")
    print(f"Warmup: {WARMUP}, Timed: {N}")
    print("-" * 60)

    # LayerNorm forward
    print(f"layer_norm_forward (fp32):              {bench(lambda: F.layer_norm(x, [C], gamma, beta)):.4f} ms")

    # LayerNorm backward
    def ln_bwd():
        bx = x.clone().requires_grad_(True)
        y = F.layer_norm(bx, [C], gamma, beta)
        y.backward(grad)
    print(f"layer_norm_backward (fp32):             {bench(ln_bwd):.4f} ms  (includes forward)")

    # RMSNorm (PyTorch >= 2.4 has F.rms_norm, fallback for older)
    try:
        print(f"rms_norm_forward (fp32):                {bench(lambda: F.rms_norm(x, [C], gamma)):.4f} ms")
        def rms_bwd():
            bx = x.clone().requires_grad_(True)
            y = F.rms_norm(bx, [C], gamma)
            y.backward(grad)
        print(f"rms_norm_backward (fp32):               {bench(rms_bwd):.4f} ms  (includes forward)")
    except AttributeError:
        # Manual RMSNorm for older PyTorch
        def manual_rms_norm(x, gamma):
            rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1e-5)
            return x / rms * gamma
        print(f"rms_norm_forward (fp32, manual):        {bench(lambda: manual_rms_norm(x, gamma)):.4f} ms")
        def rms_bwd():
            bx = x.clone().requires_grad_(True)
            y = manual_rms_norm(bx, gamma)
            y.backward(grad)
        print(f"rms_norm_backward (fp32, manual):       {bench(rms_bwd):.4f} ms  (includes forward)")
