import torch
import time

torch.manual_seed(1337)
x = torch.randn((8, 1024, 384), dtype=torch.float32, device='cuda')

# Warmup
for _ in range(50):
    res = torch.sum(x, dim=0)

torch.cuda.synchronize()

# Benchmark
iterations = 1000
start = time.perf_counter()
for _ in range(iterations):
    res = torch.sum(x, dim=0)

torch.cuda.synchronize()
end = time.perf_counter()

print(f"[PyTorch] Average Time over {iterations} ops: {(end - start) / iterations:.8f} seconds", flush=True)
