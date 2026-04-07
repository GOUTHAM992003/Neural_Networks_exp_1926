import torch
import time

torch.set_num_threads(28)

print("Actual PyTorch Outer Reduction Benchmark: [1000000, C] -> [1, C]")
print("--------------------------------------------------------------------------------")
print(f"{'Output Slots':<15} {'PyTorch Actual (ms)':<20}")
print("--------------------------------------------------------------------------------")

for c in range(28, 0, -1):
    t = torch.ones(1000000, c, dtype=torch.float32)
    
    # Warmup
    for _ in range(5):
        torch.sum(t, dim=0)
        
    iters = 10
    start = time.perf_counter()
    for _ in range(iters):
        torch.sum(t, dim=0)
    end = time.perf_counter()
    
    time_ms = ((end - start) / iters) * 1000
    print(f"{c:<15} {time_ms:<20.2f}")
