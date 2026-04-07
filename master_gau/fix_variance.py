#!/usr/bin/env python3
path = "/home/blu-bridge016/Downloads/Neural_Networks_exp_1926/master_gau/src/UnaryOps/cuda/ReductionImplGPU.cu"
with open(path, 'r') as f:
    lines = f.readlines()

# Delete lines 363 to 493 (1-indexed) which are orphaned old variance code
# 0-indexed: 362 to 492 inclusive
del lines[362:493]

with open(path, 'w') as f:
    f.writelines(lines)
print(f"Deleted orphaned old variance code (lines 363-493). File now has {len(lines)} lines.")
