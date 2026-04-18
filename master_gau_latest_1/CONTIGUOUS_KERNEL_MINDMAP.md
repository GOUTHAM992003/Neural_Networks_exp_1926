# Contiguous Strided Copy Architecture — Visual Mindmap

## 🗺️ MAIN MINDMAP

```
╔═══════════════════════════════════════════════════════════════════════════╗
║         contiguous_strided_copy_cuda(src, dst, shape, strides)           ║
╚═══════════════════════════════════════════════════════════════════════════╝
                                     │
                ┌────────────────────┼────────────────────┐
                │                    │                    │
            STEP 1              STEP 2                STEP 3
            COALESCE            COPY               DISPATCH
            ────────            ────               ────────
            Merge dims          Local              4-way
            where aligned       arrays             decision
                                                   tree
                │                    │                    │
                └────────────────────┼────────────────────┘
                                     │
                    ┌────────────────┴────────────────┐
                    │                                 │
            ┌───────▼────────┐            ┌───────────▼─────────┐
            │ COALESCED      │            │ NOT FULLY           │
            │ [P,R,C]        │            │ CONTIGUOUS?         │
            │ Planes,Rows,   │            │                     │
            │ Columns        │            │ YES → Check Pattern │
            └────────┬───────┘            └─────────┬───────────┘
                     │                              │
                ┌────▼─────────────────┐           │
                │ is_fully_contiguous? │           │
                │ (P=1, R=1, C linear) │           │
                └────┬─────────┬───────┘           │
                     │         │                    │
                  YES│         │NO                  │
                     │         │                    │
            ┌────────▼──┐  ┌───▼──────────────────┐│
            │  PATH 3a  │  │  Check next pattern  ││
            │  ─────────│  │                      ││
            │ cudaD2D   │  └───┬──────────────────┘│
            │ Hardware  │      │                    │
            │ DMA       │      └─────────┬──────────┘
            │           │                │
            │ ⚡⚡⚡  │        ┌───────┴────────┬──────────┬─────────┐
            │ RETURN    │        │              │          │         │
            └───────────┘   2D Trans       3D Trans    Strided   Generic
                            Pattern        Pattern      Edge      Fallback
                               │              │         Case        │
                        ┌──────▼──┐    ┌──────▼──┐              ┌───▼────┐
                        │ PATH 3b │    │ PATH 3c │     SKIP     │ PATH 3d│
                        │ ────────│    │ ────────│   (no code)   │ ────────│
                        │ Tiled   │    │ Tiled 3D│              │FastDiv │
                        │ 2D      │    │ Batched │              │ Magic  │
                        │ Kernel  │    │ Kernel  │              │Numbers │
                        │         │    │         │              │        │
                        │ ⚡⚡   │    │ ⚡⚡    │              │ ⚡     │
                        │ RETURN  │    │ RETURN  │              │ RETURN │
                        └─────────┘    └─────────┘              └────────┘
```

---

## 📐 WHAT EACH PATH DOES

### **PATH 3a: Hardware DMA (cudaMemcpyAsync)**

```
INPUT TENSOR:  [Planes, Rows, Cols]  where ALL fully contiguous
               [P,      R,     C]

CONDITION: strides align → stride[0] = R*C, stride[1] = C, stride[2] = 1
           (linear memory layout, no jumps)

OPERATION:
┌─────────────────────────────────────┐
│  GPU DMA Engine (hardware offload)  │
│                                     │
│  src ────────────────────> dst     │
│   ↓                          ↑      │
│ GPU HBM              GPU HBM       │
│ (source buffer)      (dest buffer) │
│                                     │
│ Speed: ~1.5-2 TB/sec                │
│ CPU: Not involved (zero kernel)     │
└─────────────────────────────────────┘

EXAMPLE: [2 planes, 3 rows, 4 cols] linear
         8 × 3 × 4 = 96 elements
         → Copy all 96 sequentially via DMA

⚡⚡⚡ FASTEST PATH
```

---

### **PATH 3b: 2D Tiled Transpose**

```
INPUT:  [Rows, Cols]           OUTPUT: [Cols, Rows]
        [R,    C]                      [C,    R]

EXAMPLE VISUALIZATION:

INPUT MEMORY (Row-Major):      OUTPUT MEMORY (Row-Major):
┌──────────┬──────────┐        ┌──────────┬──────────┐
│ [0,0][0,1][0,2][0,3]│        │ [0,0][1,0][2,0][3,0]│
│ [1,0][1,1][1,2][1,3]│        │ [0,1][1,1][2,1][3,1]│
│ [2,0][2,1][2,2][2,3]│        │ [0,2][1,2][2,2][3,2]│
│ [3,0][3,1][3,2][3,3]│        │ [0,3][1,3][2,3][3,3]│
└──────────┴──────────┘        └──────────┴──────────┘
 4 rows × 4 cols              4 cols × 4 rows
      ↓ TRANSPOSE ↑

KERNEL OPTIMIZATION:
┌──────────────────────────────────────────────────────────┐
│ Block = 32×8 threads                                     │
│ Tile = 32×32 elements                                    │
│ Shared Memory = [32][33] (extra col for bank conflicts) │
│                                                          │
│ Phase 1: Coalesced READ from src[rows×cols]             │
│          Thread (tx,ty) reads src[y+j][x]               │
│          Writes to shared[ty+j][tx]                     │
│                                                          │
│ __syncthreads()                                          │
│                                                          │
│ Phase 2: Coalesced WRITE to dst[cols×rows]              │
│          Thread (tx,ty) reads shared[tx][ty+j]          │
│          Writes to dst[y+j][x]  (transposed!)           │
│                                                          │
│ Grid = ((cols+31)/32, (rows+31)/32)                     │
└──────────────────────────────────────────────────────────┘

CONDITION: ndim==2 AND strides[0]==1 AND strides[1]==rows
           (aka: strides[1] == dim[0])

⚡⚡ FAST (~2× speedup vs naive)
```

---

### **PATH 3c: 3D Batched Transpose**

```
INPUT:  [Planes, Rows, Cols]    OUTPUT: [Planes, Cols, Rows]
        [P,      R,     C]               [P,      C,     R]

PROCESS: For EACH plane independently, transpose last 2 dims
         (Row,Col) → (Col,Row) within that plane


EXAMPLE VISUALIZATION (3 planes):

PLANE 0:
Input [R,C]           Output [C,R]
┌────┬────┐           ┌────┬────┐
│[0,0]│[0,1]│   →    │[0,0]│[1,0]│
│[1,0]│[1,1]│        │[0,1]│[1,1]│
└────┴────┘           └────┴────┘

PLANE 1:
Input [R,C]           Output [C,R]
┌────┬────┐           ┌────┬────┐
│[0,0]│[0,1]│   →    │[0,0]│[1,0]│
│[1,0]│[1,1]│        │[0,1]│[1,1]│
└────┴────┘           └────┴────┘

PLANE 2:
Input [R,C]           Output [C,R]
┌────┬────┐           ┌────┬────┐
│[0,0]│[0,1]│   →    │[0,0]│[1,0]│
│[1,0]│[1,1]│        │[0,1]│[1,1]│
└────┴────┘           └────┴────┘


KERNEL OPTIMIZATION:
┌──────────────────────────────────────────────────────────┐
│ Block = 32×8 threads                                     │
│ Tile = 32×32 elements PER PLANE                          │
│ Shared Memory = [32][33] (bank conflict avoidance)       │
│                                                          │
│ Grid = ((cols+31)/32, (rows+31)/32, planes)             │
│         ↑               ↑                    ↑            │
│      tile cols     tile rows           which plane       │
│                                                          │
│ blockIdx.z = which plane (0 to P-1)                     │
│ Each block handles ONE plane's 32×32 tile               │
│ ALL planes processed in PARALLEL (3D grid!)              │
│                                                          │
│ Phase 1: Coalesced READ from src[plane][rows][cols]     │
│ Phase 2: Coalesced WRITE to dst[plane][cols][rows]      │
└──────────────────────────────────────────────────────────┘

CONDITION: ndim==3 AND strides[2]==1 AND strides[1]==cols 
                   AND strides[0]==rows*cols
           (aka: last 2 dims form transpose pattern)

⚡⚡ FAST (parallelizes across planes)
```

---

### **PATH 3d: Generic Fallback (FastDivmod)**

```
INPUT: [P, R, C] with ANY stride pattern
       (non-contiguous, non-transpose, irregular strides)

CONDITION: Everything that didn't match 3a, 3b, or 3c

OPERATION:
┌──────────────────────────────────────────────────────────┐
│ GENERIC STRIDED COPY with FastDivmod                     │
│                                                          │
│ For each element index i in [0, total_elems):            │
│   1. Compute multi-dimensional index from linear i       │
│      Using FastDivmod (magic number ÷ instead of ÷)      │
│      Cost: 6 cycles vs 40 cycles per div                 │
│                                                          │
│   2. Apply strides to find src address:                  │
│      elem_offset = storage_offset                        │
│      FOR each dimension d:                               │
│        elem_offset += idx[d] * stride[d]                 │
│                                                          │
│   3. Convert to byte address and copy:                   │
│      src_byte = elem_offset * elem_size                  │
│      dst_byte = i * elem_size                            │
│      dst[dst_byte] = src[src_byte]                       │
│                                                          │
│ Block = 256 threads                                      │
│ Grid = (total_elems / 256) blocks                        │
│ Process 4 elements per thread (unrolled loop)            │
└──────────────────────────────────────────────────────────┘

EXAMPLE: Weird reshape with non-aligned strides
         Input shape [2, 3, 5] with strides [17, 7, 1]
         → Can't match any pattern above
         → Use 3d: FastDivmod each element's position

⚡ OKAY (covers all edge cases, but slower)
```

---

## 🎯 DECISION TREE (Clean Logic)

```
                        INPUT [P,R,C]
                             │
                    ┌────────┴────────┐
                    │                 │
        ┌──────────▼──────────┐   ┌─────▼─────────┐
        │ Fully Contiguous?   │   │ After         │
        │ (P*R*C linear)      │   │ Coalescing    │
        │ strides=[RC,C,1]    │   │               │
        └──────┬──────┬───────┘   └─────┬─────────┘
               │      │                 │
            YES│      │NO               │
               │      │                 │
        ┌──────▼──┐   │          ┌──────▼────────┐
        │ 3a: DMA │   │          │ Pattern Test  │
        │         │   │          │               │
        │⚡⚡⚡ │   │          └──────┬────────┘
        │ DONE    │   │                │
        └─────────┘   │          ┌─────┴──────────────┬──────────────┐
                      │          │                    │              │
                   ┌──▼─────────┐│                    │              │
                   │ Is 2D Tr   ││ Is 3D Batched      │ Else: Weird
                   │ (ndim=2)?  ││ (ndim=3)?          │ Stride Pattern
                   └──┬─────┬───┘│                    │              │
                    YES│     │NO  │ YES               │ YES          │ YES
                      │     │     │ │                 │              │
                   ┌──▼─┐  │  ┌──▼─┴──┐          ┌───▼──┐       ┌────▼────┐
                   │ 3b │  │  │  3c   │          │3d-No │       │ 3d-Fall │
                   │Tiled├──┘  │ Tiled│          │Code  │       │ Back    │
                   │2D   │     │3D    │          │(skip)│       │ FastDiv │
                   │⚡⚡ │     │⚡⚡  │          │      │       │         │
                   │DONE │     │ DONE │          └──────┘       │⚡       │
                   └─────┘     └──────┘                         │ RETURN │
                                                                 └────────┘
```

---

## 📊 PERFORMANCE SUMMARY

```
╔════════════════════════════════════════════════════════════════╗
║            PATH PERFORMANCE & USE CASES                       ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║ 3a: Hardware DMA                                              ║
║ ├─ Speed: ⚡⚡⚡ (1.5-2 TB/s)                                  ║
║ ├─ Use: Linear buffers, embeddings, output features          ║
║ └─ Cost: Zero kernel overhead                                ║
║                                                                ║
║ 3b: 2D Tiled Transpose                                        ║
║ ├─ Speed: ⚡⚡ (400-800 GB/s)                                 ║
║ ├─ Use: Matrix transpose, 2D permutations                    ║
║ └─ Cost: Kernel launch + shared memory tiling                ║
║                                                                ║
║ 3c: 3D Batched Transpose                                      ║
║ ├─ Speed: ⚡⚡ (400-800 GB/s)                                 ║
║ ├─ Use: Batched 2D transposes, attention heads               ║
║ └─ Cost: 3D grid parallelization                             ║
║                                                                ║
║ 3d: Generic FastDivmod Fallback                               ║
║ ├─ Speed: ⚡ (variable, 100-300 GB/s)                       ║
║ ├─ Use: Non-aligned, irregular strides                       ║
║ └─ Cost: Divmod magic numbers (~6 cycles each)               ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

---

## 🧠 SIMPLE EXAMPLE: Training Forward Pass

```
SHAPE: [Planes=2, Rows=1024, Cols=768]

Step 1: Linear projection (contiguous output)
        → OUTPUT: [2, 1024, 768] with strides [786432, 768, 1]
        → Pattern: FULLY CONTIGUOUS
        → 🎯 PATH 3a: cudaMemcpyAsync (~0.5 ms, hardware DMA)

Step 2: Reshape for attention (still contiguous)
        → OUTPUT: [2, 1024, 12, 64] reshaped to [2, 1024, 768]
        → Pattern: FULLY CONTIGUOUS
        → 🎯 PATH 3a: cudaMemcpyAsync (~0.5 ms)

Step 3: Permute attention heads (transpose in last 2 dims)
        → INPUT: [2, 1024, 768] (reshape done)
        → OUTPUT: [2, 64, 1024, 12] → view as [2, 64, 12288]
        → After coalescing: [2, 64, 12288] with transpose pattern
        → 🎯 PATH 3c: Batched 3D tiled kernel (~1-2 ms, parallel)

Step 4: Weird view + non-aligned stride (edge case)
        → Pattern: Non-standard strides, can't match above
        → 🎯 PATH 3d: Generic FastDivmod (~2-3 ms, catch-all)
```

---

Done! All 4 paths explained with **[Planes, Rows, Cols]** consistently. ✅
