# Nsight Compute (ncu) — Complete Deep Dive

## 1. Nsight Systems vs Nsight Compute: The Core Difference

From your **report23** Nsight Systems output, you can see a table like this:

| % Time | Kernel Name | Instances |
|--------|-------------|-----------|
| 15.4% | `mem_efficient_bwd_unified_kernel_exp12<64, true>` | 3,840 |
| 14.1% | `cutlass_80_tensorop_s1688gemm_128x64_16x6_nn_align4` | 17,280 |
| 8.1% | `fused_attn_forward_kernel_tc_sm89<64,64,64,1>` | 4,320 |
| ... | ... | ... |

**Nsight Systems** tells you:

- WHICH kernels ran
- HOW LONG each kernel took (wall-clock time)
- HOW MANY times each was launched
- The timeline order of all kernels, memcpys, API calls

**Nsight Systems does NOT tell you:**

- WHY a kernel is slow
- Whether your kernel is memory-bound or compute-bound
- What % of the GPU's compute or memory bandwidth you're actually using
- Where the bottleneck is INSIDE the kernel (warp stalls, shared memory bank conflicts, register pressure, occupancy issues, etc.)

**That is exactly what Nsight Compute does.**

Nsight Compute takes ONE kernel launch (or a few kernel launches you select), and does an extremely deep analysis of everything that happened during that kernel's execution — down to individual hardware performance counters.

---

## 2. How Hardware Performance Counters Work (Why Multi-Pass Exists)

### What are hardware performance counters?

Every NVIDIA GPU has a set of **hardware performance counters** (also called "PM counters" — Performance Monitor counters). These are physical circuits built into the GPU silicon that count specific events during kernel execution.

Examples of things counters can count:

- Number of global memory load instructions executed
- Number of shared memory bank conflicts
- Number of L2 cache hits vs misses
- Number of warp stalls due to memory dependency
- Number of FP32 instructions executed
- Number of tensor core instructions executed
- Number of registers used per thread

### The problem: limited counter slots

Your **Ada Lovelace RTX 6000 (SM89)** GPU has a **fixed, limited number of counter slots** that can be active simultaneously. The exact number varies by GPU generation, but it's roughly in the range of **4 to 8 counters at a time** (this is a hardware limitation of the Performance Monitor unit on each SM).

But a **full** Nsight Compute profile wants to collect **~7,794 metrics** (you saw this number from `ncu --list-sets`).

**You cannot read 7,794 counters simultaneously when the hardware only has ~4-8 counter slots.**

### The solution: replay the kernel multiple times

This is the key insight. Nsight Compute's approach:

1. Launch your kernel the 1st time → configure counter slots to read counter group A → collect those ~4-8 counters
2. **Replay** the exact same kernel launch (same grid, same block, same inputs) → configure counter slots to read counter group B → collect the next ~4-8 counters
3. Replay again → counter group C → collect more counters
4. ... repeat until ALL required counters have been collected

**Each such replay is called a "pass".**

The number of passes depends on how many metrics you requested:

- **basic** set: ~213 metrics → roughly **10-30 passes** per kernel
- **detailed** set: ~906 metrics → roughly **30-80 passes** per kernel
- **full** set: ~7,794 metrics → roughly **100-300+ passes** per kernel

> [!IMPORTANT]
> **This is why Nsight Compute takes so long.** If your `mem_efficient_bwd_unified_kernel_exp12<64, true>` kernel normally takes ~4.3ms per launch, and a "full" profile requires ~200 passes, that single kernel launch alone would take 4.3ms × 200 = **~860ms** just for one invocation. And if you profile multiple invocations or multiple kernels, it multiplies further.

---

## 3. The Replay Mechanism in Detail

### 3.1 Kernel Replay (Default: `--replay-mode kernel`)

This is the default and most commonly used mode.

**How it works step by step:**

1. Your application starts executing normally
2. When a kernel launch is intercepted by ncu (and it matches your filter):
   - ncu **saves a snapshot** of all GPU memory that the kernel will read from (inputs)
   - ncu launches the kernel with counter group 1 → records counters
   - ncu **restores** the GPU memory to the saved snapshot (because the kernel may have written outputs that changed memory)
   - ncu launches the kernel again with counter group 2 → records counters
   - ncu restores memory again
   - ... repeats for all counter groups
   - After the final pass, ncu lets the kernel run **one more time for real** (so the application sees correct output and can continue)
3. Your application continues to the next kernel launch

**Key points about kernel replay:**

- The application only runs **once**. The entire training loop does NOT restart.
- Only the specific kernel is replayed internally — the rest of the application does not know this is happening.
- The memory save/restore ensures deterministic results across passes (same inputs → same counter values).
- This save/restore itself takes time (reading back all GPU memory the kernel touches), which adds more overhead on top of the replay time.

---

## 4. Answering Your Direct Questions

### Q: If I profile `snippet_runner` with ncu, will it give details of ALL kernels or only one?

**It depends entirely on the flags you pass.**

**Default behavior (no `-k` filter):**

```bash
ncu ./snippet_runner
```

This will profile **EVERY single kernel launch** in your application. From your report23, you have **228,832 kernel launches**. Each one would be replayed multiple times. This would take **days** (literally).

**With a kernel filter (`-k`):**

```bash
ncu -k "mem_efficient_bwd_unified_kernel_exp12" ./snippet_runner
```

This will **only** stop and profile kernels whose name matches. All other kernels (cutlass gemms, gelu, layernorm, etc.) will run normally at full speed with zero overhead. Only when a `mem_efficient_bwd_unified_kernel_exp12` launch is encountered will ncu pause, do the multi-pass replay, and collect data.

**With a kernel filter AND launch count (`-k` + `-c`):**

```bash
ncu -k "mem_efficient_bwd_unified_kernel_exp12" -c 1 ./snippet_runner
```

This will profile **only the FIRST matching launch** of that kernel, collect all the data, and then let the rest of the application run without profiling. The `--kill` flag can even terminate the app after profiling is done:

```bash
ncu -k "mem_efficient_bwd_unified_kernel_exp12" -c 1 --kill yes ./snippet_runner
```

### Q: Nsight Systems gives data for one pass, right?

**Correct**, but the word "pass" means something different here:

- **Nsight Systems "one pass"** = it runs your application **once**, start to finish, and captures the timeline of everything that happened. It uses lightweight tracing (timestamps, not counter replay), so there is almost no slowdown. It captures data for ALL kernels in that single run.
- **Nsight Compute "multiple passes"** = for EACH kernel it profiles, it **replays that kernel** multiple times to collect different hardware counters each time. But the application itself only runs once (in kernel replay mode).

### Q: How many passes does Nsight Compute do?

The number of passes per kernel depends on the **section set** you choose:

| Set | Sections Included | Estimated Metrics | Approx. Passes per Kernel |
|-----|-------------------|-------------------|---------------------------|
| **basic** (default) | LaunchStats, Occupancy, SpeedOfLight, WorkloadDistribution | 213 | ~10-30 |
| **detailed** | + ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, SourceCounters | 906 | ~30-80 |
| **full** | + InstructionStats, SchedulerStats, WarpStateStats, RooflineChart, PmSampling, etc. | 7,794 | ~100-300+ |

---

## 5. What Each Section Set Tells You

### basic (default)

- **LaunchStats**: Grid size, block size, registers per thread, shared memory per block, achieved occupancy
- **Occupancy**: Theoretical vs achieved occupancy, what limits occupancy (registers, shared mem, block size)
- **SpeedOfLight (SOL)**: The most important section — tells you what % of the GPU's peak compute throughput and peak memory bandwidth your kernel achieved. Immediately tells you if you're compute-bound or memory-bound.
- **WorkloadDistribution**: How evenly the work is distributed across SMs

### detailed (adds to basic)

- **ComputeWorkloadAnalysis**: Breakdown of executed instruction types (FP32, FP16, INT, tensor core, etc.)
- **MemoryWorkloadAnalysis**: L1/L2 cache hit rates, shared memory throughput, global memory throughput, bank conflicts
- **SourceCounters**: If you compiled with `-lineinfo`, maps counters back to specific lines of your CUDA source code

### full (adds to detailed)

- **InstructionStats**: Per-instruction-type breakdown of pipeline utilization
- **SchedulerStats**: Warp scheduler issue efficiency, eligible warps per cycle
- **WarpStateStats**: Why warps are stalled (memory dependency, execution dependency, synchronization barrier, etc.)
- **RooflineChart**: The roofline model plot showing where your kernel sits relative to the compute and memory rooflines
- **PmSampling**: Statistical sampling of warp states over time

---

## 6. Why Nsight Compute Takes So Long — Full Breakdown

For your specific kernel `mem_efficient_bwd_unified_kernel_exp12<64, true>` which takes ~4.3ms per launch:

| Factor | Time Added | Explanation |
|--------|-----------|-------------|
| Kernel re-execution | 4.3ms × N passes | The kernel runs N times (one per counter group) |
| Memory save before each pass | ~2-10ms per pass | ncu must snapshot all GPU memory regions the kernel reads/writes |
| Memory restore between passes | ~2-10ms per pass | ncu must restore all GPU memory to the saved state |
| Counter configuration | ~0.1-1ms per pass | Reprogramming the PM counter multiplexers |
| Data readback | ~0.5-2ms per pass | Reading collected counter values from GPU to host |

**For a "basic" profile with ~20 passes:**

- Per kernel: (4.3 + ~5 + ~5 + ~0.5 + ~1) × 20 ≈ **~316ms** per kernel launch
- Normal time: 4.3ms
- **Slowdown: ~73×**

**For a "full" profile with ~200 passes:**

- Per kernel: (4.3 + ~5 + ~5 + ~0.5 + ~1) × 200 ≈ **~3,160ms** (~3.2 seconds) per kernel launch
- Normal time: 4.3ms
- **Slowdown: ~735×**

Now multiply by the number of matching kernel launches:

- `mem_efficient_bwd_unified_kernel_exp12` has **3,840 instances** in your 10-step training run
- If you profile ALL of them with "full": 3,160ms × 3,840 = **~12,134 seconds ≈ 3.4 hours**
- If you profile just 1 (`-c 1`): **~3.2 seconds** for the profiled kernel + normal runtime for everything else

> [!CAUTION]
> **Always use `-c 1` (or `-c 2` or small number) when you're exploring.** Profiling one instance of a kernel gives you all the information you need. The counters don't change between invocations unless the kernel's grid/block config changes.

---

## 7. Practical Commands for Your Setup

### Your environment

- Binary: `/home/blu-bridge016/Downloads/Neural_Networks_exp_1926/master_gau_latest_ada_6000_sm89/snippet_runner`
- GPU: Ada Lovelace RTX 6000 (SM89), device 6
- ncu version: 2025.3.1.0
- Target kernel: `mem_efficient_bwd_unified_kernel_exp12`

### Command 1: Quick first look (basic set, 1 instance)

```bash
CUDA_VISIBLE_DEVICES=6 ncu \
  -k "regex:mem_efficient_bwd_unified_kernel_exp12" \
  -c 1 \
  --set basic \
  -o attn_bwd_basic \
  ./snippet_runner
```

### Command 2: Detailed analysis (detailed set, 1 instance)

```bash
CUDA_VISIBLE_DEVICES=6 ncu \
  -k "regex:mem_efficient_bwd_unified_kernel_exp12" \
  -c 1 \
  --set detailed \
  -o attn_bwd_detailed \
  ./snippet_runner
```

### Command 3: Full analysis with roofline (full set, 1 instance)

```bash
CUDA_VISIBLE_DEVICES=6 ncu \
  -k "regex:mem_efficient_bwd_unified_kernel_exp12" \
  -c 1 \
  --set full \
  -o attn_bwd_full \
  ./snippet_runner
```

---

## 8. What Happens Inside ncu — The Full Execution Flow

1. **ncu starts snippet_runner** as a child process and injects its library.
2. **snippet_runner begins execution**. Non-matching kernels run at full speed.
3. **Target kernel is reached**:
   - ncu snapshots memory.
   - ncu runs the kernel multiple times (passes) to read counters.
   - ncu restores memory between each pass.
   - ncu performs one final "real" run.
4. **ncu finishes** and saves the `.ncu-rep` file.

---

## 9. What You'll See in the Report

### SpeedOfLight (SOL)

- **SM Throughput %**: Compute utilization.
- **Memory Throughput %**: Bandwidth utilization.

---

## 10. Output File Formats

| Extension | What It Is | How to Open |
|-----------|-----------|-------------|
| `.ncu-rep` | Binary Nsight Compute report | `ncu -i report.ncu-rep` or Nsight Compute GUI |

---

## 11. Comparison Summary: nsys vs ncu

| Aspect | Nsight Systems (nsys) | Nsight Compute (ncu) |
|--------|----------------------|---------------------|
| **Scope** | Entire application | Individual kernel(s) |
| **Overhead** | Very low | Very high |
| **Use when** | Find WHICH kernel is slow | Find WHY it is slow |

---

## 12. Recommended Workflow

### Step 0: The Prerequisite Check (Shape Analysis)

Before running `ncu`, you must ask: **Are my kernel calls identical?**

1. **Check nsys first**: Look at the `Duration` of your kernel in Nsight Systems.
    - All durations nearly equal? → Shapes are constant. **Use `-c 1`**.
    - Durations vary wildly? → Shapes are dynamic. **Use `-c 5` or more**.
2. **Check code**: Does your DataLoader use padding or constant sequence lengths?

**Why this matters:**

- A small shape will look "Latency Bound" (Low GPU utilization).
- A large shape will look "Throughput Bound" (High GPU utilization).

**Strategy for Dynamic Shapes:**
If your shapes change, run:
`ncu -c 5 --set basic -o multi_shape_report ./app`
Then in the GUI, use the **"Launch" dropdown** at the top to compare.

### Step 1: Find the bottleneck

`nsys profile ./snippet_runner`

### Step 2: Get high-level metrics

`ncu -k "regex:kernel_name" -c 1 --set basic ./snippet_runner`

> [!TIP]
> **Start with `--set basic -c 1`.** This is the fastest way to see if you are compute-bound or memory-bound.

---

## 13. Deep Dive: Hardware Counters vs. Software Metrics

To understand why Nsight Compute is so powerful, you have to understand the difference between what the **Hardware** (the GPU chip) does and what the **Software** (Nsight) shows you.

### 13.1 The "Car Dashboard" Analogy

Think of your GPU like a high-performance race car:

- **Performance Counters (Hardware Sensors)**: These are physical circuits built into the silicon. There is a sensor for "Engine RPM," a sensor for "Fuel Flow," and a sensor for "Wheel Rotations." These are raw numbers.
- **Metrics (The Dashboard)**: Your dashboard doesn't just show raw numbers; it shows things like **"Miles Per Gallon (MPG)"**.
  - There is no "MPG sensor" in an engine.
  - Instead, the car's computer takes the **Fuel Counter** and the **Distance Counter** and calculates: `Distance / Fuel = MPG`.

**In Nsight Compute:** Most of those **~7,794 Metrics** are just formulas. Nsight reads a few raw Hardware Counters and then does the math to show you the "Dashboard" view.

### 13.2 The "Slot" System (Multiplexing)

Each SM (Streaming Multiprocessor) in your **Ada SM89** GPU has a very small number of physical "wires" or **Counter Slots** (usually 4 to 8 per sub-partition).

1. **They are Programmable**: These slots are not fixed to one job. They are like universal gauges.
2. **Pass 1 (Group A)**: Nsight "re-programs" the 8 slots to listen to **Memory Traffic**.
3. **Pass 2 (Group B)**: Nsight "re-programs" those same 8 slots to listen to **Tensor Core Math**.
4. **Mirroring**: Every SM on the chip is programmed exactly the same way at the same time. Nsight then collects the data from all 144+ SMs and combines them for your report.

### 13.3 Summary: How the Data Flows

1. **The Trigger**: You run `ncu`.
2. **The Replay**: `ncu` pauses your app at the kernel you want to profile.
3. **The Setup**: For the first "Pass," it programs the 8 hardware slots to Group A.
4. **The Run**: The kernel runs. The hardware counters tick up.
5. **The Save**: `ncu` reads the numbers, saves them, and **resets the GPU memory** to the beginning state.
6. **The Next Pass**: It programs the same 8 slots to Group B and runs again.
7. **The Calculation**: Once all passes are done, Nsight takes all the saved raw counts and runs them through the ~7,794 formulas to generate your report.

---

## 14. The Sampling Strategy: Why we use `-c 1`

One of the most confusing things is seeing thousands of kernel instances in Nsight Systems (like your 3,840 launches) but only profiling one in Nsight Compute.

### 14.1 The "Identical Twin" Rule (Determinism)

In a typical neural network training loop, every time a specific kernel is launched, it uses the **exact same code**, the **exact same grid/block size**, and the **exact same GPU hardware**.

- **The Cookie Analogy**: If you bake a batch of 3,840 cookies from the same recipe, you only need to taste **one cookie** to know how they all taste. You don't need to eat the whole batch!

### 14.2 The Time Penalty (Sampling vs. Exhaustive)

Nsight Compute is slow because of the multi-pass replay. The `-c` (count) flag controls how many "cookies" you taste.

| Feature | `-c 1` (Recommended) | No `-c` flag (Exhaustive) |
| :--- | :--- | :--- |
| **Action** | Profile 1st instance, run others at full speed. | Profile **EVERY** instance. |
| **Wait Time** | ~2 minutes total. | **~5.3 HOURS** (for 3,840 instances). |
| **Disk Space** | ~5 MB (Easy to open). | **~20 GB** (Might crash your UI). |
| **Value** | 100% of the info you need. | 3,840 copies of the same info. |

### 14.3 What happens to the "Un-profiled" instances?

They are **NOT** skipped. They still run and do the math for your training. Nsight Compute just lets them run at their normal, fast speed without stopping them to read counters.

---

## 15. FAQ: Regex, GUI, and Hardware Requirements

### 15.1 What is the `regex:` prefix in the command?

CUDA kernel names are often very long and contain template parameters like `<float, 64, true>`.

- **Without regex**: You must provide the **exact, full name**. If you miss one character, `ncu` will fail to find the kernel.
- **With `regex:`**: This tells Nsight Compute to use **Regular Expression** matching. You can just provide a part of the name (e.g., `regex:mem_efficient_bwd`). It's like a "search" instead of an "exact match."
- **Verdict**: **Always use `regex:`** to avoid "No kernels profiled" errors.

### 15.2 Can I do this in the Nsight Compute GUI?

**Yes.** You can use the **Interactive Profile** activity in the GUI.

- **Pros**: Visual buttons, easier to see all options.
- **Cons**: Can be very laggy over SSH or remote connections.
- **Best Practice**: Run the profiling on the server via CLI to get the `.ncu-rep` file, then copy that file to your local laptop to view it in the GUI.

### 15.3 Do I need a GPU?

**YES.** You cannot profile CUDA kernels without an NVIDIA GPU.

- **Collection Phase**: You MUST have a GPU. The hardware counters are physical circuits on the chip.
- **Viewing Phase**: You do NOT need a GPU. You can open a saved `.ncu-rep` file on any computer (even a Mac or a laptop without a GPU) to analyze the data.

### 15.4 Why does it say "0 kernels profiled"?

This is the most common error. It usually means:

1. Your `-k` filter name was slightly wrong (use `regex:`!).
2. The kernel didn't actually run (check your training loop).
3. You are using `-c 1` but the kernel you are looking for only runs in the 2nd step, and you stopped after the 1st step.

> [!IMPORTANT]
> If you get "0 kernels profiled," run `ncu --list-kernels ./app` to see the exact names of every kernel that runs in your program.
