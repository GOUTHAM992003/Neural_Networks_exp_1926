#!/usr/bin/bash
 
 # Add CUDA path if it exists to ensure ncu is found even under sudo
 if [ -d "/usr/local/cuda/bin" ]; then
     export PATH="/usr/local/cuda/bin:$PATH"
 fi
 if [ -d "/usr/local/cuda/lib64" ]; then
     export LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
 fi

################################################################################
# Comprehensive Benchmarking Script for Tensor-Implementations
# Supports CPU (Valgrind, perf) and GPU (Nsight Compute) profiling
################################################################################

set -e  # Exit on error

# ============================= Configuration ==================================
TARGET_FILE="$1"
OUTPUT_LOG="BenchmarkLog.txt"
EXECUTABLE="snippet_runner"

# Compiler settings (must match Makefile)
CXX="g++"
CPPFLAGS="-Iinclude -I/usr/local/cuda/include -DWITH_CUDA"
CXXFLAGS="-std=c++20 -fPIC -Wall -Wextra -g -fopenmp"
RPATH="-Wl,-rpath,\$ORIGIN/lib"
LDFLAGS="-L/usr/local/cuda/lib64 -Llib $RPATH"
LDLIBS="-lcudart -ltbb -lcurand -ltensor"

# ============================= Validation =====================================
if [ -z "$TARGET_FILE" ]; then
    echo ""
    echo "ERROR: No file specified"
    echo "Usage: make run-script FILE=path/to/your/file.cpp"
    echo ""
    exit 1
fi

if [ ! -f "$TARGET_FILE" ]; then
    echo "ERROR: File '$TARGET_FILE' not found"
    exit 1
fi

# ============================= Initialize Log =================================
echo "" | tee -a "$OUTPUT_LOG"
echo "BENCHMARK RUN: $(date)" | tee -a "$OUTPUT_LOG"
echo "Target File: $TARGET_FILE" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"

# ============================= Compilation ====================================
echo "" | tee -a "$OUTPUT_LOG"
echo " Compiling $TARGET_FILE..." | tee -a "$OUTPUT_LOG"
$CXX $CPPFLAGS $CXXFLAGS -o $EXECUTABLE "$TARGET_FILE" $LDFLAGS $LDLIBS 2>&1 | tee -a "$OUTPUT_LOG"

if [ $? -eq 0 ]; then
    echo " Compilation successful" | tee -a "$OUTPUT_LOG"
else
    echo " Compilation failed" | tee -a "$OUTPUT_LOG"
    exit 1
fi

# ============================= CPU Benchmarking ===============================
echo "" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"
echo " CPU BENCHMARKING" | tee -a "$OUTPUT_LOG"
echo "━" | tee -a "$OUTPUT_LOG"

# --- Memory Profiling (Peak RSS) ---
echo "" | tee -a "$OUTPUT_LOG"
echo " Running Memory Profiling (Peak RSS)..." | tee -a "$OUTPUT_LOG"
/usr/bin/time -v ./$EXECUTABLE > .temp_stdout 2> .temp_stderr
cat .temp_stdout >> "$OUTPUT_LOG"
cat .temp_stderr >> "$OUTPUT_LOG"

# Extract peak memory usage (Maximum resident set size in Kbytes)
peak_kb=$(grep "Maximum resident set size" .temp_stderr | awk '{print $NF}')
peak_mb=$(awk "BEGIN {printf \"%.3f\", $peak_kb / 1024}")

# Calculate memory utilization % (of total system RAM)
total_mem_kb=$(grep MemTotal /proc/meminfo | awk '{print $2}')
utilization=$(awk "BEGIN {printf \"%.4f\", ($peak_kb / $total_mem_kb) * 100}")

echo "METRIC:PEAK_MEMORY_MB:$peak_mb" >> "$OUTPUT_LOG"
echo "METRIC:PEAK_MEMORY_UTILIZATION_PERCENT:$utilization" >> "$OUTPUT_LOG"
echo "    Peak Memory Usage (RSS): $peak_mb MB" | tee -a "$OUTPUT_LOG"
echo "    Peak Memory Utilization: $utilization%" | tee -a "$OUTPUT_LOG"

# Cleanup temp files
rm -f .temp_stdout .temp_stderr

# --- Cache Profiling with Valgrind Cachegrind ---
echo "" | tee -a "$OUTPUT_LOG"
echo "  Running Valgrind Cachegrind (Cache Analysis)..." | tee -a "$OUTPUT_LOG"
rm -f cachegrind.out.*
valgrind --tool=cachegrind --cachegrind-out-file=cachegrind.out.%p ./$EXECUTABLE >> "$OUTPUT_LOG" 2>&1

# Extract cache statistics
if ls cachegrind.out.* >/dev/null 2>&1; then
    cg_annotate cachegrind.out.* >> "$OUTPUT_LOG" 2>&1
    
    # Parse cache misses and references
    d1_miss_rate=$(grep "D1  miss rate:" "$OUTPUT_LOG" | tail -n 1 | sed 's/.*miss rate:[[:space:]]*\([0-9.]*%\).*/\1/')
    ll_miss_rate=$(grep "LL miss rate:" "$OUTPUT_LOG" | tail -n 1 | sed 's/.*miss rate:[[:space:]]*\([0-9.]*%\).*/\1/')
    i1_miss_rate=$(grep "I1  miss rate:" "$OUTPUT_LOG" | tail -n 1 | sed 's/.*miss rate:[[:space:]]*\([0-9.]*%\).*/\1/')
    
    # Get total references (remove commas for cleaner output)
    total_data_refs=$(grep "D   refs:" "$OUTPUT_LOG" | tail -n 1 | awk '{print $4}' | tr -d ',')
    total_inst_refs=$(grep "I   refs:" "$OUTPUT_LOG" | tail -n 1 | awk '{print $4}' | tr -d ',')
    
    # Get absolute miss counts
    d1_misses=$(grep "D1  misses:" "$OUTPUT_LOG" | tail -n 1 | awk '{print $4}' | tr -d ',')
    ll_misses=$(grep "LL misses:" "$OUTPUT_LOG" | tail -n 1 | awk '{print $4}' | tr -d ',')
    i1_misses=$(grep "I1  misses:" "$OUTPUT_LOG" | tail -n 1 | awk '{print $4}' | tr -d ',')
    
    echo "METRIC:L1_DATA_CACHE_MISS_RATE:$d1_miss_rate" >> "$OUTPUT_LOG"
    echo "METRIC:L1_INST_CACHE_MISS_RATE:$i1_miss_rate" >> "$OUTPUT_LOG"
    echo "METRIC:LL_CACHE_MISS_RATE:$ll_miss_rate" >> "$OUTPUT_LOG"
    echo "METRIC:TOTAL_DATA_REFS:$total_data_refs" >> "$OUTPUT_LOG"
    echo "METRIC:TOTAL_INST_REFS:$total_inst_refs" >> "$OUTPUT_LOG"
    echo "METRIC:L1_DATA_MISSES:$d1_misses" >> "$OUTPUT_LOG"
    echo "METRIC:L1_INST_MISSES:$i1_misses" >> "$OUTPUT_LOG"
    echo "METRIC:LL_MISSES:$ll_misses" >> "$OUTPUT_LOG"
    
    echo "    L1 Data Cache Miss Rate: $d1_miss_rate (${d1_misses} misses / ${total_data_refs} refs)" | tee -a "$OUTPUT_LOG"
    echo "    L1 Instruction Cache Miss Rate: $i1_miss_rate (${i1_misses} misses / ${total_inst_refs} refs)" | tee -a "$OUTPUT_LOG"
    echo "    Last Level Cache Miss Rate: $ll_miss_rate (${ll_misses} total misses)" | tee -a "$OUTPUT_LOG"
fi

# --- CPU Performance Counters with perf (if available) ---
echo "" | tee -a "$OUTPUT_LOG"
echo " Running CPU Performance Counters..." | tee -a "$OUTPUT_LOG"

if command -v perf &> /dev/null; then
    # Try to run perf, but don't fail if we don't have permissions
    if sudo -n true 2>/dev/null; then
        sudo perf stat -x "," -e cpu_core/cycles/,cpu_core/instructions/ ./$EXECUTABLE >> "$OUTPUT_LOG" 2>&1 || true
        
        # Extract perf metrics
        cycles=$(grep "cpu_core/cycles/" "$OUTPUT_LOG" | tail -n 1 | cut -d',' -f1)
        instructions=$(grep "cpu_core/instructions/" "$OUTPUT_LOG" | tail -n 1 | cut -d',' -f1)
        
        if [ -n "$cycles" ] && [ -n "$instructions" ] && [ "$cycles" != "0" ] && [ "$instructions" != "0" ]; then
            cpi=$(awk "BEGIN {printf \"%.2f\", $cycles / $instructions}")
            echo "METRIC:CPU_CYCLES:$cycles" >> "$OUTPUT_LOG"
            echo "METRIC:CPU_INSTRUCTIONS:$instructions" >> "$OUTPUT_LOG"
            echo "METRIC:CPU_CPI:$cpi" >> "$OUTPUT_LOG"
            echo "    CPU Cycles: $cycles" | tee -a "$OUTPUT_LOG"
            echo "    CPU Instructions: $instructions" | tee -a "$OUTPUT_LOG"
            echo "    CPU CPI: $cpi" | tee -a "$OUTPUT_LOG"
        else
            echo "    perf metrics not available (may need sudo permissions)" | tee -a "$OUTPUT_LOG"
        fi
    else
        echo "    perf requires sudo permissions - skipping" | tee -a "$OUTPUT_LOG"
    fi
else
    echo "    perf not installed - skipping" | tee -a "$OUTPUT_LOG"
fi

# ============================= GPU Benchmarking ===============================
echo "" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"
echo " GPU BENCHMARKING" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"

if command -v ncu &> /dev/null; then
    # --- GPU Performance Metrics ---
    echo "" | tee -a "$OUTPUT_LOG"
    echo " Running Nsight Compute (GPU Profiling)..." | tee -a "$OUTPUT_LOG"
    
    # Run ncu for cycles and instructions
    ncu --csv --metrics smsp__cycles_active.avg,smsp__inst_executed.sum ./$EXECUTABLE >> "$OUTPUT_LOG" 2>&1 || true
    
    # Parse GPU metrics (handle commas in quoted numbers)
    gpu_cycles=$(grep "smsp__cycles_active" "$OUTPUT_LOG" | tail -n 1 | sed 's/.*,"\(.*\)"/\1/' | tr -d ',')
    gpu_inst=$(grep "smsp__inst_executed" "$OUTPUT_LOG" | tail -n 1 | sed 's/.*,"\(.*\)"/\1/' | tr -d ',')
    
    if [ -n "$gpu_cycles" ] && [ -n "$gpu_inst" ] && [ "$gpu_cycles" != "0" ] && [ "$gpu_inst" != "0" ]; then
        gpu_cpi=$(awk "BEGIN {printf \"%.2f\", $gpu_cycles / $gpu_inst}")
        echo "METRIC:GPU_CYCLES:$gpu_cycles" >> "$OUTPUT_LOG"
        echo "METRIC:GPU_INSTRUCTIONS:$gpu_inst" >> "$OUTPUT_LOG"
        echo "METRIC:GPU_CPI:$gpu_cpi" >> "$OUTPUT_LOG"
        echo "    GPU Cycles: $gpu_cycles" | tee -a "$OUTPUT_LOG"
        echo "    GPU Instructions: $gpu_inst" | tee -a "$OUTPUT_LOG"
        echo "    GPU CPI: $gpu_cpi" | tee -a "$OUTPUT_LOG"
    fi
    
    # --- GPU Cache Performance ---
    echo "" | tee -a "$OUTPUT_LOG"
    echo "  Running GPU Cache Analysis..." | tee -a "$OUTPUT_LOG"
    ncu --csv --metrics l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct ./$EXECUTABLE >> "$OUTPUT_LOG" 2>&1 || true
    
    # Parse cache metrics
    l1_hit=$(grep "l1tex__t_sector_hit_rate" "$OUTPUT_LOG" | tail -n 1 | sed 's/.*,"\(.*\)"/\1/' | tr -d ',')
    l2_hit=$(grep "lts__t_sector_hit_rate" "$OUTPUT_LOG" | tail -n 1 | sed 's/.*,"\(.*\)"/\1/' | tr -d ',')
    
    if [ -n "$l1_hit" ] && [ -n "$l2_hit" ]; then
        l1_miss=$(awk "BEGIN {printf \"%.2f\", 100 - $l1_hit}")
        l2_miss=$(awk "BEGIN {printf \"%.2f\", 100 - $l2_hit}")
        
        echo "METRIC:GPU_L1_HIT_RATE:$l1_hit" >> "$OUTPUT_LOG"
        echo "METRIC:GPU_L1_MISS_RATE:$l1_miss" >> "$OUTPUT_LOG"
        echo "METRIC:GPU_L2_HIT_RATE:$l2_hit" >> "$OUTPUT_LOG"
        echo "METRIC:GPU_L2_MISS_RATE:$l2_miss" >> "$OUTPUT_LOG"
        
        echo "    GPU L1 Cache Hit Rate: ${l1_hit}%" | tee -a "$OUTPUT_LOG"
        echo "    GPU L1 Cache Miss Rate: ${l1_miss}%" | tee -a "$OUTPUT_LOG"
        echo "    GPU L2 Cache Hit Rate: ${l2_hit}%" | tee -a "$OUTPUT_LOG"
        echo "    GPU L2 Cache Miss Rate: ${l2_miss}%" | tee -a "$OUTPUT_LOG"
    fi
    
    # --- GPU Memory Bandwidth ---
    echo "" | tee -a "$OUTPUT_LOG"
    echo " Running GPU Memory Bandwidth Analysis..." | tee -a "$OUTPUT_LOG"
    ncu --csv --metrics dram__bytes.sum ./$EXECUTABLE >> "$OUTPUT_LOG" 2>&1 || true
    
    dram_bytes=$(grep "dram__bytes" "$OUTPUT_LOG" | tail -n 1 | sed 's/.*,"\(.*\)"/\1/' | tr -d ',')
    if [ -n "$dram_bytes" ] && [ "$dram_bytes" != "0" ]; then
        dram_mb=$(awk "BEGIN {printf \"%.2f\", $dram_bytes / (1024*1024)}")
        echo "METRIC:GPU_DRAM_MB:$dram_mb" >> "$OUTPUT_LOG"
        echo "    GPU DRAM Bytes Transferred: ${dram_mb} MB" | tee -a "$OUTPUT_LOG"
    fi
else
    echo "    Nsight Compute (ncu) not found - skipping GPU benchmarks" | tee -a "$OUTPUT_LOG"
fi

# ============================= Summary ========================================
echo "" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"
echo " BENCHMARK SUMMARY" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"

# Extract and display all metrics
echo "" | tee -a "$OUTPUT_LOG"
echo "All metrics have been logged to: $OUTPUT_LOG"
echo "You can extract specific metrics using:"
echo "  grep 'METRIC:' $OUTPUT_LOG"
echo ""

# ============================= Cleanup ========================================
echo " Cleaning up temporary files..." | tee -a "$OUTPUT_LOG"
rm -f $EXECUTABLE massif.out.* cachegrind.out.*
echo " Benchmark complete!" | tee -a "$OUTPUT_LOG"
echo "" | tee -a "$OUTPUT_LOG"

exit 0