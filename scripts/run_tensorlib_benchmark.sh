#!/bin/bash
# Run TensorLib benchmark and generate CSV outputs
set -e

# Configuration
TEST_NAME=${1:-"benchmark_all"}
SEED=${2:-"1926"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
TENSORLIB_DIR="$PROJECT_ROOT/Tensor_Implementations_kota/Tensor_Implementations"
OUTPUT_DIR="$PROJECT_ROOT/benchmark_results"

echo "╔════════════════════════════════════════════════╗"
echo "║       Running TensorLib Benchmark              ║"
echo "╚════════════════════════════════════════════════╝"
echo ""
echo "Test Name:  $TEST_NAME"
echo "Seed:       $SEED"
echo "Output Dir: $OUTPUT_DIR"
echo ""

# Create output directories
mkdir -p "$OUTPUT_DIR/inputs"
mkdir -p "$OUTPUT_DIR/tensorlib"

# Navigate to TensorLib directory
cd "$TENSORLIB_DIR"

# Build the library
# Build the library (Force clean)
echo "=== Building TensorLib ==="
rm -rf lib/objects lib/libtensor.a lib/libtensor.so
make -j$(nproc)

# Run the benchmark
make run-snippet FILE="Tests/benchmark/${TEST_NAME}.cpp"

# Verify generated CSVs
echo "Verifying outputs..."
REQUIRED_FILES=(
    "tensorlib_timings.csv"
    "tensorlib_values.csv"
    "tensorlib_throughput.csv"
    "tensorlib_bandwidth.csv"
    "tensorlib_flops.csv"
)

for file in "${REQUIRED_FILES[@]}"; do
    if [ -f "$OUTPUT_DIR/tensorlib/$file" ]; then
        echo "  ✓ Found $file"
    else
        echo "  ✗ ERROR: Missing $file"
        exit 1
    fi
done

cd "$PROJECT_ROOT"
echo ""
echo "════════════════════════════════════════════════"
echo "TensorLib benchmark completed successfully"
echo "════════════════════════════════════════════════"
echo "Results available in: $OUTPUT_DIR/tensorlib/"
echo ""
