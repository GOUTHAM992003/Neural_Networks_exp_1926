#pragma once

/**
 * @file CSV I/O Utilities for TensorLib Benchmarks
 * 
 * This header provides utilities for:
 * - Exporting tensors to CSV files (for sharing inputs with other libraries)
 * - Importing tensors from CSV files (for loading shared inputs)
 * - Writing benchmark outputs to CSV
 * 
 * Usage Example:
 * 
 * // 1. Export inputs for another library to use
 * #include "benchmark/CSVWriter.h"
 * 
 * Tensor a = Tensor::rand({{3, 3}}, opts);
 * Tensor b = Tensor::rand({{3, 3}}, opts);
 * 
 * OwnTensor::CSVUtils::export_inputs(
 *     "inputs.csv",
 *     {a, b},
 *     {"input_a", "input_b"},
 *     {3, 3}
 * );
 * 
 * // 2. Load inputs from CSV
 * #include "benchmark/CSVReader.h"
 * 
 * auto [a, b] = OwnTensor::CSVUtils::load_tensor_pair("inputs.csv", opts);
 * 
 * // 3. Export outputs
 * Tensor result = a + b;
 * 
 * OwnTensor::CSVUtils::export_outputs(
 *     "outputs.csv",
 *     {a, b},
 *     {"input_a", "input_b"},
 *     {result},
 *     {"add"},
 *     {3, 3}
 * );
 */

#include "CSVWriter.h"
#include "CSVReader.h"