#pragma once

#include "core/Tensor.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <string>

namespace OwnTensor {
namespace CSVUtils {

/**
 * @brief Get float value from tensor at index (handles different dtypes)
 */
inline float get_float_value(const Tensor& t, int64_t idx) {
    switch (t.dtype()) {
        case Dtype::Float32:
            return t.data<float>()[idx];
        case Dtype::Float64:
            return static_cast<float>(t.data<double>()[idx]);
        case Dtype::Int32:
            return static_cast<float>(t.data<int32_t>()[idx]);
        case Dtype::Int64:
            return static_cast<float>(t.data<int64_t>()[idx]);
        default:
            return 0.0f;
    }
}

/**
 * @brief Export tensor inputs to CSV file
 * 
 * @param filename Output CSV filename
 * @param tensors Vector of tensors to export
 * @param tensor_names Names for each tensor (e.g., {"input_a", "input_b"})
 * @param shape Tensor shape for metadata
 * @param precision Decimal precision (default: 12 for inputs)
 */
inline void export_inputs(
    const std::string& filename,
    const std::vector<Tensor>& tensors,
    const std::vector<std::string>& tensor_names,
    const std::vector<int64_t>& shape,
    int precision = 12
) {
    std::ofstream csv(filename);
    if (!csv.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    csv << std::fixed << std::setprecision(precision);
    
    // Write metadata as comments
    csv << "# TensorLib Benchmark Inputs\n";
    csv << "# Shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        csv << shape[i];
        if (i < shape.size() - 1) csv << ",";
    }
    csv << "]\n";
    csv << "# Tensors: " << tensors.size() << "\n";
    
    // Calculate total elements
    int64_t total_elements = 1;
    for (auto dim : shape) {
        total_elements *= dim;
    }
    
    // Write header
    csv << "index";
    
    // Add dimension columns based on shape
    if (shape.size() == 1) {
        csv << ",col";
    } else if (shape.size() == 2) {
        csv << ",row,col";
    } else if (shape.size() == 3) {
        csv << ",depth,row,col";
    } else {
        for (size_t i = 0; i < shape.size(); ++i) {
            csv << ",dim" << i;
        }
    }
    
    // Add tensor columns
    for (const auto& name : tensor_names) {
        csv << "," << name;
    }
    csv << "\n";
    
    // Write data
    for (int64_t idx = 0; idx < total_elements; ++idx) {
        csv << idx;
        
        // Calculate indices for each dimension
        int64_t remaining = idx;
        std::vector<int64_t> indices(shape.size());
        for (int d = shape.size() - 1; d >= 0; --d) {
            indices[d] = remaining % shape[d];
            remaining /= shape[d];
        }
        
        // Write dimension indices
        for (auto i : indices) {
            csv << "," << i;
        }
        
        // Write tensor values
        for (const auto& tensor : tensors) {
            csv << "," << get_float_value(tensor, idx);
        }
        csv << "\n";
    }
    
    csv.close();
}

/**
 * @brief Export tensor outputs to CSV file
 * 
 * @param filename Output CSV filename
 * @param inputs Input tensors (to include in output)
 * @param input_names Names for input tensors
 * @param outputs Output tensors from operations
 * @param output_names Names for output tensors (operation names)
 * @param shape Tensor shape
 * @param precision Decimal precision (default: 6 for outputs)
 */
inline void export_outputs(
    const std::string& filename,
    const std::vector<Tensor>& inputs,
    const std::vector<std::string>& input_names,
    const std::vector<Tensor>& outputs,
    const std::vector<std::string>& output_names,
    const std::vector<int64_t>& shape,
    int precision = 6
) {
    std::ofstream csv(filename);
    if (!csv.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    csv << std::fixed << std::setprecision(precision);
    
    int64_t total_elements = 1;
    for (auto dim : shape) {
        total_elements *= dim;
    }
    
    // Write header
    csv << "index";
    
    // Dimension columns
    if (shape.size() == 1) {
        csv << ",col";
    } else if (shape.size() == 2) {
        csv << ",row,col";
    } else if (shape.size() == 3) {
        csv << ",depth,row,col";
    } else {
        for (size_t i = 0; i < shape.size(); ++i) {
            csv << ",dim" << i;
        }
    }
    
    // Input columns
    for (const auto& name : input_names) {
        csv << "," << name;
    }
    
    // Output columns
    for (const auto& name : output_names) {
        csv << "," << name;
    }
    csv << "\n";
    
    // Write data
    for (int64_t idx = 0; idx < total_elements; ++idx) {
        csv << idx;
        
        // Calculate indices
        int64_t remaining = idx;
        std::vector<int64_t> indices(shape.size());
        for (int d = shape.size() - 1; d >= 0; --d) {
            indices[d] = remaining % shape[d];
            remaining /= shape[d];
        }
        
        for (auto i : indices) {
            csv << "," << i;
        }
        
        // Input values
        for (const auto& tensor : inputs) {
            csv << "," << get_float_value(tensor, idx);
        }
        
        // Output values
        for (const auto& tensor : outputs) {
            csv << "," << get_float_value(tensor, idx);
        }
        csv << "\n";
    }
    
    csv.close();
}

/**
 * @brief Simplified version: export single input/output pair
 */
inline void export_test_data(
    const std::string& input_file,
    const std::string& output_file,
    const Tensor& input_a,
    const Tensor& input_b,
    const std::vector<Tensor>& results,
    const std::vector<std::string>& result_names,
    const std::vector<int64_t>& shape
) {
    // Export inputs
    export_inputs(input_file, {input_a, input_b}, {"input_a", "input_b"}, shape);
    
    // Export outputs
    export_outputs(output_file, {input_a, input_b}, {"input_a", "input_b"},
                   results, result_names, shape);
}

} // namespace CSVUtils
} // namespace OwnTensor
