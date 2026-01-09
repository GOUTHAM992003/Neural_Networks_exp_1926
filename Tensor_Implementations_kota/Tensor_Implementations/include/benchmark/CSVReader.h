#pragma once

#include "core/Tensor.h"
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

namespace OwnTensor {
namespace CSVUtils {

/**
 * @brief Structure to hold loaded tensor data
 */
struct LoadedTensorData {
    std::vector<float> data;
    std::vector<int64_t> shape;
    int64_t num_elements;
};

/**
 * @brief Load tensors from CSV input file
 * 
 * @param filename Input CSV filename
 * @param num_tensors Number of tensors to load (e.g., 2 for input_a, input_b)
 * @return Vector of LoadedTensorData
 */
inline std::vector<LoadedTensorData> load_inputs(
    const std::string& filename,
    int num_tensors = 2
) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    std::string line;
    std::vector<int64_t> shape;
    
    // Parse metadata from comments
    while (std::getline(file, line)) {
        if (line.find("# Shape:") != std::string::npos) {
            // Extract shape from "# Shape: [2,3,3]"
            size_t start = line.find('[');
            size_t end = line.find(']');
            if (start != std::string::npos && end != std::string::npos) {
                std::string shape_str = line.substr(start + 1, end - start - 1);
                std::stringstream ss(shape_str);
                std::string dim;
                while (std::getline(ss, dim, ',')) {
                    shape.push_back(std::stoll(dim));
                }
            }
        } else if (line.find("index") != std::string::npos) {
            // Found header line, data follows
            break;
        }
    }
    
    if (shape.empty()) {
        throw std::runtime_error("Could not parse shape from CSV file");
    }
    
    // Calculate total elements
    int64_t total_elements = 1;
    for (auto dim : shape) {
        total_elements *= dim;
    }
    
    // Initialize data containers
    std::vector<LoadedTensorData> result(num_tensors);
    for (int i = 0; i < num_tensors; ++i) {
        result[i].shape = shape;
        result[i].num_elements = total_elements;
        result[i].data.reserve(total_elements);
    }
    
    // Read data rows
    int64_t rows_read = 0;
    while (std::getline(file, line) && rows_read < total_elements) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        // Skip index and dimension columns
        int dim_cols = shape.size();
        int start_col = 1 + dim_cols; // 1 for index + dimension columns
        
        // Extract tensor values
        for (int i = 0; i < num_tensors && (start_col + i) < tokens.size(); ++i) {
            result[i].data.push_back(std::stof(tokens[start_col + i]));
        }
        
        rows_read++;
    }
    
    file.close();
    
    // Validate
    for (int i = 0; i < num_tensors; ++i) {
        if (result[i].data.size() != total_elements) {
            throw std::runtime_error("Data size mismatch for tensor " + std::to_string(i));
        }
    }
    
    return result;
}

/**
 * @brief Create TensorLib tensors from loaded CSV data
 * 
 * @param filename Input CSV filename
 * @param opts Tensor options (dtype, device)
 * @return Pair of tensors (input_a, input_b)
 */
inline std::pair<Tensor, Tensor> load_tensor_pair(
    const std::string& filename,
    const TensorOptions& opts = TensorOptions{}
) {
    auto loaded = load_inputs(filename, 2);
    
    // Create tensors
    Shape shape;
    shape.dims = loaded[0].shape;
    
    Tensor a(shape, opts);
    Tensor b(shape, opts);
    
    // Copy data
    if (opts.dtype == Dtype::Float32) {
        std::copy(loaded[0].data.begin(), loaded[0].data.end(), a.data<float>());
        std::copy(loaded[1].data.begin(), loaded[1].data.end(), b.data<float>());
    } else if (opts.dtype == Dtype::Float64) {
        for (size_t i = 0; i < loaded[0].data.size(); ++i) {
            a.data<double>()[i] = static_cast<double>(loaded[0].data[i]);
            b.data<double>()[i] = static_cast<double>(loaded[1].data[i]);
        }
    }
    
    return {a, b};
}

/**
 * @brief Load multiple tensors from CSV
 * 
 * @param filename Input CSV filename
 * @param num_tensors Number of tensors to load
 * @param opts Tensor options
 * @return Vector of tensors
 */
inline std::vector<Tensor> load_tensors(
    const std::string& filename,
    int num_tensors,
    const TensorOptions& opts = TensorOptions{}
) {
    auto loaded = load_inputs(filename, num_tensors);
    
    std::vector<Tensor> tensors;
    tensors.reserve(num_tensors);
    
    Shape shape;
    shape.dims = loaded[0].shape;
    
    for (int i = 0; i < num_tensors; ++i) {
        Tensor t(shape, opts);
        
        if (opts.dtype == Dtype::Float32) {
            std::copy(loaded[i].data.begin(), loaded[i].data.end(), t.data<float>());
        } else if (opts.dtype == Dtype::Float64) {
            for (size_t j = 0; j < loaded[i].data.size(); ++j) {
                t.data<double>()[j] = static_cast<double>(loaded[i].data[j]);
            }
        }
        
        tensors.push_back(std::move(t));
    }
    
    return tensors;
}

} // namespace CSVUtils
} // namespace OwnTensor
