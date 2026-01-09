#pragma once

#include <torch/torch.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>

namespace LibTorchCSVUtils {

/**
 * @brief Structure to hold loaded tensor data
 */
struct LoadedTensorData {
    std::vector<float> data;
    std::vector<int64_t> shape;
    int64_t num_elements;
};

/**
 * @brief Parse shape from CSV metadata comment
 * Example: "# Shape: [2,3,3]" -> {2, 3, 3}
 */
inline std::vector<int64_t> parse_shape_from_csv(const std::string& line) {
    std::vector<int64_t> shape;
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
    return shape;
}

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
            shape = parse_shape_from_csv(line);
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
        for (int i = 0; i < num_tensors && (start_col + i) < static_cast<int>(tokens.size()); ++i) {
            result[i].data.push_back(std::stof(tokens[start_col + i]));
        }
        
        rows_read++;
    }
    
    file.close();
    
    // Validate
    for (int i = 0; i < num_tensors; ++i) {
        if (static_cast<int64_t>(result[i].data.size()) != total_elements) {
            throw std::runtime_error("Data size mismatch for tensor " + std::to_string(i));
        }
    }
    
    return result;
}

/**
 * @brief Create LibTorch tensors from loaded CSV data
 * 
 * @param filename Input CSV filename
 * @param num_tensors Number of tensors to load (default: 2)
 * @param options Tensor options (dtype, device)
 * @return Vector of torch::Tensor
 */
inline std::vector<torch::Tensor> load_tensors(
    const std::string& filename,
    int num_tensors = 2,
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
) {
    auto loaded = load_inputs(filename, num_tensors);
    
    std::vector<torch::Tensor> tensors;
    tensors.reserve(num_tensors);
    
    for (int i = 0; i < num_tensors; ++i) {
        // Create tensor from data
        torch::Tensor t = torch::from_blob(
            loaded[i].data.data(),
            loaded[i].shape,
            options
        ).clone();  // Clone to own the data
        
        tensors.push_back(t);
    }
    
    return tensors;
}

/**
 * @brief Load a pair of tensors (common case: input_a, input_b)
 * 
 * @param filename Input CSV filename
 * @param options Tensor options
 * @return Pair of torch::Tensor
 */
inline std::pair<torch::Tensor, torch::Tensor> load_tensor_pair(
    const std::string& filename,
    torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU)
) {
    auto tensors = load_tensors(filename, 2, options);
    return {tensors[0], tensors[1]};
}

/**
 * @brief Helper to get float value from tensor at index
 */
inline float get_float_value(const torch::Tensor& t, int64_t idx) {
    auto cpu = t.to(torch::kCPU).contiguous();
    switch (cpu.scalar_type()) {
        case torch::kFloat32:
            return cpu.data_ptr<float>()[idx];
        case torch::kFloat64:
            return static_cast<float>(cpu.data_ptr<double>()[idx]);
        case torch::kInt32:
            return static_cast<float>(cpu.data_ptr<int32_t>()[idx]);
        case torch::kInt64:
            return static_cast<float>(cpu.data_ptr<int64_t>()[idx]);
        default:
            return 0.0f;
    }
}

/**
 * @brief Export LibTorch tensors to CSV (for output comparison)
 */
inline void export_outputs(
    const std::string& filename,
    const std::vector<torch::Tensor>& inputs,
    const std::vector<std::string>& input_names,
    const std::vector<torch::Tensor>& outputs,
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

} // namespace LibTorchCSVUtils
