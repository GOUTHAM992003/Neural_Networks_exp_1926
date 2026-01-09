#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include "core/Tensor.h"
namespace OwnTensor{

    std::string dtype_to_string(Dtype dtype) {
    switch (dtype) {
        case Dtype::Float32: return "Float32";
        case Dtype::Int32:   return "Int32";
        case Dtype::Int16: return "Int16";
        case Dtype::Int64: return "Int64";
        case Dtype::Float64: return "Float64";
        case Dtype::Bfloat16: return "Bfloat16";
        case Dtype::Float16: return "Float16";
        default:             return "Unknown Dtype";
    }
}


   inline void writeTensorCSV(
    const Tensor& t,
    std::ofstream& file) {

    const Tensor* cpu_tensor = &t;
    Tensor cpu_copy({{1}}, Dtype::Float32); 

    if (t.is_cuda()) {
        cpu_copy = t.to_cpu();   
        cpu_tensor = &cpu_copy;
    }

    if (!cpu_tensor->is_contiguous()) {
        throw std::runtime_error("CSV dump requires contiguous tensor");
    }

    const auto& dims = cpu_tensor->shape().dims;
    const int ndim = dims.size();


    file << "# device = " << (t.is_cuda() ? "cuda" : "cpu") << "\n";
    file << "# dtype = " << dtype_to_string(cpu_tensor->dtype()) << "\n";

    file << "# shape = ";
    for (int i = 0; i < ndim; i++) {
        file << dims[i];
        if (i + 1 < ndim) file << ",";
    }
    file << "\n";

    auto write_value = [&](int64_t idx) {
        switch (cpu_tensor->dtype()) {
            case Dtype::Float32: file << cpu_tensor->data<float>()[idx]; break;
            case Dtype::Float64: file << cpu_tensor->data<double>()[idx]; break;
            case Dtype::Bfloat16:file<< cpu_tensor->data<bfloat16_t>()[idx];break;
            case Dtype::Float16: file<<cpu_tensor-> data<float16_t>()[idx];break;
            case Dtype::Int16:   file<<cpu_tensor->data<int16_t>()[idx];break;
            case Dtype::Int32:   file << cpu_tensor->data<int32_t>()[idx]; break;
            case Dtype::Int64:   file << cpu_tensor->data<int64_t>()[idx]; break;
            //case Dtype::Bool:    file << (cpu_tensor->data<bool>()[idx] ? 1 : 0); break;
            default:
                throw std::runtime_error("Unsupported dtype in CSV dump");
        }
    };

    // -------- 1D --------
    if (ndim == 1) {
        for (int64_t i = 0; i < dims[0]; i++) {
            write_value(i);
            if (i + 1 < dims[0]) file << "\n";
        }
        file << "\n";
    }

    // -------- 2D --------
    else if (ndim == 2) {
        int64_t R = dims[0];
        int64_t C = dims[1];

        for (int64_t r = 0; r < R; r++) {
            for (int64_t c = 0; c < C; c++) {
                write_value(r * C + c);
                if (c + 1 < C) file << "\n";
            }
            file << "\n";
        }
    }

    // -------- 3D --------
    else if (ndim == 3) {
        int64_t D = dims[0];
        int64_t R = dims[1];
        int64_t C = dims[2];
        int64_t stride = R * C;

        for (int64_t d = 0; d < D; d++) {
            file << "# slice " << d << "\n";
            int64_t base = d * stride;

            for (int64_t r = 0; r < R; r++) {
                for (int64_t c = 0; c < C; c++) {
                    write_value(base + r * C + c);
                    if (c + 1 < C) file << "\n";
                }
                file << "\n";
            }
            file << "\n";
        }
    }
    else {
        throw std::runtime_error("CSV dump supports up to 3D tensors only");
    }
}


inline void dumpOpCSV(
    const std::string& op_name,
    const std::vector<const Tensor*>& inputs,
    const Tensor& output,
    const std::string& filename){
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open CSV file");
    }

    file << "# op = " << op_name << "\n";
    file << "# input_count = " << inputs.size() << "\n\n";

    // Inputs
    for (size_t i = 0; i < inputs.size(); i++) {
        file << "# input[" << i << "]\n";
        writeTensorCSV(*inputs[i], file);
        file << "\n";
    }

    // Output
    file << "# output\n";
    writeTensorCSV(output, file);

    file.close();
}

}//namespace OwnTensor