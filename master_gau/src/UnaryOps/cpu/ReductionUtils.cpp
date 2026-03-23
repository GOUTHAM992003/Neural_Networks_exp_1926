#include <algorithm> //For std::find
#include <numeric>   // For std::accumulate, std::multiplies
#include <set>       // For unique axes
#include <stdexcept> // For runtime_error
#include <cstdint>   // For int64_t and size_t

#include "ops/helpers/ReductionUtils.h" // Provides declarations for all functions
#include "core/Tensor.h"          // Provides OwnTensor::Shape (required for calculate_output_shape return type)

namespace OwnTensor {
namespace detail {

/**
 * @brief Normalizes the input axes to positive indices (0 to N-1) and validates them.
 * @param input_dims The shape/dimensions of the input tensor.
 * @param axes The dimensions to reduce over (can include negative indices).
 * @return A vector of positive, unique axis indices, sorted ascendingly.
 */
std::vector<int64_t> normalize_axes(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& axes) {
    const int64_t ndim = input_dims.size();
    std::set<int64_t> unique_axes_set;

    // Handle empty axes: means reduce over all dimensions
    if (axes.empty()) {
        for (int64_t i = 0; i < ndim; ++i) {
            unique_axes_set.insert(i);
        }
    } else {
        for (int64_t axis : axes) {
            int64_t normalized_axis = axis;
            
            // 1. Handle negative axes (e.g., -1 becomes ndim - 1)
            if (normalized_axis < 0) {
                normalized_axis += ndim;
            }

            // 2. Validate bounds
            if (normalized_axis < 0 || normalized_axis >= ndim) {
                throw std::runtime_error("Reduction axis " + std::to_string(axis) + 
                                         " is out of bounds for tensor of rank " + std::to_string(ndim) + ".");
            }

            unique_axes_set.insert(normalized_axis);
        }
    }

    // Convert the set back to a sorted vector
    std::vector<int64_t> normalized_axes(unique_axes_set.begin(), unique_axes_set.end());
    return normalized_axes;
}

/**
 * @brief Calculates the shape of the output tensor after reduction.
 * @param input_dims The shape of the input tensor.
 * @param normalized_axes The axes being reduced (positive indices).
 * @param keepdim If true, keeps reduced dimensions as 1.
 * @return The Shape struct of the output tensor.
 */
Shape calculate_output_shape(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& normalized_axes, bool keepdim) {
    std::vector<int64_t> output_dims;
    
    // Lambda to check if a dimension index is marked for reduction
    auto is_reduced = [&](int64_t dim_idx) {
        return std::find(normalized_axes.begin(), normalized_axes.end(), dim_idx) != normalized_axes.end();
    };

    // Use size_t for the loop counter 'i' to avoid signed/unsigned comparison warnings.
    for (size_t i = 0; i < input_dims.size(); ++i) {
        if (is_reduced(static_cast<int64_t>(i))) {
            if (keepdim) {
                // Reduced axis is kept as size 1
                output_dims.push_back(1);
            }
            // If keepdim is false, this axis is dropped
        } else {
            // Unreduced axis, keep original size
            output_dims.push_back(input_dims[i]);
        }
    }
    
    // Handle scalar output 
    if (output_dims.empty()) {
        output_dims.push_back(1);
    }

    return Shape{output_dims};
}

/**
 * @brief Calculates the total number of elements that will be combined for each reduction slice.
 * @param input_dims The shape of the input tensor dimensions.
 * @param normalized_axes The axes being reduced.
 * @return The total number of elements being reduced.
 */
int64_t calculate_reduced_count(const std::vector<int64_t>& input_dims, const std::vector<int64_t>& normalized_axes) {
    // If normalized_axes is empty, it means reduce over all dimensions
    if (normalized_axes.empty()) {
        return std::accumulate(input_dims.begin(), input_dims.end(), 1LL, std::multiplies<int64_t>());
    }
    
    int64_t count = 1;
    for (int64_t axis : normalized_axes) {
        count *= input_dims[axis];
    }
    return count;
}


//returns true if normalized_axes == {ndim-k,ndim-k+1, ...., ndim-1}
//i.e., axes are the rightmost k consecutive dimensions 
static bool axes_are_innermost(const std::vector<int64_t>& normalized_axes,int64_t ndim){
    int64_t k = static_cast<int64_t>(normalized_axes.size());
    int64_t start = ndim - k;
    for (int64_t i = 0;i<k;++i)
        if (normalized_axes[i] != start + i) return false;  
    return true ;
}

//returns true if normalized_axes == {0,1,2,....,k-1}
////i.e., aces are the leftmost k consecutive dimensions 
static bool axes_are_outermost(const std::vector<int64_t>& normalized_axes){
    int64_t k = static_cast<int64_t>(normalized_axes.size());
    for(int64_t i = 0; i<k;++i)
        if(normalized_axes[i] != i) return false;
    return true ;
}
ReductionLayout compute_reduction_layout(const Tensor& input , const std::vector<int64_t>& normalized_axes){
    const auto& dims = input.shape().dims;
    const int64_t ndim = static_cast<int64_t>(dims.size());
    const int64_t k = static_cast<int64_t>(normalized_axes.size());
    ReductionLayout layout ; //path defaults to Generic
    //SIMD paths only works for C-contiguous input  and all non-contiguous --->Generic path
    if(!input.is_contiguous()) return layout ;

    //Case-1 : Innercontiguous 
    //axes are the rightmost k dims : {ndim-k,....,ndim-1)
    //also covers full-tensor reduction : k == ndim,num_outputs = 1 
    if(axes_are_innermost(normalized_axes,ndim)){
        int64_t  red=1;
        for(int64_t i=ndim-k;i<ndim;++i)
            red*=dims[i];     
        int64_t outer = 1;
        for(int64_t i=0 ; i<ndim-k;++i)
            outer*=dims[i];
        layout.path = ReductionLayout::Path::InnerContiguous;
        layout.num_outputs = outer; //outer OpenMP loop = inner_count positions 
        layout.reduced_count = red ;
        layout.input_outer_stride = red; // C-contiguous : row stride == reduced_count
        //inner_count and input_row_stride not used for this path
        return layout;
    }
     // Case-2 : OuterContiguous 
        //axes are the leftmost k dims: {0,1,....,k-1}
        if (axes_are_outermost(normalized_axes)){
            int64_t red = 1;
            for (int64_t i=0;i<k;++i)
                red*= dims[i];
            int64_t inner = 1;
            for(int64_t i=k;i<ndim;++i)
                inner*=dims[i];
            layout.path = ReductionLayout::Path::OuterContiguous;
            layout.num_outputs = inner; //outer OpenMP loop = inner_count_positions
            layout.reduced_count = red ;
            layout.inner_count = inner ;
            layout.input_outer_stride = inner ; //same as row-stride for C-contiguous 
            layout.input_row_stride = inner ; //C-contiguous : stride between reduction rows
            return layout;
        }

        // Case -3 : Generic fallback 
        //Middle-dim reduction ,non-consecutive axes ,or non-contiguous input 
        return layout ;
}
} // namespace detail
} // namespace OwnTensor