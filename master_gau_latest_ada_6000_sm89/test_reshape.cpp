#include <iostream>
#include <vector>

bool computeStride_view(const std::vector<int64_t>& old_shape, const std::vector<int64_t>& old_stride,
                        const std::vector<int64_t>& new_shape, std::vector<int64_t>& new_stride) {
    new_stride.resize(new_shape.size());
    if (old_shape.empty()) {
        return true;
    }
    
    int old_nd = old_shape.size();
    int new_nd = new_shape.size();
    int old_idx = 0;
    int new_idx = 0;
    
    while (old_idx < old_nd && new_idx < new_nd) {
        if (old_shape[old_idx] == 1) {
            old_idx++;
            continue;
        }
        if (new_shape[new_idx] == 1) {
            new_stride[new_idx] = 1;
            new_idx++;
            continue;
        }
        
        if (old_shape[old_idx] == new_shape[new_idx]) {
            new_stride[new_idx] = old_stride[old_idx];
            old_idx++;
            new_idx++;
            continue;
        }
        
        int64_t old_size = old_shape[old_idx];
        int64_t new_size = new_shape[new_idx];
        
        if (old_size < new_size) {
            int64_t expected_stride = old_stride[old_idx];
            int64_t current_size = old_size;
            int next_old_idx = old_idx + 1;
            
            while (next_old_idx < old_nd && current_size < new_size) {
                if (old_shape[next_old_idx] != 1) {
                    if (expected_stride != old_stride[next_old_idx] * old_shape[next_old_idx]) {
                        return false; 
                    }
                    expected_stride = old_stride[next_old_idx];
                    current_size *= old_shape[next_old_idx];
                }
                next_old_idx++;
            }
            if (current_size != new_size) return false;
            
            new_stride[new_idx] = expected_stride;
            old_idx = next_old_idx;
            new_idx++;
        } else {
            int64_t expected_stride = old_stride[old_idx];
            int64_t current_size = new_size;
            int next_new_idx = new_idx + 1;
            
            new_stride[new_idx] = expected_stride * (old_size / new_size);
            
            while (next_new_idx < new_nd && current_size < old_size) {
                if (new_shape[next_new_idx] != 1) {
                    current_size *= new_shape[next_new_idx];
                    int64_t factor = old_size / current_size;
                    new_stride[next_new_idx] = expected_stride * factor;
                } else {
                    new_stride[next_new_idx] = 1;
                }
                next_new_idx++;
            }
            if (current_size != old_size) return false;
            
            old_idx++;
            new_idx = next_new_idx;
        }
    }
    
    while (old_idx < old_nd) {
        if (old_shape[old_idx] != 1) return false;
        old_idx++;
    }
    while (new_idx < new_nd) {
        if (new_shape[new_idx] != 1) return false;
        new_stride[new_idx] = 1;
        new_idx++;
    }
    
    return true;
}

int main_old() {
    std::vector<int64_t> old_shape = {16, 1024, 768};
    std::vector<int64_t> old_stride = {1024*2304, 2304, 1};
    std::vector<int64_t> new_shape = {16, 1024, 12, 64};
    std::vector<int64_t> new_stride;
    
    bool ok = computeStride_view(old_shape, old_stride, new_shape, new_stride);
    if (ok) {
        std::cout << "OK! New strides: ";
        for (auto s : new_stride) std::cout << s << " ";
        std::cout << "\n";
    } else {
        std::cout << "FAILED to view\n";
    }
}

void test_backward() {
    std::vector<int64_t> old_shape = {16, 1024, 12, 64};
    std::vector<int64_t> old_stride = {12*1024*64, 64, 1024*64, 1};
    std::vector<int64_t> new_shape = {16, 1024, 768};
    std::vector<int64_t> new_stride;
    
    bool ok = computeStride_view(old_shape, old_stride, new_shape, new_stride);
    if (ok) {
        std::cout << "Backward OK! New strides: ";
        for (auto s : new_stride) std::cout << s << " ";
        std::cout << "\n";
    } else {
        std::cout << "Backward FAILED to view\n";
    }
}

int main() {
    test_backward();
    return 0;
}
