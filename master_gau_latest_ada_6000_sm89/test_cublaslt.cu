#include <cublasLt.h>
#include <iostream>

int main() {
    cublasLtHandle_t lt;
    if (cublasLtCreate(&lt) == CUBLAS_STATUS_SUCCESS) {
        std::cout << "cuBLASLt available!" << std::endl;
        cublasLtDestroy(lt);
    }
    return 0;
}
