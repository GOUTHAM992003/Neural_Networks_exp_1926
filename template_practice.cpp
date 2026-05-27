#include <iostream>
#include <string>
#include <typeinfo>

// =================================================================
// 1. FUNCTION TEMPLATES
// =================================================================
// Instead of writing one add for 'int', one for 'float', one for 'double'...
// We define a placeholder 'T'. The compiler will generate the actual code.
template <typename T>
T add(T a, T b) {
    std::cout << "[Template add] Operating on type: " << typeid(T).name() << "\n";
    return a + b;
}

// =================================================================
// 2. TEMPLATE SPECIALIZATION
// =================================================================
// What if a general template doesn't work for a specific type?
// For example, what if we want to add two boolean values using logical OR?
// We "specialize" the template for 'bool' specifically.
template <>
bool add<bool>(bool a, bool b) {
    std::cout << "[Specialized add] Doing logical OR for booleans!\n";
    return a || b;
}

// =================================================================
// 3. STRUCT/CLASS TEMPLATES
// =================================================================
// This is exactly like your SumOp<T> or WelfordOps<T>!
// It's a structure that can hold or operate on any type 'T'.
template <typename T>
struct MathOperator {
    T identity;

    T multiply(T a, T b) {
        return a * b;
    }
};

// =================================================================
// 4. NON-TYPE TEMPLATE PARAMETERS
// =================================================================
// You can pass integers, sizes, or configurations as template parameters!
// This is evaluated at compile time. This is how warp/block sizes are configured in CUDA.
template <typename T, int VectorSize>
struct SIMDVector {
    T data[VectorSize];

    void print_info() {
        std::cout << "Vector of type " << typeid(T).name() 
                  << " with compile-time constant size: " << VectorSize << "\n";
    }
};

int main() {
    std::cout << "--- 1. Function Template Practice ---\n";
    // The compiler automatically figures out T is 'int' and generates: int add(int, int)
    int int_result = add(10, 20); 
    std::cout << "Int result: " << int_result << "\n\n";

    // The compiler automatically figures out T is 'double' and generates: double add(double, double)
    double double_result = add(5.5, 4.5); 
    std::cout << "Double result: " << double_result << "\n\n";

    // This calls the specialized version we wrote specifically for bool
    bool bool_result = add(true, false);
    std::cout << "Bool result: " << std::boolalpha << bool_result << "\n\n";

    std::cout << "--- 2. Struct/Class Template Practice ---\n";
    // We explicitly tell the struct to use 'float'
    MathOperator<float> float_op;
    float_op.identity = 1.0f;
    std::cout << "Float mult: " << float_op.multiply(2.5f, 4.0f) << "\n\n";

    std::cout << "--- 3. Non-Type Template Parameter Practice ---\n";
    // Creating a vector of size 8 (evaluated at compile time)
    SIMDVector<float, 8> my_vector;
    my_vector.print_info();

    return 0;
}
