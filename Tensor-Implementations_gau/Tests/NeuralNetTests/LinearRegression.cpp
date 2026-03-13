/**
 * Linear Regression using Gradient Descent
 * 
 * This implements the same algorithm as the Python/NumPy code:
 * 
 * for i in range(100000):
 *   ycap = X.dot(W)
 *   loss = ((Y - ycap)**2).mean()
 *   delta = X.T.dot(Y - ycap) / Y.shape[0]
 *   W += delta * learning_rate
 * 
 * Dataset: Salary vs Years of Experience (30 samples)
 */

#include "TensorLib.h"
#include <iostream>
#include <cmath>
#include <iomanip>

using namespace OwnTensor;

// ============================================================================
// Helper: Create dataset from the salary data (from user's images)
// ============================================================================
void create_salary_dataset(Tensor& X, Tensor& Y) {
    // Years of experience data (30 samples)
    // X has shape [30, 2]: first column is bias (1.0), second is years
    float x_data[] = {
        1.0f, 1.2f,
        1.0f, 1.4f,
        1.0f, 1.6f,
        1.0f, 2.1f,
        1.0f, 2.3f,
        1.0f, 3.0f,
        1.0f, 3.1f,
        1.0f, 3.3f,
        1.0f, 3.3f,
        1.0f, 3.8f,
        1.0f, 4.0f,
        1.0f, 4.1f,
        1.0f, 4.1f,
        1.0f, 4.2f,
        1.0f, 4.6f,
        1.0f, 5.0f,
        1.0f, 5.2f,
        1.0f, 5.4f,
        1.0f, 6.0f,
        1.0f, 6.1f,
        1.0f, 6.9f,
        1.0f, 7.2f,
        1.0f, 8.0f,
        1.0f, 8.3f,
        1.0f, 8.8f,
        1.0f, 9.1f,
        1.0f, 9.6f,
        1.0f, 9.7f,
        1.0f, 10.4f,
        1.0f, 10.6f
    };
    
    // Salary data (30 samples) - shape [30, 1]
    float y_data[] = {
        39344.0f,
        46206.0f,
        37732.0f,
        43526.0f,
        39892.0f,
        56643.0f,
        60151.0f,
        54446.0f,
        64446.0f,
        57190.0f,
        63219.0f,
        55795.0f,
        56958.0f,
        57082.0f,
        61112.0f,
        67939.0f,
        66030.0f,
        83089.0f,
        81364.0f,
        93941.0f,
        91739.0f,
        98274.0f,
        101303.0f,
        113813.0f,
        109432.0f,
        105583.0f,
        116970.0f,
        112636.0f,
        122392.0f,
        121873.0f
    };
    
    TensorOptions opts;
    opts.dtype = Dtype::Float32;
    opts.device = DeviceIndex(Device::CPU);
    opts.requires_grad = false;
    
    // Create tensors
    X = Tensor({{30, 2}}, opts);
    Y = Tensor({{30, 1}}, opts);
    
    // Set data (count is number of elements, not bytes)
    X.set_data(x_data, 30 * 2);
    Y.set_data(y_data, 30 * 1);
}

// ============================================================================
// R-squared metric
// ============================================================================
float compute_r_squared(const Tensor& Y_true, const Tensor& Y_pred) {
    // RSS = sum((Y_true - Y_pred)^2)
    Tensor residuals = Y_true - Y_pred;
    Tensor rss_tensor = reduce_sum(residuals * residuals);
    
    // TSS = sum((Y_true - mean(Y_true))^2)
    Tensor y_mean = reduce_mean(Y_true);
    // Broadcast y_mean to match Y_true shape
    Tensor centered = Y_true - y_mean;
    Tensor tss_tensor = reduce_sum(centered * centered);
    
    // Get scalar values
    float rss = *(rss_tensor.data<float>());
    float tss = *(tss_tensor.data<float>());
    
    return 1.0f - (rss / tss);
}

// ============================================================================
// MAPE (Mean Absolute Percentage Error) and Accuracy
// ============================================================================
float compute_mape_and_accuracy(const Tensor& Y_true, const Tensor& Y_pred) {
    // MAPE = mean(|Y_true - Y_pred| / Y_true) * 100
    // Accuracy = 100 - MAPE
    
    float* y_ptr = const_cast<Tensor&>(Y_true).data<float>();
    float* yp_ptr = const_cast<Tensor&>(Y_pred).data<float>();
    int n = static_cast<int>(Y_true.numel());
    
    float total_error = 0.0f;
    for (int i = 0; i < n; i++) {
        float abs_error = std::abs(y_ptr[i] - yp_ptr[i]);
        float pct_error = abs_error / y_ptr[i];  // Percentage error
        total_error += pct_error;
    }
    
    float mape = (total_error / n) * 100.0f;  // Convert to percentage
    return 100.0f - mape;  // Return accuracy (100 - MAPE)
}

// ============================================================================
// Linear Regression Training using Gradient Descent
// ============================================================================
void train_linear_regression() {
    std::cout << "================================================================\n";
    std::cout << "    LINEAR REGRESSION - Gradient Descent (ML Way)\n";
    std::cout << "================================================================\n\n";
    
    // Create dataset
    Tensor X, Y;
    create_salary_dataset(X, Y);
    
    std::cout << "Dataset created:\n";
    std::cout << "  X shape: [30, 2] (bias + years of experience)\n";
    std::cout << "  Y shape: [30, 1] (salary)\n\n";
    
    // Display first few samples
    std::cout << "First 5 samples of X:\n";
    X.display(std::cout, 4);
    std::cout << "\nFirst 5 samples of Y:\n";
    Y.display(std::cout, 0);
    
    // Initialize weights randomly using library's randn function
    // Shape [2, 1] (bias weight + feature weight)
    TensorOptions opts;
    opts.dtype = Dtype::Float32;
    opts.device = DeviceIndex(Device::CPU);
    Tensor W = Tensor::randn({{2, 1}}, opts);
    
    std::cout << "\nInitial weights (random):\n";
    W.display(std::cout, 6);
    
    // Hyperparameters
    float learning_rate = 0.01f;  // Adjusted for the scale of salary data
    float convergence_threshold = 0.0001f;
    int max_iterations = 100000;
    
    std::cout << "\nTraining parameters:\n";
    std::cout << "  Learning rate: " << learning_rate << "\n";
    std::cout << "  Convergence threshold: " << convergence_threshold << "\n";
    std::cout << "  Max iterations: " << max_iterations << "\n\n";
    
    std::cout << "Starting training...\n";
    std::cout << "================================================================\n";
    
    float prev_loss = 0.0f;
    bool converged = false;
    int final_iteration = 0;
    
    // Number of samples
    float n_samples = 30.0f;
    
    for (int i = 0; i < max_iterations; i++) {
        // Forward pass: Y_pred = X @ W
        Tensor Y_pred = matmul(X, W);
        
        // Compute loss: MSE = mean((Y - Y_pred)^2)
        Tensor error = Y - Y_pred;
        Tensor squared_error = error * error;
        Tensor loss_tensor = reduce_mean(squared_error);
        float current_loss = *(loss_tensor.data<float>());
        
        // Print progress every 1000 iterations
        if (i % 1000 == 0) {
            std::cout << "Iteration " << std::setw(6) << i 
                      << " | Loss: " << std::scientific << std::setprecision(6) 
                      << current_loss << std::endl;
        }
        
        // Compute gradient: delta = X.T @ (Y - Y_pred) / n_samples
        Tensor X_T = X.t();  // Transpose X
        Tensor gradient = matmul(X_T, error);
        
        // Scale gradient by 1/n_samples (manual scalar division)
        Tensor scale_tensor = Tensor::full({{2, 1}}, opts, 1.0f / n_samples);
        gradient = gradient * scale_tensor;
        
        // Update weights: W += learning_rate * gradient
        Tensor lr_tensor = Tensor::full({{2, 1}}, opts, learning_rate);
        W = W + (gradient * lr_tensor);
        
        // Check convergence
        if (i > 0 && std::abs(prev_loss - current_loss) <= convergence_threshold) {
            std::cout << "\n Converged after " << (i + 1) << " iterations!\n";
            converged = true;
            final_iteration = i + 1;
            break;
        }
        
        prev_loss = current_loss;
        final_iteration = i + 1;
    }
    
    if (!converged) {
        std::cout << "\n  Training completed (max iterations reached)\n";
    }
    
    std::cout << "================================================================\n\n";
    
    // Final weights
    std::cout << "Trained weights (W):\n";
    W.display(std::cout, 6);
    
    float* w_ptr = W.data<float>();
    std::cout << "\nInterpretation:\n";
    std::cout << "  Intercept (bias): " << std::fixed << std::setprecision(2) << w_ptr[0] << "\n";
    std::cout << "  Slope (per year): " << std::fixed << std::setprecision(2) << w_ptr[1] << "\n";
    std::cout << "  => Salary = " << w_ptr[0] << " + " << w_ptr[1] << " × Years\n\n";
    
    // Final predictions and R-squared
    Tensor Y_pred_final = matmul(X, W);
    float r_squared = compute_r_squared(Y, Y_pred_final);
    float accuracy = r_squared * 100.0f;  // R² as percentage
    
    std::cout << "Model Performance:\n";
    std::cout << "  R-squared: " << std::fixed << std::setprecision(4) << r_squared << "\n";
    std::cout << "  Accuracy:  " << std::fixed << std::setprecision(2) << accuracy << "%\n";
    
    // Show some predictions
    std::cout << "Sample Predictions:\n";
    std::cout << "  Years | Actual Salary | Predicted Salary\n";
    std::cout << "  ------|---------------|------------------\n";
    
    float* x_ptr = X.data<float>();
    float* y_ptr = Y.data<float>();
    float* yp_ptr = Y_pred_final.data<float>();
    
    for (int i = 0; i < 30; i += 6) {  // Show every 6th sample
        float years = x_ptr[i * 2 + 1];  // Second column is years
        std::cout << "  " << std::fixed << std::setprecision(1) << std::setw(5) << years 
                  << " | " << std::setw(13) << std::setprecision(0) << y_ptr[i]
                  << " | " << std::setw(16) << std::setprecision(0) << yp_ptr[i] << "\n";
    }
    
    std::cout << "\n================================================================\n";
    std::cout << "    Training Complete!\n";
    std::cout << "================================================================\n";
}

int main() {
    try {
        train_linear_regression();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
