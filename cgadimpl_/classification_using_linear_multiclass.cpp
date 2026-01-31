#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<limits>
#include<ad/ag_all.hpp>

int main(){
    std::vector<float> age_data;
    std::vector<float> study_hours_data;
    std::vector<float> attendance_data;
    std::vector<float> sleep_hours_data;
    std::vector<float> Y_onehot_data;  // Flattened one-hot vectors
    std::vector<int> Y_class_data;     // Class labels for accuracy calculation

    std::string line;
    std::ifstream file("/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/inputs/Exam_Score_Prediction.csv");
    std::getline(file, line);  // Skip header

    while(std::getline(file, line)){
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss, cell, ',');  // Skip student_id
        std::getline(ss, cell, ',');
        age_data.push_back(std::stof(cell));
        std::getline(ss, cell, ',');  // Skip gender
        std::getline(ss, cell, ',');  // Skip ethnicity
        std::getline(ss, cell, ',');
        study_hours_data.push_back(std::stof(cell));
        std::getline(ss, cell, ',');
        attendance_data.push_back(std::stof(cell));
        std::getline(ss, cell, ',');  // Skip parental_involvement
        std::getline(ss, cell, ',');  // Skip extra_curricular
        sleep_hours_data.push_back(std::stof(cell));
        std::getline(ss, cell, ',');  // Skip previous_grade
        std::getline(ss, cell, ',');  // Skip motivation
        std::getline(ss, cell, ',');  // Skip tutoring
        std::getline(ss, cell, ',');  // Skip access_to_learning
        std::getline(ss, cell, ',');
        float score = std::stof(cell);
        
        // Create one-hot encoding based on score
        // Class 0: Fail (score < 40)
        // Class 1: Average (40 <= score < 70)
        // Class 2: Pass (score >= 70)
        if(score < 40.0f){
            Y_onehot_data.push_back(1.0f);
            Y_onehot_data.push_back(0.0f);
            Y_onehot_data.push_back(0.0f);
            Y_class_data.push_back(0);
        } else if(score < 70.0f){
            Y_onehot_data.push_back(0.0f);
            Y_onehot_data.push_back(1.0f);
            Y_onehot_data.push_back(0.0f);
            Y_class_data.push_back(1);
        } else {
            Y_onehot_data.push_back(0.0f);
            Y_onehot_data.push_back(0.0f);
            Y_onehot_data.push_back(1.0f);
            Y_class_data.push_back(2);
        }
    }

    int64_t total_samples = age_data.size();
    std::cout << "Total samples: " << total_samples << std::endl;

    // Count class distribution
    int class_counts[3] = {0, 0, 0};
    for(int c : Y_class_data) class_counts[c]++;
    std::cout << "Class distribution: Fail=" << class_counts[0] 
              << ", Average=" << class_counts[1] 
              << ", Pass=" << class_counts[2] << std::endl;

    // Combine features into X_data (row-major)
    std::vector<float> X_data;
    X_data.reserve(total_samples * 4);
    for(int i = 0; i < total_samples; i++){
        X_data.push_back(age_data[i]);
        X_data.push_back(study_hours_data[i]);
        X_data.push_back(attendance_data[i]);
        X_data.push_back(sleep_hours_data[i]);
    }

    // Create tensors
    OwnTensor::TensorOptions opts = OwnTensor::TensorOptions()
        .with_device(OwnTensor::Device::CUDA)
        .with_dtype(OwnTensor::Dtype::Float32);

    OwnTensor::Tensor X(OwnTensor::Shape{{total_samples, 4}}, opts);
    X.set_data(X_data);

    // Normalize X
    float X_min = OwnTensor::reduce_min(X).to_cpu().data<float>()[0];
    float X_max = OwnTensor::reduce_max(X).to_cpu().data<float>()[0];
    std::cout << "X min: " << X_min << ", X max: " << X_max << std::endl;
    OwnTensor::Tensor X_norm = (X - X_min) / (X_max - X_min + 1e-7f);
    std::cout << "Normalized X:" << std::endl;
    X_norm.to_cpu().display();

    // Create Y tensor (one-hot encoded, shape: N x 3)
    OwnTensor::Tensor Y(OwnTensor::Shape{{total_samples, 3}}, opts);
    Y.set_data(Y_onehot_data);
    std::cout << "Y (one-hot):" << std::endl;
    Y.to_cpu().display();

    // Create ag::Values
    ag::Value X_Value = ag::make_tensor(X_norm, "X_norm");
    ag::Value Y_Value = ag::make_tensor(Y, "Y");

    // Model: Linear(4, 3) - 4 inputs, 3 outputs (one per class)
    ag::nn::Linear layer(4, 3, OwnTensor::Device::CUDA);

    // Optimizer
    float learning_rate = 0.1f;
    ag::SGDOptimizer optimizer(layer.parameters(), learning_rate);

    // Training
    int epochs = 10000;
    float convergence = 0.00001f;
    float ploss = std::numeric_limits<float>::infinity();
    bool converged = false;

    for(int i = 1; i <= epochs; i++){
        // Forward pass
        ag::Value logits = layer(X_Value);  // Shape: (N, 3)
        
        // Cross entropy loss (includes softmax internally!)
        ag::Value loss = ag::cross_entropy_with_logits(logits, Y_Value);
        
        // Get loss value
        float closs = loss.val().to_cpu().data<float>()[0];
        
        if(i % 100 == 0){
            std::cout << "Epoch: " << i << ", Loss: " << closs << std::endl;
        }
        
        if(std::abs(ploss - closs) < convergence){
            std::cout << "Converged at epoch " << i << ", Loss: " << closs << std::endl;
            converged = true;
            ag::backward(loss);
            optimizer.step();
            optimizer.zero_grad();
            break;
        }
        
        ploss = closs;
        ag::backward(loss);
        optimizer.step();
        optimizer.zero_grad();
    }

    if(!converged){
        std::cout << "\nTraining completed, loss not converged" << std::endl;
    }

    // Total parameters
    int total_params = 0;
    for(auto& p : layer.parameters()){
        total_params += p.val().numel();
    }
    std::cout << "Total parameters: " << total_params << std::endl;

    // Accuracy calculation
    ag::Value final_logits = layer(X_Value);
    ag::Value probs = ag::softmax_row(final_logits);  // Convert to probabilities
    
    float* prob_data = probs.val().to_cpu().data<float>();
    
    int correct = 0;
    int confusion[3][3] = {{0}};  // confusion[actual][predicted]
    
    for(int i = 0; i < total_samples; i++){
        // Find argmax of predictions
        int pred_class = 0;
        float max_prob = prob_data[i * 3 + 0];
        for(int c = 1; c < 3; c++){
            if(prob_data[i * 3 + c] > max_prob){
                max_prob = prob_data[i * 3 + c];
                pred_class = c;
            }
        }
        
        int actual_class = Y_class_data[i];
        confusion[actual_class][pred_class]++;
        
        if(pred_class == actual_class){
            correct++;
        }
    }

    float accuracy = ((float)correct / total_samples) * 100.0f;
    
    std::cout << "\n=== RESULTS ===" << std::endl;
    std::cout << "Confusion Matrix:" << std::endl;
    std::cout << "              Pred:Fail  Pred:Avg  Pred:Pass" << std::endl;
    std::cout << "Actual:Fail   " << confusion[0][0] << "\t\t" << confusion[0][1] << "\t\t" << confusion[0][2] << std::endl;
    std::cout << "Actual:Avg    " << confusion[1][0] << "\t\t" << confusion[1][1] << "\t\t" << confusion[1][2] << std::endl;
    std::cout << "Actual:Pass   " << confusion[2][0] << "\t\t" << confusion[2][1] << "\t\t" << confusion[2][2] << std::endl;
    std::cout << "\nAccuracy: " << accuracy << "%" << std::endl;

    return 0;
}
