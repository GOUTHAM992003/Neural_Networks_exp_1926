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
    std::vector<float> exam_result_data;
std::string line;
std::ifstream file("/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/inputs/Exam_Score_Prediction.csv");
std::getline(file,line);
while(std::getline(file,line)){
    std::stringstream ss(line);
    std::string cell;
    std::getline(ss,cell,',');
    std::getline(ss,cell,',');
    age_data.push_back(std::stof(cell));
    std::getline(ss,cell,',');
    std::getline(ss,cell,',');
    std::getline(ss,cell,',');
    study_hours_data.push_back(std::stof(cell));
    std::getline(ss,cell,',');
    attendance_data.push_back(std::stof(cell));
    std::getline(ss,cell,',');
    std::getline(ss,cell,',');
    sleep_hours_data.push_back(std::stof(cell));
    std::getline(ss,cell,',');
    std::getline(ss,cell,',');
    std::getline(ss,cell,',');
    std::getline(ss,cell,',');
    std::getline(ss,cell,',');
    exam_result_data.push_back((std::stof(cell))>50.0f?1.0f:0.0f);
}
std::cout<<"age_data | study_hours_data | attendance_data | sleep_hours_data | exam_result_data" << std::endl; 
for(int i=0;i<6;i++){
std::cout<<age_data[i]<<" | "<<study_hours_data[i]<<" | "<<attendance_data[i]<<" | "<<sleep_hours_data[i]<<" | "<<exam_result_data[i]<<std::endl;
}
int total_samples=study_hours_data.size();
std::cout<<"Total no.of samples :"<<total_samples<<std::endl;
std::vector<float> X_data;
for(int i=0;i<total_samples;i++){
    X_data.push_back(age_data[i]);
    X_data.push_back(study_hours_data[i]);
    X_data.push_back(attendance_data[i]);
    X_data.push_back(sleep_hours_data[i]);
}
// for(int i=0;i<6;i++){
//     std::cout<<X_data[i]<<std::endl;
// }
OwnTensor::TensorOptions opts = OwnTensor::TensorOptions().with_device(OwnTensor::Device::CUDA).with_dtype(OwnTensor::Dtype::Float32);
OwnTensor::Tensor X(OwnTensor::Shape{{total_samples,4}},opts);
X.set_data(X_data);

// Normalize X features (min-max normalization to [0, 1])
// Get min and max for each column
float X_min = OwnTensor::reduce_min(X).to_cpu().data<float>()[0];
float X_max = OwnTensor::reduce_max(X).to_cpu().data<float>()[0];
std::cout << "X min: " << X_min << ", X max: " << X_max << std::endl;

// Normalize: X_norm = (X - X_min) / (X_max - X_min)
OwnTensor::Tensor X_norm = (X - X_min) / (X_max - X_min + 1e-7f);

std::cout << "Normalized Tensor X" << std::endl;
X_norm.to_cpu().display();
OwnTensor::Tensor Y(OwnTensor::Shape{{total_samples,1}},opts);
 Y.set_data(exam_result_data);
 //std::cout<<"First 5-samples of Y"<<std::endl;
//  for(int i=0;i<5;i++){
//     std::cout<<Y.to_cpu().data<float>()[i]<<std::endl;
//  }
std::cout<<"Tensor Y"<<std::endl;
 Y.to_cpu().display();

 ag::Value X_Value=ag::make_tensor(X_norm, "X_norm");
 ag::Value Y_Value=ag::make_tensor(Y,"Y");
//  ag::nn::Linear layer(4, 1, OwnTensor::Device::CUDA);
ag::nn::Linear layer(4,1,opts.device.device); //opts.device --->DeviceIndex struct; opts.device.device --->Device(the enum); opts.device.index --->GPU index (int)
 float learning_rate=0.1f;
ag::SGDOptimizer optimizer(layer.parameters(),learning_rate);
int epochs=5000;
float convergence=0.000001;
float ploss=std::numeric_limits<float>::infinity(); //or else you can use float ploss=1e10f ;
bool converged=false;
// Create epsilon tensor to prevent log(0)
OwnTensor::Tensor eps_tensor = OwnTensor::Tensor::full(
    OwnTensor::Shape{{total_samples, 1}}, opts, 1e-7f
);
ag::Value eps = ag::make_tensor(eps_tensor, "eps");

// Create ones tensor for (1 - Y) and (1 - Y_pred)
OwnTensor::Tensor ones_tensor = OwnTensor::Tensor::full(
    OwnTensor::Shape{{total_samples, 1}}, opts, 1.0f
);
ag::Value ones = ag::make_tensor(ones_tensor, "ones");

// Create zeros tensor for negation
OwnTensor::Tensor zeros_tensor = OwnTensor::Tensor::full(
    OwnTensor::Shape{{total_samples, 1}}, opts, 0.0f
);
ag::Value zeros = ag::make_tensor(zeros_tensor, "zeros");
for(int i=1;i<=epochs;i++){
    ag::Value logits=layer(X_Value);
    ag::Value Y_pred=ag::sigmoid(logits);
 // BCE = -mean( Y * log(Y_pred + eps) + (1-Y) * log(1-Y_pred + eps) )
    ag::Value log_pred = ag::log(Y_pred + eps);
    ag::Value log_1_minus_pred = ag::log(ones - Y_pred + eps);
    ag::Value one_minus_Y = ones - Y_Value;
    
    // bce_neg = Y*log(pred) + (1-Y)*log(1-pred) which is NEGATIVE
    ag::Value bce_neg = Y_Value * log_pred + one_minus_Y * log_1_minus_pred;
    
    // loss = -mean(bce_neg) = mean(zeros - bce_neg)
    ag::Value neg_bce = zeros - bce_neg;
    ag::Value loss = ag::mean_all(neg_bce);
    
    // Get loss value
    float closs = loss.val().to_cpu().data<float>()[0];
    if(i%100==0){
        std::cout<<"Epoch: "<< i << ", Loss: "<< closs <<std::endl;
    }
    if((closs<convergence) || ((std::abs(ploss-closs)<convergence) && (closs<0.6))){
        std::cout<<"Loss converged at "<<i<<" epoch"<<std::endl;
        std::cout<<"Loss value at "<<i<<" epoch: "<<closs<<std::endl;
        converged=true;
        ag::backward(loss);
        optimizer.step();
        optimizer.zero_grad();
        break;
    }
    ploss=closs;
    ag::backward(loss);
    optimizer.step();
    optimizer.zero_grad();
}
if(!converged){
    std::cout<<"\n Training completed,loss not converged,epoch reached,try with diff hyperparameters"<<std::endl;
}
int total_params=0;
for(auto& p:layer.parameters()){
    total_params+=p.val().numel();
}
std::cout<<"Total no.of parameters: "<<total_params<<std::endl;
ag::Value logits = layer(X_Value);
ag::Value Y_pred = ag::sigmoid(logits);
//OwnTensor::Tensor Y_pred_tensor = Y_pred.val().to_cpu();
float* prob = Y_pred.val().to_cpu().data<float>();
float* actual = Y.to_cpu().data<float>();
int correct=0;
int TP=0, TN=0, FP=0, FN=0;
int total_positives=0, total_negatives=0;
for(int i=0;i<total_samples;i++){
    float pred =(prob[i]>0.5f) ?1.0f :0.0f ;
    if(actual[i] == 1.0f) total_positives++;
    else total_negatives++;
    
    if(pred==actual[i]){
        correct++;
        if(pred==1.0f) TP++;
        else TN++;
    } else {
        if(pred==1.0f) FP++;
        else FN++;
    }
}
std::cout << "\\nClass distribution: Positives=" << total_positives << ", Negatives=" << total_negatives << std::endl;
std::cout << "TP=" << TP << ", TN=" << TN << ", FP=" << FP << ", FN=" << FN << std::endl;
float accuracy = ((float)correct / total_samples) * 100.0f;
std::cout<<"Accuracy: "<<accuracy<<"%"<<std::endl;
}
