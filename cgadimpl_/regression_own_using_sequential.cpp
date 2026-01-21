#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
#include<limits>
#include<ad/ag_all.hpp>
int main(){
    std::vector<float> X_data;
    std::vector<float> Y_data;
    std::string line;
    std::ifstream file("/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/inputs/Salary_dataset.csv");
    std::getline(file,line);
    while(std::getline(file,line)){
        std::stringstream ss(line);
        std::string cell;
        std::getline(ss,cell,',');
        std::getline(ss,cell,',');
        X_data.push_back(std::stof(cell));
        std::getline(ss,cell,',');
        Y_data.push_back(std::stof(cell));
    }
    size_t n=X_data.size();
    auto opts=OwnTensor::TensorOptions().with_device(OwnTensor::Device::CUDA).with_dtype(OwnTensor::Dtype::Float32);
    //if not auto ,u need to catch that opts in OwnTensor::TensorOptions object,like this : OwnTensor::TensorOptions opts = OwnTensor::TensorOptions().with_device(Owntensor::Device::Cuda).with_dtype(OwnTensor::Dtype::Float32);
    OwnTensor::Tensor X(OwnTensor::Shape{{(int64_t(n)),1}},opts);
    X.set_data(X_data);
    std::cout<<"X (Years_of_Experience):"<<std::endl;
    X.to_cpu().display();
    OwnTensor::Tensor Y(OwnTensor::Shape{{(int64_t(n)),1}},opts);
    Y.set_data(Y_data);
    std::cout<<"Y (Salary):"<<std::endl;
    Y.to_cpu().display();
float X_max = OwnTensor::reduce_max(X).to_cpu().data<float>()[0];
float X_min = OwnTensor::reduce_min(X).to_cpu().data<float>()[0];
float Y_max = OwnTensor::reduce_max(Y).to_cpu().data<float>()[0];
float Y_min = OwnTensor::reduce_min(Y).to_cpu().data<float>()[0];
OwnTensor::Tensor X_norm = (X-X_min)/(X_max-X_min);
OwnTensor::Tensor Y_norm = (Y-Y_min)/(Y_max-Y_min);
ag::Value X_Value = ag::make_tensor(X_norm);
ag::Value Y_Value = ag::make_tensor(Y_norm);
std::cout<<"Input Tensor(features) :"<<std::endl;
X_Value.val().to_cpu().display();
std::cout<<"Output Tensor(labels) :"<<std::endl;
Y_Value.val().to_cpu().display();
ag::nn::Linear layer1(1,4,OwnTensor::Device::CUDA);
ag::nn::Linear layer2(4,8,OwnTensor::Device::CUDA);
ag::nn::Linear layer3(8,1,OwnTensor::Device::CUDA);
ag::nn::ReLU relu;
ag::nn::Sequential model({&layer1,&relu,&layer2,&relu,&layer3});
float convergence=0.00001;
// float ploss=1e10f; //we can use this also,a big number instead of infinity ( by including std::limits)
float ploss=std::numeric_limits<float>::infinity();
bool converged=false;
float learning_rate=0.1f;
int epochs =1000;
// std::vector<ag::Value> all_params;
// for(auto& p : layer1.parameters()){
//     all_params.pushback(p);
// };
// for(auto& p : layer2.parameters()){
//     all_params.pushback(p);
// };
// for(auto& p : layer3.parameters()){
//     all_params.pushback(p);
// };
//instead of storing all parameters in a vector and then passing them to optimizer,we can pass like model.parameters() to optimizer ,where "model" is "sequential-class" object.
ag::SGDOptimizer optimizer(model.parameters(),learning_rate);
for (int i=1;i<epochs;i++){
ag::Value Y_pred=model(X_Value);
ag::Value loss=ag::mse_loss(Y_pred,Y_Value);
float closs=loss.val().to_cpu().data<float>()[0];
if(i%100==0){
std::cout<<"loss value at:"<<i<<" th epoch is :"<<closs<<std::endl;
}
if((closs<convergence) || ((std::abs(ploss-closs)<convergence) && (closs<0.005)) ){
    std::cout<<"loss converged at "<<i<<"epoch"<<std::endl;
    std::cout<<"loss value at:"<<i<<" th epoch is :"<<closs<<std::endl;
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

//method1 to calculate total no.of parameters(weights + biases) :
// int total_params=0;
// for(auto& p : layer1.parameters()){
//     total_params+=p.val().to_cpu().numel();
// }
// for(auto& p : layer2.parameters()){
//     total_params+=p.val().to_cpu().numel();
// }
// for(auto& p : layer3.parameters()){
//     total_params+=p.val().to_cpu().numel();
// }
// std::cout<<"total parameters :"<<total_params<<std::endl;

// methhod2 of calculating total no.of parameters (weights + biases) :
// int total_params=0;
// for(auto& p : all_params){ //all_params is a vector of ag::Value objects defined above.
//     total_params+=p.val().to_cpu().numel(); 
// }
//if you dont want auto& ---> you can use "std::vector<ag::Value> p : " 
//Method3 of Calculating total no.of parameters (weights + biases):
int total_params = 0;
for(auto& p:model.parameters()){
//total_params+=p.val().to_cpu().numel();
total_params+=p.val().numel();
};
std::cout<<"total-parameters:"<<total_params<<std::endl;
ag::Value Y_pred_final_norm=model(X_Value);
OwnTensor::Tensor Y_pred_final=Y_pred_final_norm.val();
float Y_range=Y_max-Y_min;
OwnTensor::Tensor Y_predT=Y_pred_final*Y_range+Y_min;
std::cout<<"\n Original values Vs Predictions"<<std::endl;
std::cout<<"Sample  | Original | Predicted | error | percentage error " <<std::endl;
for(int i =1;i<Y_data.size();i++){
    float original = Y_data[i];
    float predicted = Y_predT.to_cpu().data<float>()[i];
    float error = original - predicted ;
    float percentage_error = (error/original)*100;
    std::cout<<i<<" | "<<original<<" | "<<predicted<<" | "<<error<<" | "<<percentage_error << std::endl;
}
float mean= OwnTensor::reduce_mean(Y_predT).to_cpu().data<float>()[0];
float rss = OwnTensor::reduce_sum(OwnTensor::square((Y-Y_predT))).to_cpu().data<float>()[0];
float tss = OwnTensor::reduce_sum(OwnTensor::square((Y-mean))).to_cpu().data<float>()[0];
float r2=1-(rss/tss);
std::cout<<"R2 score :"<<r2<<std::endl;
std::cout<<"R2 score percentage  :"<<r2*100<<"%"<<std::endl;
}


