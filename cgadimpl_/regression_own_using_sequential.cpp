#include<iostream>
#include<fstream>
#include<sstream>
#include<vector>
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


}
