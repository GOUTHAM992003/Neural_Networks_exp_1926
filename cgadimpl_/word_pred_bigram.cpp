#include <ad/ag_all.hpp>
#include <iostream>
#include <fstream>
#include<vector>
#include<cstdint>
std::vector<uint16_t> load_tokens(const std::string& filepath,int& out_vocab_size){
    std::ifstream file(filepath,std::ios::binary | std::ios::ate);
    if(!file){
        std::cerr << "Error opening" << filepath << std::endl;
        exit(1);
        //or else you can write like this too :
        // throw std::runtime_error("Error opening " + filepath); -->include <stdexcept> header to use runtime_error or to throw error "string"
    }
    std::streamsize size_bytes = file.tellg(); // .tellg() --->tell_get_position --->ifstream class method,"file" is an object of ifsteam class,and file.tellg() returns a std::streamsize (fancy name for long long/long int/signed int) object, tell = report ( the position of the cursor/file-pointer counter),g=get(read)
    size_t num_tokens = size_bytes/sizeof(uint16_t);
    std::cout<<" Loading "<< num_tokens <<" tokens ."<<std::endl;
    std::vector<uint16_t> tokens(num_tokens);
    file.seekg(0, std::ios::beg); // .seekg() --->seek_get_position --->ifstream class method,"file" is an object of ifsteam class,and file.seekg() --->it helps in moving the read cursor ,argument1 --->offset/how many bytes to move w.r.t anchor,and std::ios::beg --->anchor ,so here this line tells to move the cursor  0 - bytes from the beginning.
    //read data from file,--->cast tokens.data() (which returns a uint16* pointer) to char* becoz 'read' expects raw bytes.
 if(!file.read(reinterpret_cast<char*>(tokens.data()),size_bytes)){
    std::cerr << "Read failed!" <<std::endl;
    exit(1);
 }
 //finding max_id
 int max_id = 0;
 for(uint16_t t : tokens){
    if(t>max_id){
         max_id=t;
        }
}
out_vocab_size = max_id+1 ; // ex :- Max ID 100 ---> Size 101
return tokens;
}
 class BigramLanguageModel {
  public:
  ag::Value token_embedding_table; // a trainable tensor (node)
  //vocab_size : How many rows (words) we have (e.g : 50257)
  //n_embd : How many columns per word (e.g : 32) --->its like giving 32 powers/features for each token  
  ag::Value head;  // head table projects back to Vocab size,shape : (vocab_size,n_embd)
  BigramLanguageModel(int vocab_size,int n_embd){
    auto opts=OwnTensor::TensorOptions().with_device(OwnTensor::Device::CUDA).with_dtype(OwnTensor::Dtype::Float32).with_req_grad(true);
    //random weights (Normal distribution*0.02 scale)
    OwnTensor::Tensor raw_weights =OwnTensor::Tensor::randn({{vocab_size,n_embd}},opts)*0.02f;
    //wrappint it in Autograd node allows to train it.
    token_embedding_table = ag::make_tensor(raw_weights,"token_emb");
    OwnTensor::Tensor head_weights = OwnTensor::Tensor::randn({{n_embd,vocab_size}},opts)*0.02f;
    head = ag::make_tensor(head_weights,"head");
  }
  //forward pass : Tnput --->Output
  //Input : One-hot Encoded Tensor [Batch,VocabSize]
  ag::Value forward(ag::Value inputs){
    // Embedding : inputs*table
    // [B,V]@ [V,D] ---> [B,D]
    ag::Value tok_emb = ag::matmul(inputs,token_embedding_table);
    //2. Head : tok_emb *head
    // [B,D]@[D,V] -->[B,V]
    ag::Value logits = ag::matmul(tok_emb,head);
    return logits;
  }
 };
int main(){
    int vocab_size=0;
    std::vector<uint16_t> tokens = load_tokens("/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/inputs/edufineweb_train_000001.bin",vocab_size);
//or else you can use " auto tokens=load_tokens("file_path") " too ,for avoiding explicit output object type specifying.
 std::cout<<" Vocab_ Size : "<<vocab_size<<std::endl;
    //  std::cout << "Original Vocab Size: " << vocab_size << std::endl;

    // // --- EXPERIMENT: Force small vocab to prove learning works ---
    // int small_vocab = 200; // Only learn first 200 possible words
    // vocab_size = small_vocab; 

    // // Filter tokens: Replace any token >= 200 with 0 (Unknown)
    // // This allows the model to see the same "words" more frequently
    // for(auto& t : tokens) {
    //     if(t >= small_vocab) t = 0;
    // }
    // std::cout << "Reduced Vocab Size to: " << vocab_size << std::endl;
    // // -------------------------------------------------------------
 // setting up model
 int n_embd=32;
 float learning_rate =0.001f;
 BigramLanguageModel model(vocab_size,n_embd);
 //model params for optimizer
 std::vector<ag::Value> params = {model.token_embedding_table,model.head};
 ag::Adam optimizer(params,learning_rate);
 std::cout << "Starting Training"<<std::endl;
 //Training settings 
 int batch_size = 64;
 int max_steps = 1000; 
 std::ofstream loss_file("/home/blu-bridge016/Downloads/test_env_gau/benchmark_results/loss.txt");
 // Tensors need to be on GPU (Float32)
 auto opts = OwnTensor::TensorOptions().with_device(OwnTensor::Device::CUDA).with_dtype(OwnTensor::Dtype::Float32);
 for(int step =0;step<=max_steps;++step){
//one-hot batches on cpu (batch* Vocab)
//filled with zeroes 
std::vector<float>x_one_hot(batch_size*vocab_size,0.0f);
std::vector<float>y_one_hot(batch_size*vocab_size,0.0f);
for(int b=0;b<batch_size;b++){
  //1.pick random index in dataset
  //tokens.size()-1 because we need a  " next word"
  int idx = rand() % (tokens.size()-1);
  //2.Get Token IDs
  int token_id = tokens[idx]; //Input (ex : apple)
  int next_token_id = tokens[idx+1]; // Target (ex : pie)
  //3.Set "1.0" in one-hot vectors 
  //Formula : Row*Width +Col
  x_one_hot[b*vocab_size+token_id]=1.0f;
  y_one_hot[b*vocab_size+next_token_id]=1.0f;
}
//2.Upload to gpu
OwnTensor::Tensor x_tensor(OwnTensor::Shape{{batch_size,vocab_size}},opts);
x_tensor.set_data(x_one_hot);
OwnTensor::Tensor y_tensor(OwnTensor::Shape{{batch_size,vocab_size}},opts);
y_tensor.set_data(y_one_hot);
//wrap in autograd value
ag::Value inputs = ag::make_tensor(x_tensor,"inputs");
ag::Value targets = ag::make_tensor(y_tensor,"targets");
//forward pass
ag::Value logits = model.forward(inputs);
//loss(cross-entropy)
ag::Value loss=ag::cross_entropy_with_logits(logits,targets);

//backprop
ag::backward(loss);
//update params
optimizer.step();
//zero_grad
optimizer.zero_grad();

//loss

if(step %10 ==0){
  float val = loss.val().to_cpu().data<float>()[0];
  std::cout<<"step :"<<step<<" Loss : " <<val << std::endl;
  loss_file<<step<<","<<val<<"\n";
}
 } 
 }
