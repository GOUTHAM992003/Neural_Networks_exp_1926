#include<iostream>
#include<chrono>
#include<cuda_runtime.h>
#include<cstring>
void check_memory_type(void* ptr){
  cudaPointerAttributes attrs;
  cudaError_t err = cudaPointerGetAttributes(&attrs,ptr);
  std::cout<<"Memory Status : ";
  if (err != cudaSuccess || attrs.type == cudaMemoryTypeUnregistered){
    // If CUDA throws error OR says type=0 (Unregistered), it is standard Pageable.
    std::cout<<"[Pageable] (Reason: ";
    if(err!=cudaSuccess) std::cout << cudaGetErrorName(err);
    else std::cout << "Type=" << attrs.type << " Unregistered";
    std::cout << ")\n";
    
    cudaGetLastError(); //clear the error flag.
  }else{
      // If Type=1 (Host) or Type=2 (Device), it is Known/Pinned.
      std::cout<<"[Pinned/Registered] (CUDA attributes found! Type=" << attrs.type <<")\n";
  }
}
int main(){
//     std::cout<<"Memcpy tests\n";
//     size_t size=1024*1024*500; //500MiB
//     //1. Allocate source and destination memory on heap (pageable)
//     uint8_t* h_src = new uint8_t[size];
//     uint8_t* h_dst = new uint8_t[size];
//     //Initialize to force physical allocation(OS Logic)
//     std::memset(h_src,1,size);
//     std::memset(h_dst,0,size);
//     //checking by prinitng first 10 values.
//     for(int i=0;i<10;i++){
//       //std::cout<<+h_src[i]<<"\n";
//     std::cout<<+h_dst[i]<<"\n";   //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer
//      }
// //2.Memcpy (CPU --->CPU)
// auto start = std::chrono::high_resolution_clock::now();
// //Standard  C++ memcpy 
// std::memcpy(h_dst,h_src,size);
// auto end = std::chrono::high_resolution_clock::now();
// // auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
// //std::cout<<"CPU--->CPU Time(Memcpy 500 MiB) :" << duration.count() << "ms\n";
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"CPU --->CPU (Memcpy - 500MiB) time :"<<duration<<"ms\n";
// //Bandwidth (GB/s)[ (size/duration)--->(Bytes/ms) --->convert to GiB/sec ]
// double bandwidth = ((size/(1024.0*1024.0*1024.0))/(duration/1000.0));// [GiB/sec]
// std::cout<<"Bandwidth(GiB/sec) :" <<bandwidth<<"\n";
// //checking whether they got copied correctly or not.
// for(int i=0;i<10;i++){
//   //std::cout<<+h_src[i]<<"\n";
//   std::cout<<+h_dst[i]<<"\n"; //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer
     
// }
// check_memory_type(h_src);
// check_memory_type(h_dst);
// std::cin.get();
//delete[] h_src;
//delete[] h_dst;
//std::cout<<"pageable memory freed\n";
// void* src = nullptr;
// cudaError_t err = cudaMallocHost(&src,size);
// if (err!=cudaSuccess){
//   std::cout<<"Failed to allocate pinned memory : "<<cudaGetErrorString(err)<<"\n";

// }else{
//   std::cout<<"Success! Pinned Memory allocated at address : "<< src<<"\n";
//   check_memory_type(src);
//   std::cin.get();
//   //cudaFreeHost(src);
//   //std::cout<<"Memory freed\n";
// }
//cudaFreeHost(src);
// delete[] h_src;
// delete[] h_dst;
// std::cout<<"Memory Freed\n";




// --- CPU -> CPU VARIATIONS ---
size_t copy_size = 1024 * 1024 * 500; // 500 MB (Renamed to avoid conflict)

// // Allocate 4 Buffers
// uint8_t* h_pg1 = new uint8_t[copy_size]; // Pageable 1
// uint8_t* h_pg2 = new uint8_t[copy_size]; // Pageable 2
// uint8_t* h_pin1 = nullptr; cudaMallocHost(&h_pin1, copy_size); // Pinned 1
// uint8_t* h_pin2 = nullptr; cudaMallocHost(&h_pin2, copy_size); // Pinned 2

// // Initialize
// std::memset(h_pg1, 1, copy_size);
// std::memset(h_pg2, 0, copy_size);
// std::memset(h_pin1, 2, copy_size);
// std::memset(h_pin2, 0, copy_size);

// // Lambda to benchmark (Fixed Syntax!)
// auto test_cpucpu = [&](const char* name, void* dst, void* src) {
//     auto s = std::chrono::high_resolution_clock::now();
//     std::memcpy(dst, src, copy_size);
//     auto e = std::chrono::high_resolution_clock::now();
    
//     double ms = std::chrono::duration<double, std::milli>(e - s).count();
//     double gb = (copy_size / (1024.0 * 1024.0 * 1024.0)) / (ms / 1000.0);
//     std::cout << name << ": " << ms << " ms | " << gb << " GiB/s\n";
// };

// std::cout << "\n--- CPU -> CPU Variations ---\n";
// test_cpucpu("1. Pageable -> Pageable", h_pg2, h_pg1);
// test_cpucpu("2. Pageable -> Pinned  ", h_pin1, h_pg1);
// test_cpucpu("3. Pinned   -> Pageable", h_pg2, h_pin2);
// test_cpucpu("4. Pinned   -> Pinned  ", h_pin1, h_pin2);

// // Cleanup
// delete[] h_pg1; delete[] h_pg2;
// cudaFreeHost(h_pin1); cudaFreeHost(h_pin2);

// CPU to  CPU (pageable to pinned)

// uint8_t* h_pg1= new uint8_t[copy_size];
// uint8_t* h_pin1=nullptr;
// cudaMallocHost(&h_pin1,copy_size);
// std::memset(h_pg1,1,copy_size);
// std::memset(h_pin1,0,copy_size);
// for(int i =0;i<10;i++){
//   //std::cout<<+h_pg1[i]<<"\n";
//   std::cout<<+h_pin1[i]<<"\n"; //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer
// }
// check_memory_type(h_pg1);
// check_memory_type(h_pin1);
// auto start = std::chrono::high_resolution_clock::now();
// std::memcpy(h_pin1,h_pg1,copy_size);
// auto end= std::chrono::high_resolution_clock::now();
// double duration = std::chrono::duration<double,std::milli>(end-start).count();
// std::cout<<"CPU to CPU (pageable memory to pinned memory memcpy) time (500MiB) :" <<duration<<"ms\n";
// double bandwidth = ((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
// std::cout<<"Bandwidth(GiB/s) :"<<bandwidth<<"\n";
// //Checking whether the values are copied or not by printing first 10 bytes.
// for (int i=0;i<10;i++){
// //std::cout<<+h_pg1[i]<<"\n";
// std::cout<<+h_pin1[i]<<"\n"; //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer
// }
// std::cin.get();
// delete[] h_pg1;
// cudaFreeHost(h_pin1);
// std::cin.get();


// CPU to CPU (pinned to pageable)
uint8_t* h_pin1=nullptr;
cudaMallocHost(&h_pin1,copy_size);
uint8_t* h_pg1 = new uint8_t[copy_size];
std::memset(h_pin1,1,copy_size);
std::memset(h_pg1,0,copy_size);
for(int i=0;i<10;i++){
  std::cout<<+h_pin1[i]<<"\n";
  std::cout<<+h_pg1[i]<<"\n";  //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer
}
check_memory_type(h_pin1);
check_memory_type(h_pg1);
auto start = std::chrono::high_resolution_clock::now();
std::memcpy(h_pg1,h_pin1,copy_size);
auto end= std::chrono::high_resolution_clock::now();
double duration = std::chrono::duration<double,std::milli>(end-start).count();
std::cout<<"CPU to CPU (pinned memory to pageable memory memcpy) time (500 MiB) :"<<duration<<"ms\n";
double bandwidth =((copy_size/(1024.0*1024.0*1024.0))/(duration/1000.0));
std::cout<<"Bandwidth(GiB/s) :"<<bandwidth<<"\n";
for(int i=0;i<10;i++){
  std::cout<<+h_pin1[i]<<"\n";
  std::cout<<+h_pg1[i]<<"\n"; //+h_dst or +h_src is used to convert that char(uint8_t value is considered as char(ASCII)) into integer.
}
std::cin.get();
cudaFreeHost(h_pin1);
delete[] h_pg1;
std::cin.get();
}