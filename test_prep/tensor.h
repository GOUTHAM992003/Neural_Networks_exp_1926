#pragma once 
#include<iostream> //to inlcude standard input and output functionalities. 
#include<vector> //for storing shape dimensions.
#include<cstdint> //for uint8_t,uint16_t,int64_t --->exact width integers.
#include<cstddef> //for size_t --->platform dependent unsigned integer type.
#include<cstring> //for std::memcpy --->used for BF16 bit manipulation later.
#include<string> //for std::string
#include<stdexcept> //for std::runtime_error.

/*Now, let's define the Dtype. We need a way to tell the Tensor if its data is Float32 (4 bytes)
 or Bfloat16 (2 bytes). We use an enum for this.
 enum is  just a named list of options. Instead of using raw numbers like 0 or 1 (it'll be  confusing),
  we can now use names like Dtype::Float32,Bfloat16 .
 */

enum class Dtype{
    Float32,
    BFloat16
};

//lets create tensor class 

class Tensor{
private:
    uint8_t* data; //memory address where the actual numbers are stored (raw bytes) .
    std::vector<int64_t> shape; //stores thedimensions of tensor .
    Dtype dtype; //Float32 or Bfloat16 .
    size_t size ; //total no.of elements in a tensor (product of elements of shape)

    //Helper function to get size of dtype in bytes
    size_t dtype_size(Dtype dt) const {
        return (dt==Dtype::Float32)?sizeof(float):sizeof(uint16_t);
    }
    /*In C++, putting const at the end of a member function like size_t dtype_size(Dtype dt) const means two important things:

A Promise: You are promising the compiler that this function will not change any of the class's variables (like data, shape, or size). It only reads information, it never writes.
Compatibility: It allows this function to be called on a "Constant Tensor". If you have a const Tensor t, the compiler will only allow you to call functions that are marked with const.
Since dtype_size just returns a number and doesn't change our tensor, it's a "Read-Only" function, so we mark it as const*/

public:
//Constructor :creates the tensor and allocates memory 
Tensor(std::vector<int64_t> s,Dtype dt):shape(s),dtype(dt){
/*This syntax is called a Member Initializer List, and it's actually the "proper" way to initialize variables in C++.

Think of it this way:

Method A: Assignment (Inside the {})
Tensor(std::vector<int64_t> s) {
    shape = s; // This is like: Create an empty shape, then COPY s into it.
}
Method B : Initialization (The :shape(S) part)
Tensor(std::vector<int64_t> s):shape(s){
//This is like : Create s,andfill it with s at the same moment. 
}
Why Method B is better:

Efficiency: It skips the step of creating something "empty" just to overwrite it a millisecond later.
Required for some types: If you have a const variable or a "Reference" variable in your class, 
C++ forces you to use the initializer list because those types cannot be changed once they are created.
So, : shape(s), dtype(dt) is just telling the computer: 
"When you build this Tensor, immediately set the shape to s and the dtype to dt."
Both methods result in a copy, but they happen at different times:

Assignment (Method A - Inside { }):

The computer first builds an empty shape (an empty list).
Then it enters the constructor body and sees shape = s;.
It replaces that empty list with a copy of s.
Result: 1 Creation + 1 Replacement.
Initializer List (Method B - The : shape(s) part):

The computer builds the shape and fills it with s at the exact same moment.
Result: 1 Creation.
It’s like the difference between buying an empty house and then moving furniture in,
 versus buying a house that already has the furniture you wanted inside it!
 In C++, one-step construction is always the "gold standard."
*/
    //1.Calculate total size in bytes
    size=1;
    for(auto dim:shape){
        size*=dim;
        }
    //2.Calculate how many raw bytes we need to allocate
    size_t nbytes = size* dtype_size(dtype);

//Allocate memory 
        data = new uint8_t[nbytes];

//Initialize to zero (for not getting garbage values)
std::memset(data,0,nbytes);
    }

//destructor : Cleans up the memory when the tensor is no longer needed 
~Tensor(){
        delete[] data;
    }
/*What is the ~? The tilde ~ symbol followed by the class name defines the Destructor.
It runs automatically when a Tensor object goes out of scope or is deleted. 
delete[] data tells the computer: "I'm done with these bytes, you can have them back now."*/

/* let's add some "Accessors" (or Getters). 
Since our variables are private, we need a way to see them from the outside.*/

//Returns th total no.of elements in a tensor 
size_t numel() const{
        return size;
    }

//Retunrs the shape of the tensor (as a list)
void get_shape() const{
    std::cout<<"[";
    for(size_t i=0;i<shape.size();i++){
        std::cout<<shape[i]<<(i==shape.size()-1?"":",");
        }
    std::cout<<"]"<<std::endl;
    }

//Returns the dtype
std::string get_dtype() const{
    return (dtype==Dtype::Float32)?"Float32":"BFloat16";
    }
/* the const at the end again? It’s us promising: "I’m just looking at the value, I won't change it."*/
/*our data is stored as uint8_t* (raw bytes). But if we want to add numbers, we need a float* or a uint16_t*.

In C++, we use two powerful tools for this: Templates and reinterpret_cast.*/

};
