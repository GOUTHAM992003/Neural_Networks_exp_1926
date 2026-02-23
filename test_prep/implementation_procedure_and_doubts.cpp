/*So,as we are impleenting 3-tasks : 
1)Base tensor class implementation ; 
2)FP32,BF16 implementation and bit-manipulation logic for conversion functions ;
3)One-reduction operation implementation (ex : reduce_sum)


So,what is a Tensor code-wise? --->>>> 
1)Data --->a block of raw bytes in memory;
2)Shape --->the dimensions;
3)Dtype--->what type of data is stored in the tensor.  

Now,firstly let's talk about Header files and source files(.cpp files) in C++,
and why we shouldnt opt for Header-only approach for the libraries of this scale(deeplearning framework),
,but for this task of implementing these 3 tasks,one header file is enough ,
 or else,can be done in a single cpp file too ,
 and can we include both header files and cpp files? and Are we doing it ?(YES)

 --->Yes ,we first implement these three tasks in tensor.h,
 and include this tensor.h in a test-file ,it just copies all the code in tensor.h in that test-file ,and compiles it everytime we include it ,and if we do 
and then we will rename tensor.h as tensor.cpp file,and includes it in that test file ,
and runs it,still it runs and compiles whole thing like earlier,then whats the difference?

Usually ,in cpp --->The Classic Rule : "Keep declarations in Header(.h) files ,and definitions in Source(.cpp) files".
but in this case of implementing the above 3-tasks ,why header-only approach ?
1)Fewer files = cleaner live coding while giving seminar,and including  just one files is enough.
2)Templates must be in headers :

ex code :
template<typename T>
T* Tensor::data(){
return reinterpret_cast<T*>(raw_ptr);
}
 this template code cannnot go in a .cpp file,it must be in a .h file ? (why?)
 --->becoz the compiler needs to see the full template body at the point where its used,
 when a test-file calls tensor.data<float>(),the compiler needs to generate the specific "float" version right there.
 If the template body is in a surce file (.cpp file),the compiler can't see it from that test-file,as we include only header file.
 This is called template instantiation --->the compiler creates a separate version for each type.
 our deeplearning library is heavily template-based(dispatch_by_dtype,dtype_traits<>,dispatch_reduction<T,Sumop>),
 so most code has to live in headers.
 
 3)Inline functions and performance :
 when you put a function body in a header,the compiler can inline it,
  it replaces the function call function call with the actual ode at the call site.
  this avoids function call  overead ,which matters in a tensor library that process millions of elements .

ex : In Types.h --->compiler can inline this bf16 to fp32 conversion function(zero - overhead) 
float bfloat16_to_float(uint16_t b){
uint32_t u = (uint32_t)(b) << 16;
float f ;
std::memcpy(&f,&u,sizeof(f));
return f;
}

Qn)Cant we just #include a .cpp file ?
technically ,we can do that ,#include just copy-pastes text ,this would just copies entire file and compiles each time its included :
it works ,but dont do that,becoz it re-compiles everytime instead of jsut giving already-compiled binaries ,as usually .cpp files involes many no.of lines of codes .
and also One-Definiton-Rule(ODR) violation :
If  file_a.cpp and file_b.cpp both #include "tensor.cpp",then :
--->file_a.o contains Tensor::display() definiton,
--->file_b.o contains Tensor::display() definiton,
linker error --->"multiple definiton of Tensor::display()"
Header files avoids this becoz,declarations can appear multiple times--->no prblm,
Template definitons in headers are OK becoz templates get specal treatment (each .o file keeps its own copy,linker deduplicates)
inline keyword tells the linker :"multiple definitons are OK,just pick one" .

at the scale of a deep-learning library --->using hybrid approach is best,using headerfiles for imlementations when tmplates or inline funcitons defined is best,whereas usual .cpp files for implementaitons/definitons in normal case is best .


Now,coming to data pointer for a tensor,why uint8_t* ,not float*,or else void* pointer?
 ---->If we use float*,we cant store BF16 or other lower types ,and bare minimumm it reads is 4-bytes(32 bytes --->float) 
 so ,we use raw_bytes(uint8_t*) and then cast hen we want other dtypes.
 ex:
 float* fp32_ptr = reinterpret_cast<float*>(data_);  //when dtype is float32 .
 uint16_t* bf16_ptr = reinterpret_cast<uint16_t*>(data_); //when dtype is BFloat16 .

 then why cant void* pointer ?
 What actually a pointer is ?
 A pointer stores a memory address ,thastit ,On a 64-bit system,every pointer is 8-bytes regardless of type .

 float* p1 ; //8-bytes - stores an address .
 uint8_t* p2 ; //8-bytes - stores an address .
 uint16_t* p3 ; //8-bytes - stores an address .
 uint32_t* p4 ; //8-bytes - stores an address .
 uint64_t* p5 ; //8-bytes - stores an address .
  all of these are the same size,the address itself doesnt change based on type of pointer .

  So,what does type do ?
  The type tells the compiler 2-things :
  1)/how many bytes to read/writewhen you dereference 

  float* fp=(float*)0x1000;
  uint16_t* hp=(uint16_t*)0x1000;
  uint8_t* bp=(uint8_t*)0x1000;

  same address 0x1000,but when ou do *fp,*hp,*bp,it reads 4,2,1 bytes respectively .
  Memory at address 0x1000:
Address:  0x1000  0x1001  0x1002  0x1003  0x1004  0x1005 ...
Bytes:    [0x40]  [0x49]  [0x0F]  [0xDB]  [0xAA]  [0xBB] ...

*bp  → reads 1 byte  → 0x40
*hp  → reads 2 bytes → 0x40, 0x49 → combined as uint16_t
*fp  → reads 4 bytes → 0x40, 0x49, 0x0F, 0xDB → interpreted as float (= 3.14159...)

 The type determines how many bytes the CPU reads starting from that address.

 2)How pointer arithmetic works(p+1) ?
  float* fp = (float*)0x1000;
  uint16_t* hp=(uint16_t*)0x1000;
  uint8_t* bp=(uint8*)0x1000;

  fp+1 ---> 0x1000+4 = 0x1004 (jumps 4 bytes,sizeof(float))
  hp + 1  →  0x1000 + 2  =  0x1002   (jumps 2 bytes, sizeof(uint16_t))
bp + 1  →  0x1000 + 1  =  0x1001   (jumps 1 byte,  sizeof(uint8_t))
So p[i] is actually 

*(p + i)
, which means:

fp[3] → reads 4 bytes starting at 0x1000 + 3*4 = 0x100C
hp[3] → reads 2 bytes starting at 0x1000 + 3*2 = 0x1006
bp[3] → reads 1 byte starting at 0x1000 + 3*1 = 0x1003 .

Now, why uint8_t* for our tensor?
Say we allocate 12 bytes of raw memory:
Address:  0x1000  0x1001  0x1002  0x1003  0x1004  ...  0x100B
Bytes:    [  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ][  ]
          |______________|______________|______________|
          float[0]=4bytes  float[1]=4bytes  float[2]=4bytes   ← if Float32 (3 elements)
          |________|________|________|________|________|____|
          bf16[0]    bf16[1]   bf16[2]  bf16[3]  bf16[4]  bf16[5]  ← if BF16 (6 elements)

          The same 12 bytes can hold:

3 floats (4 bytes each) — accessed as float*
6 bfloat16s (2 bytes each) — accessed as uint16_t*
So we store the raw pointer as uint8_t* (one byte = smallest unit), and then cast when we need to access:

#include<iostream>
int main(){
uint8_t* data_ = new  uint8_t[12]; //allocates 12 raw bytes .
//when dtype is fp32 :
float* fp=reinterpret_cast<float*>(data_);
fp[0]=3.14; //writes 4 bytes at data_[0..3] 
fp[1]=2.17; //writes 4 bytes at data_[4..7]
std::cout<<fp[0]<<" "<<fp[1]<<std::endl;
//when dtype is BFloat16 :
uint16_t* hp = reinterpret_cast<uint16_t*>(data_);
hp[0]=0x1926; //writes 2 bytes at data_[0..1]
hp[1]=0x1925; //writes 2 bytes at data_[2..3]
std::cout<<hp[0]<<" "<<hp[1]<<std::endl; 
}
 
Why not void*?
void* can also store any address, BUT:

You cannot do void_ptr + 1 — the compiler doesn't know the step size,no pointer arithmetic
You cannot do new void[12] — can't allocate void
uint8_t* lets you do byte-level arithmetic: data_ + offset moves exactly offset bytes,
and uint8_t* gives us byte_level control over memory that can hold any dtype.

 void* does NOT read one byte. void* reads nothing — the compiler doesn't know the size.
 void* vs uint8_t* — what the compiler actually allows :

 void*    vp = malloc(12);
uint8_t* bp = (uint8_t*)malloc(12);

// DEREFERENCE — reading the value at the address
*bp;        //  Compiles. Reads 1 byte, returns uint8_t
*vp;        //  COMPILER ERROR: "void* is not a pointer-to-object type"

// INDEXING
bp[3];      //  Compiles. Reads 1 byte at offset 3
vp[3];      //  COMPILER ERROR: same reason

// POINTER ARITHMETIC
bp + 5;     //  Compiles. Moves 5 bytes forward (5 * sizeof(uint8_t) = 5)
vp + 5;     //  COMPILER ERROR: "arithmetic on void*"

// REINTERPRET_CAST
reinterpret_cast<float*>(bp);   //  Compiles
reinterpret_cast<float*>(vp);   //  Compiles — this DOES work!

So yes, you can reinterpret_cast from void* to float*. That part works the same. But here's where void* fails us:
The problem: memory allocation and offset calculation
When we create a tensor, we need to do things like:
// Allocate raw bytes
data_ = new uint8_t[nbytes];    //  works — allocates nbytes 

// With void:
data_ = new void[nbytes];       //  COMPILER ERROR: cannot allocate void

And when we compute offsets (for views/slicing in real tensors):
// "give me a pointer offset by 8 bytes"
uint8_t* offset_ptr = data_ + 8;    //  moves 8 bytes

void* offset_ptr = data_ + 8;       //  COMPILER ERROR: can't do arithmetic on void*
//You'd have to cast first every single time:
void* offset_ptr = (void*)((uint8_t*)data_ + 8);  // ugly and defeats the purpose

void* is essentially a "dumb address" — it holds an address but the compiler refuses to let you do anything useful with it directly. uint8_t* is a "smart byte-level address" — it holds the same address but the compiler knows each element is 1 byte, so all operations work.

That's why every tensor library uses uint8_t* (or char*, which is the same size) for raw storage, never void*.

