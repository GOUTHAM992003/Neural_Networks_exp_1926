Reduction Operations :







Reduction Utilities :

1.Normalize axes 
2.Calculate output shape
3.Calculate reduced count
4.Unravel index
5.Ravel index


1. Normalize axes :
(input :input_dims vector(shape of a tensor),axes vector(the dimensions over which reduction happens),output : output_dims vector)

--->converts the user's input set of axes(which might contain duplictes or -ve indices) into a clean sorted list of +ve indices.(from zero to n-1).

why we use this :
1.to handle -ve indices
2.To remove duplicates
3.To sort the axes
4.To Validate that he indices are within  the correct bounds for the tensor's rank.

Logic :
1.Full reduction check :
If the input axes vector is empty ,the function assumes the user wants to reduce over all dimensions,So it fills the unique_axes_set with all the indices from 0 to ndim-1 .

2.Negative index handling (or) Normalization :
It loops  through the axes vector and checks if any axis is negative  and 
if any axis<0 , it adds ndims to convert it to a positive index , ex: If axis=-1 and rank-4 tensor,reduction happens over 4+(-1)=3 rd dimension.

3.Bounds checking/Validation : After normalisation,it checks if the index is within the valid range [0, ndim-1] ,if not throws runtime error.

4.Uniqueness and sorting : It uses a std::set to automatically handle duplicates and sort the axes in ascending order .

5.Final output : the elements fromthe set are copied back into a sorted std::vector for the final return value .

doubts explain ::
1.why we use normalize_axis ,a temporary variable ?(for debugging purposes,loss of original data(copy of axis) for error reporting).
2.Why we use int64_t for axes and normalized_axes , though we use at max 3 or 4 dimensional tensors ?(-ve handling,large dimensions of tensors,to maintain same as other variables with which comaprision happens in that function).
3.Why not returning by reference when we are taking input as const reference ?(related to storage anf lifetime of the object,leading to dangling pointer) .

2. Calculate_Output_Shape : 
input : 1.input_dims vector(shape of a tensor),2.normalized_axes vector,3.keepdim boolean flag
output : output_dims vector 

--->What it does : Takes the original shape and the list of axes being reduced and calculates the shape of the resulting output tensor.

Why we use this :
The tensor operation needs to know the exact dimensions of the output before allcoating the memory for the result tensor .

Logic and How : 
1.Iteration : It loops through every dimension index(i) of the input shape vector for reduction_check 
2.Reduction_check : Uses a lambda function is_reduced with std::find to check if the current dimension index(i) is in the Normalized_axes list/vector .

3.Logic Branching :
If "i" is an axis being reduced :
1.If keep_dim is true :
it pushes a size of 1 to the output shape vector (ex : 4*3*5 reduces axis 1 with keepdim=true,results in 4*1*5 output shape )
2.If keep_dim is false :
it simply ignores the reduced dimension and doesnt add it to the output shape vector (ex : 4*3*5 reduces axis 1 with keepdim=false,results in 4*5 output shape )
and if i is not an axis being reduced it simply adds the dimension size to the output shape vector (ex : 4*3*5 reduces axis 1 with keepdim=true,results in 4*1*5 output shape ) .

4.Handle scalar output :
After the loop if the output shape vector is empty it means the reduction has reduced all dimensions to a single value (scalar output) so it pushes a size of 1 to the output shape vector .

Doubts explain :
1) why size_t for output_dims ? not int64_t i or int i in for loop ?(the dangers of signed/unsigned comparisions due to implicit conversion from signed to unsigned when we compare signed and unsigned numbers,this comparsion works fine in most of the cases but fails catastrophically when the no.is negative)  
(explain backward loop pattern : edge case 
for(int64_t i =input_dims.size()-1;i>=0;--i))--->in the case of empty tensor , -1 is converted to size_t which is huge +ve number ,causing infinite loop,when implicitly converted t unsigned itn as its comapred to size_t (unsigned)
and explain conversion risks(using static_cast) incase of :
1.large to small(int64_t to int32_t) --->dataloss or overflow (huge +ve numbners becomes -ve numbers,data corruption).
2.small to large(int32_t to int64_t) --->no data loss or overflow(no danger ).
3.signed to unsigned(int to size_t) --->Danger, (-ve) nuumbers becomes huge +ve numbers .


3.Calculate_reduced_count :
input : 1.input_dims vector(shape of a tensor),2.normalized_axes vector
output : reduced_count --->int64_t
--->used by only mean and nan-mean ops .
--->calculates the total number of elements that will be combined for each reduction slice (or) 
calculates the total no.of input elements that will be combined to calculate the output element .

Why we use this :
This count is the divisor for reduce_mean and reduce_nan_mean ops .

logic and How :
1.Full reduction check : If normalized_axes is empty ,it means all dimensions are being reduced .
it uses std::accumulate with std::multiplies to calculate the product of all dimensions in input_dims .

2.Targeted reduction/Partial reduction :
it initializes a count variable to 1 and then multiplies it with the size of each dimension in normalized axis .



4.Unravel Index : (successive division and modulo method)
Converts a linear index to a multi-dimensional coordinate vector (Unravels).
  This assumes C-order (row-major) layout.
  input : 1. linear_index (offset);2.shape vector(shape of a tensor).
 Output :A vector of coordinates (e.g., {i, j, k}).

 Why we use this :
 Locating elements :
 This is crucuial for reduction operations as it helps in accessing the correct element in the input tensor for each output element.
 -helps in indexing (reduction_kernel).
 Given the linear index of an o/p element ,we need to unravel it to find its corresposding multi-dimensional co-ordinates in the i/p tensor.

 Logic and how : This uses the standard C-order(row-major) layout to convert the linear index to multi-dimensional co-ordinates .

 Indexing Logic :
  1.Backward Iteration : The loop iterates backward from the last/innermost dimension to the first /outermost .
  2.Modulo % : coords[i] = temp_index % shape[i] gives the index within the current dimensions .
  3.Division : temp_index/=shape[i] updates the linear index to the next higher dimension .
  4.Repeat : The loop repeats until all dimensions are processed .

When a tensor is stored in memory ,it exists as a single flat ,linear list of numbers.
The position of any element in this 1D element is called linear index or offset .
The unravel index funciton takes a linear index (like 37) and a shape vector like {4,5,6} and returns the multi-dimensional co-ordinates where that element lives in that tensor {1,1,1}-C-order ; {1,4,1} - Fortran order  .

C-order storage : offset = C0*(D1*D2) + C1(D2) + C2(1) 
step1 :  C2 = offset%D2 ; next_offset = C0(D1) + C1 (result of division)
step 2 : C1 = next_offset%D1 ; next_offset1 = C0 (result of division)
step3 : C0 = next_offset1 %D0 .
C0,C1,C2 are the co-ordinates and D0,D1,D2 are the dimensions .
(to understand this,see gautam_1926 notes about row-major (C-order) and column-major (Fortran order) layouts).
(run sample prgm accessing everyelement in a tensor and accumulating ,by iterating last dimension most(following C-order style),and also iterating first dimension most(following Fortran order style) to see the difference in time taken) .


5.Ravel_Index : (Ravelling N-D co-ordinates to a single linear index)
input : 1.coords vector(co-ordinates of an element in a tensor),2.shape vector(shape of a tensor)
output : linear_index --->int64_t
Inverse operation of Unravel_Index function :

-what it does  : Converts a multi-dimensional co-ordinate vector (ex : {3,1,2}) to a single linear index (ex : 3*5*6 + 1*6 + 2 =98 for a {4,5,6} tensor) .

Why we use this  :
Memory Access : Everytime you access Tensor[i][j][k] ,the libray must internally convert {i,j,k} to a linear offset to etrieve the data pointer from the 1D memory array .

Logic and How :
1.Initialization : Initializes linear_index to 0 .
2.uses Linear index = summation ((coord[i]*stride[i])) for i=0 to n-1
where stride[i] = product of all dimensions after i (i.e. shape[i+1]*shape[i+2]*...*shape[n-1]) . 



How reduction_kernel works :

Logic :  " For this specific cell in the output ,which values from the input mirror its co-ordinates ? "
Analogy : The reduction_kernel is like a person standing in the Output Tensor trying to find their family members in the Input Tensor.


Algorithm : (Quick-glimpse)
step1 : Preparatory math (reduced_dims,output_shape,reduced_count)
step2 : Outer loop (Iterating through output tensor,no.of iterations = no.of elements in output tensor)
for each iteration in outer loop /for each i :
      step3 : outer_unravel(outer_loop_index,output_shape) --->gives the co-ordinates of the current output element in the output tensor
      step4 : Inner loop (Iterating through reduced dimensions,no.of iterations = no.of reduced dimensions)
      for each iteration in inner loop /for each j :
            step5 : inner_unravel(inner_loop_index,reduced_dims_shape) --->gives the co-ordinates of the current output element in the reduced dimension
            step6 : Merge (Merging the reduced dimensions with the output dimensions/( merge output of outer_unravel + inner_unravel))
            step7 : Ravel(output_coords_from_merging(gives the coords in input tensor),input_tensor_strides) (Ravelling the merged dimensions to get the linear index)
step8 : Accumulate (Accumulating the values in the input tensor)

Algorithm in my simple  words :

1.Preparatory math :find input shape,input_strides,reduction_axis,reduced_dims_shape,inner_loop_count/reduced_count(product of all elements in reduced_dims_shape),output_shape,outer_loop_count .

2.Outer loop (no.of iterations = no.of elements in output tensor)

     |_______for each iteration of outer loop (for every element in output tensor ,find all contributing     elements in input tensor and accumulate those values to get the value of the current output element).
                        |_____1.outer_unravel(outer_loop_index,output_shape) --->gives the co-ordinates of the current output element in the output tensor
                        |_____2.inner_loop (no.of iterations = reduced_count)
                                     |_____for each iteration of inner loop (for every reduced dimension ,find the co-ordinates of the current output element in the reduced dimension)
                                                  |_____inner_unravel(inner_loop_index,reduced_dims_shape) --->gives the co-ordinates of the current output element in the reduced dimension
                                                  |_____merge(outer_unravel_output,inner_unravel_output) --->gives the co-ordinates of the current output element in the input tensor
                                                  |_____ravel(merged_output,input_tensor_strides) --->gives the linear index of the current output element in the input tensor
                                                  |_____accumulate(input_tensor[linear_index]) --->accumulates the values in the input tensor.
                                                  





lets take an example of input tensor of shape  3*3*3 ,with numbers from 1 to 27 placed in those 27 blocks with index stating from 0 to 26 :

Case : 1  (Partial -reduction over Dimension : 0, i.e., over planes axis if we consider (planes,rows,columns ) )


 we must trace the state of the variables output_index, out_coords, red_coords, and the final input_offset for every single step.

Numerical Constants
Input Shape: [3, 3, 3] (Rank 3)
Input Strides: [9, 3, 1]
Reduction Axis: [0] (This corresponds to the first dimension)
Reduced Dims Shape: [3] (The size of the axis we are removing)
Output Shape: [3, 3] (Rank 2)
Output Elements: 9 total
Inner Loop Count: 3 (Each output cell is a sum of 3 input values)

The Navigation Trace:
Outer Loop : (9 iterations for each index value in output_tensor,3*3=9)
The kernel starts. It enters the Outer Loop (Iterating through the Output Tensor (3*3))

Output Index 0
Stage 2 (Outer Unravel): unravel(0, [3, 3])---> {0, 0} (Row 0, Col 0 of Output).

Stage 3 (Inner Loop):
Iteration i=0: 
unravel(0, [3])---> {0}.
Stage 4 (Merge): Input Dim 0 is reduced (use {0}), Dims 1 & 2 are kept (use {0, 0}). Result: {0, 0, 0}.
Stage 5 (Ravel): 
(0*9) + (0*3) + (0*1)= Offset 0.

Iteration i=1: 
unravel(1, [3])---> {1}.
Stage 4 (Merge): Input Coord: {1, 0, 0}.
Stage 5 (Ravel): 
(1*9) + (0*3) + (0*1)= Offset 9.

Iteration i=2: 
unravel(2, [3])---> {2}.
Stage 4 (Merge): Input Coord: {2, 0, 0}.
Stage 5 (Ravel): (2*9) + (0*3) + (0*1)= Offset 18.

Result(Accumulator): Output[0] = Input[0] + Input[9] + Input[18].



Output Index 1
Stage 2 (Outer Unravel): unravel(1, [3, 3])--->{0, 1}.

Stage 3 (Inner Loop):
Iteration i=0: 
unravel(0, [3])---> {0}. Input Coord: {0, 0, 1}.
 Ravel: (0*9)+(0*3)+(1*1)= Offset 1.

Iteration i=1: 
unravel(1, [3])---> {1}. Input Coord: {1, 0, 1}. 
Ravel: 
(1*9)+(0*3)+(1*1)= Offset 10.

Iteration i=2: 
unravel(2, [3])---> {2}. Input Coord: {2, 0, 1}. 
Ravel: (2*9)+(0*3)+(1*1)= Offset 19.
Result: Output[1] = Input[1] + Input[10] + Input[19].



Output Index 3

Stage 2 (Outer Unravel): unravel(3, [3, 3])--->{1, 0}.

Stage 3 (Inner Loop):
Iteration i=0: Input Coord: {0, 1, 0}. 
Ravel: (0*9)+(1*3)+(0*1)= Offset 3.

Iteration i=1: Input Coord: {1, 1, 0}. 
Ravel: (1*9)+(1*3)+(0*1)= Offset 12.

Iteration i=2: Input Coord: {2, 1, 0}.
Ravel: (2*9)+(1*3)+(0*1)= Offset 21.



Output Index 4

Stage 2 (Outer Unravel): unravel(4, [3, 3])---> {1, 1}.

Stage 3 (Inner Loop):
Iteration i=0: Input Coord: {0, 1, 1}. Offset: 4.
Iteration i=1: Input Coord: {1, 1, 1}. Offset: 13.
Iteration i=2: Input Coord: {2, 1, 1}. Offset: 22.

Output Index 5

Stage 2 (Outer Unravel): unravel(5, [3, 3])---> {1, 2}.

Stage 3 (Inner Loop):
Iteration i=0: Input Coord: {0, 1, 2}. Offset: 5.
Iteration i=1: Input Coord: {1, 1, 2}. Offset: 14.
Iteration i=2: Input Coord: {2, 1, 2}. Offset: 23.


Output Index 6
Stage 2 (Outer Unravel): unravel(6, [3, 3])---> {2, 0}.

Stage 3 (Inner Loop):

Iteration i=0: Input Coord: {0, 2, 0}. 
Ravel: (0*9)+(2*3)+(0*1)= Offset 6.
Iteration i=1: Input Coord: {1, 2, 0}.
 Ravel:(1*9)+(2*3)+(0*1)= Offset 15.
Iteration i=2: Input Coord: {2, 2, 0}. 
Ravel: (2*9)+(2*3)+(0*1)= Offset 24.

Output Index 7
Stage 2 (Outer Unravel): unravel(7, [3, 3])---> {2, 1}.
Stage 3 (Inner Loop):
Iteration i=0: Input Coord: {0, 2, 1}. Offset: 7.
Iteration i=1: Input Coord: {1, 2, 1}. Offset: 16.
Iteration i=2: Input Coord: {2, 2, 1}. Offset: 25.

Output Index 8
Stage 2 (Outer Unravel): unravel(8, [3, 3])---> {2, 2}.
Stage 3 (Inner Loop):
Iteration i=0: Input Coord: {0, 2, 2}. Offset: 8.
Iteration i=1: Input Coord: {1, 2, 2}. Offset: 17.
Iteration i=2: Input Coord: {2, 2, 2}. Offset: 26.


Case 2: Partial Reduction over Axes [0, 1] on the same 3x3x3 tensor.

Numerical Constants
Input Shape: [3, 3, 3]
Input Strides: [9, 3, 1]
Reduction Axes: [0, 1] (Collapsing Planes and Rows)
Reduced Dims Shape: [3, 3] (The grid we are collapsing)
Output Shape: [3] (Only the Column dimension remains)
Output Elements: 3 total
Inner Loop Count (reduced_count): 3*3 = 9

The Navigation Trace
Output Index 0
Stage 2 (Outer Unravel): unravel(0, [3])--->{0} (This tells the kernel: "We are currently calculating the result for Column 0 of the input.")

Stage 3 (Inner Loop): We loop i=0 to 8.
i=0: unravel(0, [3, 3])---> {0, 0}.
Merge: Use {0, 0} for Dims 0,1; use {0} for Dim 2. Result: {0, 0, 0}.
Ravel: (0*9)+(0*3)+(0*1)= Offset 0.

i=1: unravel(1, [3, 3])---> {0, 1}.
Merge: Result: {0, 1, 0}.
Ravel: (0*9)+(1*3)+(0*1)= Offset 3.

i=2: unravel(2, [3, 3])---> {0, 2}.
Merge: Result: {0, 2, 0}.
Ravel: (0*9)+(2*3)+(0*1)= Offset 6.

i=3: unravel(3, [3, 3])---> {1, 0}.
Merge: Result: {1, 0, 0}.
Ravel: (1*9)+(0*3)+(0*1)= Offset 9.

i=4: unravel(4, [3, 3])---> {1, 1}.
Merge: Result: {1, 1, 0}.
Ravel: (1*9)+(1*3)+(0*1)= Offset 12.

i=5: unravel(5, [3, 3])---> {1, 2}.
Merge: Result: {1, 2, 0}.
Ravel: (1*9)+(2*3)+(0*1)= Offset 15.

i=6: unravel(6, [3, 3])---> {2, 0}.
Merge: Result: {2, 0, 0}.
Ravel: (2*9)+(0*3)+(0*1)= Offset 18.

i=7: unravel(7, [3, 3])---> {2, 1}.
Merge: Result: {2, 1, 0}.
Ravel: (2*9)+(1*3)+(0*1)= Offset 21.

i=8: unravel(8, [3, 3])---> {2, 2}.
Merge: Result: {2, 2, 0}.
Ravel: (2*9)+(2*3)+(0*1)= Offset 24.
Result: Output[0] = Sum of Inputs at indices (0, 3, 6, 9, 12, 15, 18, 21, 24).



Output Index 1
Stage 2 (Outer Unravel): unravel(1, [3])--->{1} (Calculating for Column 1)
Stage 3 (Inner Loop):
i=0: Merge {0, 0} with {1} ---> {0, 0, 1}. Ravel: Offset 1.
i=1: Merge {0, 1} with {1} ---> {0, 1, 1}. Ravel: Offset 4.
i=2: Merge {0, 2} with {1} ---> {0, 2, 1}. Ravel: Offset 7.
i=3: Merge {1, 0} with {1} ---> {1, 0, 1}. Ravel: Offset 10.
i=4: Merge {1, 1} with {1} ---> {1, 1, 1}. Ravel: Offset 13.
i=5: Merge {1, 2} with {1} ---> {1, 2, 1}. Ravel: Offset 16.
i=6: Merge {2, 0} with {1} ---> {2, 0, 1}. Ravel: Offset 19.
i=7: Merge {2, 1} with {1} ---> {2, 1, 1}. Ravel: Offset 22.
i=8: Merge {2, 2} with {1} ---> {2, 2, 1}. Ravel: Offset 25.
Result: Output[1] = Sum of Inputs at indices (1, 4, 7, 10, 13, 16, 19, 22, 25).

Output Index 2
Stage 2 (Outer Unravel): unravel(2, [3]) ---> {2} (Calculating for Column 2)
Stage 3 (Inner Loop):
Following the exact same logic, it will collect measurements at Offsets: (2, 5, 8, 11, 14, 17, 20, 23, 26).



Case 3: Full Reduction (Axes [0, 1, 2]) on the 3x3x3 tensor.

Numerical Constants
Input Shape: [3, 3, 3]
Input Strides: [9, 3, 1]
Reduction Axes: [0, 1, 2] (Everything is flattened)
Reduced Dims Shape: [3, 3, 3] (The entire volume is collapsed)
Output Shape: [1] (A single scalar value)
Output Elements: 1
Inner Loop Count (reduced_count): 3*3*3 = 27

The Navigation Trace:

Output Index 0

Stage 2 (Outer Unravel): unravel(0, [1])---> { } (An empty coordinate set).

Logic: Since there are no dimensions kept, there are no coordinates to track in the output.
 The person is standing at the only available spot: Point 0.

Stage 3 (Inner Loop): We loop i=0 to 26.

i=0: 
unravel(0, [3, 3, 3])--->{0, 0, 0}.
Merge: Since 100% of dimensions are reduced, we only use the inner unravel output. 
Result: {0, 0, 0}.
Ravel: (0*9)+(0*3)+(0*1) = Offset 0.

i=1: 
unravel(1, [3, 3, 3])--->{0, 0, 1}.
Merge/Ravel: Offset 1.
........
i=9: 
unravel(9, [3, 3, 3])--->{1, 0, 0}.
Merge/Ravel: Offset 9.
........
i=26: 
unravel(26, [3, 3, 3])--->{2, 2, 2}.
Merge/Ravel: Offset 26.
Result: Output[0] = Sum of every single element in the input (Offsets 0 through 26).



Pending steps :
*Optimization on this Full-reduction case :
Because the math results in a 1-to-1 mapping (i == offset), the library often detects this "Full Reduction" case and skips the ravel/unravel steps entirely. It just does a simple C-style loop:
for (int i=0; i < N; ++i) {
    accumulator += input[i];
}

*Bug in index-returning ops,incase of partial-reduction over multiple-axes .



CPU parallelism :
Outer loop is parallelized by OpenMP,one CPU thread takes one output_index and does the entire job for that cell .
Inner Loop ---> A simple serial for loop inside each CPU thread .
Accumulation ---> A single variable accumulator per thread (accumulator+= val) . 


GPU Parallelism :

Outer loop : Handled by CUDA Blocks,one block of threads is assigned to one output_index .
Inner loop : Massively parallel,every thread in the block takes a portion of the big_chunk(reduced_count) .
Accumulation : Warp shuffles and shared Memory--->threads use high-speed on-chip memory to sum their values together in logarithmic time (log N).

The execution configuration is set up in 
ReductionImplGPU.cu
- inside the dispatcher functions (dispatch_reduction_gpu).
1. The Configuration Logic (Line 103-148)

ReductionImplGPU.cu file :

// --- Kernel configuration ---
int threads_per_block = 256;      // Every Output cell gets a team of 256 threads
int num_blocks = num_slices;       // num_slices = total number of elements in the OUTPUT tensor
// --- Launch ---
cuda::reduce_kernel<CudaT, OutputCudaT, OpType>
    <<<num_blocks, threads_per_block, shared_mem_size, stream>>>(
        // ... pointers ...
        num_slices,
        reduced_count,
        // ... metadata ...
    );

Grid Size (num_blocks): This is equal to the number of cells in the Output Tensor.
If you are reducing a [100, 200] tensor along axis 1 to get a [100] output, you launch 100 blocks.

Mapping: 1 Block = 1 Output Element.

Block Size (threads_per_block): Fixed at 256 threads.
This group of 256 threads works together to tackle the reduced_count(the "Tower" of values to be summed).

Shared Memory: Notice the third parameter shared_mem_size. This is dynamically allocated on the GPU chip so threads can share their partial sums instantly.

How the Kernel uses this Grid (The Loop) :

Inside the 

ReductionKernels.cuh file, the kernel starts by identifying which "Job" it belongs to:
// Identification
for (int64_t output_index = blockIdx.x; output_index < num_slices; output_index += gridDim.x) {
    
    // The Team effort for the Inner Loop
    for (int64_t i = threadIdx.x; i < reduced_count; i += blockDim.x) {
        // ... Math ...
    }
}

blockIdx.x (The Job): Each block reads its ID and says "I am responsible for Output Index X."

threadIdx.x (The Sub-Task): Inside the block, threads use their local IDs (0 to 255) to decide which input values to grab.
Thread 0 grabs element 0, 256, 512...
Thread 1 grabs element 1, 257, 513...

Grid-Stride Loop: Even if you have 1 million output slices but only 65,535 blocks (the hardware limit), the output_index += gridDim.x logic ensures every slice gets processed eventually.