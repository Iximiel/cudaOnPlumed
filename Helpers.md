### cudaHelpers.cuh
cudaHelpers.cuh is an very simple template-only library made ad hoc for this project. It contains a simple interface for working with the memory in cuda, and the `__device__` function for completing a binary reduction.

The memory interface is a templated class: `template <typename T> class memoryHolder;` will ease the memory management with an RAII approach (so by hiding the need to call `cudaMalloc` and `cudaFree`).
It `memoryHolder` is set up as a move-only object in a `std::unique_ptr` style. It initializes the wanted quantity of memory on construction. And has the following methods:
 - `T* pointer()`{:.c++} returns the GPU address of the allocated memory in the GPU
 - `size_t size()` returns the size of the memory allocated on the gpu that is accessible with the `copyTo` and `copyFrom` methods (in numbers of `T`, like in the STL)
 - `size_t reserved()` returns the size of the memory allocated on the gpu (in numbers of `T`, like in the STL), it may be bigger than the number returned by size
 - `void resize(size_t)` changes the size of the memory accessible, if the asked size is greater than the reserved memory, the memory will be reallocated, otherwise will only be changed the avaiable memory to the `copyTo` and `copyFrom` methods. A parameter regulates if the new array should contain the old data during a reallocation
 - `void reserve(size_t)` reallocate the memory on the GPU (after calling this method `size==reserved`). A parameter regulates if the new array should contain the old data during a reallocation
 - `void copyToCuda(Y *)` copies a number of `T` elements equal to `size` from the CPU to the GPU memory (if called with a type different from `T` a conversion will be made on the CPU before copying the memory)
 - `void copyFromCuda(Y *)` copies a number of `T` elements equal to `size` from the GPU to the CPU memory (if called with a type different from `T` a conversion will be made on the CPU after copying the memory)

 both `copyTo` and `copyFrom` have an overloaded async version that is invoked by calling the function with specifying a stream as second argument  (`void copyToCuda(Y *,cudaStream_t )`, `void copyFromCuda(Y *, cudaStream_t )`).