/* +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   Copyright (c) 2023 Daniele Rapetti

   This file is part of cudaOnPlumed.

   cudaOnPlumed is free software: you can redistribute it and/or modify
   it under the terms of the GNU Lesser General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   cudaOnPlumed is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public License
   along with cudaOnPlumed.  If not, see <http://www.gnu.org/licenses/>.
+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ */
#ifndef __PLUMED_cuda_helpers_cuh
#define __PLUMED_cuda_helpers_cuh

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <vector>

namespace CUDAHELPERS {
/// @brief a interface to help in the data I/O to the GPU
struct DataInterface {
  double *ptr = nullptr;
  size_t size = 0;
  DataInterface() = delete;
  template <unsigned n>
  explicit DataInterface(PLMD::VectorGeneric<n> &vg) : ptr(&vg[0]), size(n) {}
  template <unsigned n>
  explicit DataInterface(PLMD::VectorGeneric<n> &vg, size_t s)
      : ptr(&vg[0]), size(s * n) {}

  template <unsigned n, unsigned m>
  explicit DataInterface(PLMD::TensorGeneric<n, m> &tns)
      : ptr(&tns[0][0]), size(n * m) {}
  template <unsigned n, unsigned m>
  explicit DataInterface(PLMD::TensorGeneric<n, m> &tns, size_t s)
      : ptr(&tns[0][0]), size(s * n * m) {}
  template <typename T>
  explicit DataInterface(std::vector<T> &vt)
      : DataInterface(vt[0], vt.size()) {}
};

inline void plmdDataFromGPU(thrust::device_vector<double> &dvmem,
                            DataInterface data) {
  cudaMemcpy(data.ptr, thrust::raw_pointer_cast(dvmem.data()),
             data.size * sizeof(double), cudaMemcpyDeviceToHost);
  //  cudaMemcpyAsync
}

inline void plmdDataFromGPU(thrust::device_vector<float> &dvmem,
                            DataInterface data) {

  std::vector<float> tempMemory(data.size);
  cudaMemcpy(tempMemory.data(), thrust::raw_pointer_cast(dvmem.data()),
             data.size * sizeof(float), cudaMemcpyDeviceToHost);
  //  cudaMemcpyAsync
  for (auto i = 0u; i < data.size; ++i) {
    data.ptr[i] = tempMemory[i];
  }
}

inline void plmdDataToGPU(thrust::device_vector<double> &dvmem,
                          DataInterface data) {
  dvmem.resize(data.size);
  cudaMemcpy(thrust::raw_pointer_cast(dvmem.data()), data.ptr,
             data.size * sizeof(double), cudaMemcpyHostToDevice);
}

inline void plmdDataToGPU(thrust::device_vector<float> &dvmem,
                          DataInterface data) {
  dvmem.resize(data.size);
  std::vector<float> tempMemory(data.size);
  for (auto i = 0u; i < data.size; ++i) {
    tempMemory[i] = data.ptr[i];
  }
  cudaMemcpy(thrust::raw_pointer_cast(dvmem.data()), tempMemory.data(),
             data.size * sizeof(float), cudaMemcpyHostToDevice);
}

// the explicit constructors of DataInterface create the need for a wrapper
template <typename T, typename Y>
inline void plmdDataToGPU(thrust::device_vector<T> &dvmem, Y &data) {
  plmdDataToGPU(dvmem, DataInterface(data));
}

// the explicit constructors of DataInterface create the need for a wrapper
template <typename T, typename Y>
inline void plmdDataFromGPU(thrust::device_vector<T> &dvmem, Y &data) {
  plmdDataFromGPU(dvmem, DataInterface(data));
}

// after c++14 the template activation will be shorter to write:
// template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>

/// finds the nearest upper multiple of the given reference
template <typename T, typename std::enable_if<std::is_integral<T>::value,
                                              bool>::type = true>
inline T nearestUpperMultipleTo(T number, T reference) {
  return ((number - 1) | (reference - 1)) + 1;
}

template <class T> __device__ constexpr const T &mymin(const T &a, const T &b) {
  return (b < a) ? b : a;
}

// this make possible to use shared memorywithin a templated kernel
template <typename T> __device__ T *shared_memory_proxy() {
  // do we need an __align__() here?
  extern __shared__ unsigned char memory[];
  return reinterpret_cast<T *>(memory);
}

/// @brief This is the "body" of th ereduction algorithm: you should prepare the
/// shared memory and the plug it in this, due to the templated numthreads most
/// of the if are done at compile time
template <unsigned numThreads, typename T>
__device__ void reductor(volatile T *sdata, T *outputArray,
                         const unsigned int where) {
  // this is an unrolled loop
  const unsigned int tid = threadIdx.x;
  if (numThreads >= 1024) { // compile time
    if (tid < 512)
      sdata[tid] += sdata[tid + 512];
    __syncthreads();
  }
  if (numThreads >= 512) { // compile time
    if (tid < 256)
      sdata[tid] += sdata[tid + 256];
    __syncthreads();
  }
  if (numThreads >= 256) { // compile time
    if (tid < 128)
      sdata[tid] += sdata[tid + 128];
    __syncthreads();
  }
  if (numThreads >= 128) { // compile time
    if (tid < 64)
      sdata[tid] += sdata[tid + 64];
    __syncthreads();
  }
  // a warp is composed by 32 threads that executes instructions syncrhonized,
  // so no  need to use __syncthreads() for the last iterations;
  if (tid < mymin(32u, numThreads / 2)) {
    // warpReduce<numThreads>(sdata, tid);
    if (numThreads >= 64) { // compile time
      sdata[tid] += sdata[tid + 32];
    }
    if (numThreads >= 32) { // compile time
      sdata[tid] += sdata[tid + 16];
    }
    if (numThreads >= 16) { // compile time
      sdata[tid] += sdata[tid + 8];
    }
    if (numThreads >= 8) { // compile time
      sdata[tid] += sdata[tid + 4];
    }
    if (numThreads >= 4) { // compile time
      sdata[tid] += sdata[tid + 2];
    }
    if (numThreads >= 2) { // compile time
      sdata[tid] += sdata[tid + 1];
    }
  }
  // write result for this block to global memory
  if (tid == 0) {
    outputArray[where] = sdata[0];
  }
}

} // namespace CUDAHELPERS
#endif //__PLUMED_cuda_helpers_cuh
