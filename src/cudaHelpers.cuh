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

#include "plumed/tools/Tensor.h"
#include "plumed/tools/Vector.h"
#include <thrust/device_vector.h>
#include <vector>
namespace CUDAHELPERS {

// the two dim classes are placeholders
template <unsigned n> constexpr unsigned dim(PLMD::VectorGeneric<n>) {
  return n;
}

template <unsigned n, unsigned m>
constexpr unsigned dim(PLMD::TensorGeneric<n, m>) {
  return n * m;
}

template <typename T>
inline double *cast_to_simple_pointer(std::vector<T> &data) {
  return static_cast<double *>(&data[0][0]);
}

template <typename T>
inline void plmdDataToGPU(thrust::device_vector<double> &dvmem,
                          std::vector<T> &data) {
  const auto usedim_ = dim(data[0]) * data.size();
  dvmem.resize(usedim_);
  cudaMemcpy(thrust::raw_pointer_cast(dvmem.data()), &data[0][0],
             usedim_ * sizeof(double), cudaMemcpyHostToDevice);
}

template <typename T>
inline void plmdDataToGPU(thrust::device_vector<float> &dvmem,
                          std::vector<T> &data) {
  const auto usedim_ = dim(data[0]) * data.size();
  dvmem.resize(usedim_);
  std::vector<float> tempMemory(3 * data.size());
  for (auto i = 0u; i < usedim_; ++i) {
    tempMemory[i] = cast_to_simple_pointer(data)[i];
  }
  cudaMemcpy(thrust::raw_pointer_cast(dvmem.data()), tempMemory.data(),
             usedim_ * sizeof(float), cudaMemcpyHostToDevice);
}

template <typename T>
inline void plmdDataFromGPU(thrust::device_vector<float> &dvmem,
                            std::vector<T> &data) {
  const auto usedim_ = dim(data[0]) * data.size();
  std::vector<float> tempMemory(usedim_);
  cudaMemcpy(tempMemory.data(), thrust::raw_pointer_cast(dvmem.data()),
             usedim_ * sizeof(float), cudaMemcpyDeviceToHost);
  //  cudaMemcpyAsync
  for (auto i = 0u; i < usedim_; ++i) {
    cast_to_simple_pointer(data)[i] = tempMemory[i];
  }
}

template <typename T>
inline void plmdDataFromGPU(thrust::device_vector<double> &dvmem,
                            std::vector<T> &data) {
  const auto usedim_ = dim(data[0]) * data.size();
  cudaMemcpy(&data[0][0], thrust::raw_pointer_cast(dvmem.data()),
             usedim_ * sizeof(double), cudaMemcpyDeviceToHost);
  //  cudaMemcpyAsync
}

template <typename T>
inline void plmdDataFromGPU(thrust::device_vector<float> &dvmem, T &data) {
  auto usedim_ = dim(data);
  std::vector<float> tempMemory(usedim_);
  cudaMemcpy(tempMemory.data(), thrust::raw_pointer_cast(dvmem.data()),
             usedim_ * sizeof(float), cudaMemcpyDeviceToHost);
  //  cudaMemcpyAsync
  for (auto i = 0u; i < usedim_; ++i) {
    *(static_cast<double *>(&data[0][0]) + i) = tempMemory[i];
  }
}

template <typename T>
inline void plmdDataFromGPU(thrust::device_vector<double> &dvmem, T &data) {
  auto usedim_ = dim(data);
  cudaMemcpy(&data[0][0], thrust::raw_pointer_cast(dvmem.data()),
             usedim_ * sizeof(double), cudaMemcpyDeviceToHost);
  //  cudaMemcpyAsync
}

// after c++14 the template activation will be shorter to write:
// template<typename T, std::enable_if_t<std::is_integral_v<T>, bool> = true>

/// finds the nearest upper multiple of the given reference
template <typename T, typename std::enable_if<std::is_integral<T>::value,
                                              bool>::type = true>
inline T nearestUpperMultipleTo(T number, T reference) {
  return ((number - 1) | (reference - 1)) + 1;
}

/// We'll find the ideal number of blocks using the Brent's theorem
size_t idealGroups(size_t numberOfElements, size_t runningThreads) {
  // nearest upper multiple to the numberof threads
  const size_t nnToGPU =
      nearestUpperMultipleTo(numberOfElements, runningThreads);
  /// Brent’s theorem says each thread should sum O(log n) elements
  // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  // const size_t elementsPerThread=log2(runningThreads);
  const size_t expectedTotalThreads = ceil(nnToGPU / log2(runningThreads));
  // hence the blocks should have this size:
  const unsigned ngroups =
      nearestUpperMultipleTo(expectedTotalThreads, runningThreads) /
      runningThreads;
  return ngroups;
}

size_t threadsPerBlock(unsigned N, unsigned maxNumThreads) {
  // this seeks the minimum number of threads to use a sigle block (and end the
  // recursion)
  size_t dim = 32;
  for (dim = 32; dim < 1024; dim <<= 1) {
    if (maxNumThreads < dim) {
      dim >>= 1;
      break;
    }
    if (N < dim) {
      break;
    }
  }
  return dim;
}

} // namespace CUDAHELPERS
#endif //__PLUMED_cuda_helpers_cuh
