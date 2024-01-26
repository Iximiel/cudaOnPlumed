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

/// We'll find the ideal number of blocks using the Brent's theorem
size_t idealGroups(const size_t numberOfElements, const size_t runningThreads) {
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

size_t threadsPerBlock(const unsigned N, const unsigned maxNumThreads) {
  // this seeks the minimum number of threads to use a sigle block (and end
  // the recursion)
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
