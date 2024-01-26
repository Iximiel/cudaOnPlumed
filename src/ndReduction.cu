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
#include "ndReduction.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// needed for plumed_merror
#include "plumed/tools/Exception.h"

#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

// #define vdbg(...) std::cerr << std::setw(4) << __LINE__ <<":" <<
// std::setw(20)<< #__VA_ARGS__ << " " << (__VA_ARGS__) <<'\n'
#define vdbg(...)

// Some help for me:
// * Grids map to GPUs
// * Blocks map to the MultiProcessors (MP)
// * Threads map to Stream Processors (SP)
// * Warps are groups of (32) threads that execute simultaneously

// There are a LOTS of unrolled loop down here,
// see this
// https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
//  to undestand why

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#dim3
// 7.3.2. dim3
// This type is an integer vector type based on uint3 that is used to specify
// dimensions.

namespace CUDAHELPERS {
template <unsigned numThreads, typename T>
__global__ void reductionND(const T *inputArray, T *outputArray,
                            const unsigned int len) {
  // playing with this
  // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  auto sdata = shared_memory_proxy<T>();
  // const unsigned int coord = blockIdx.y;
  const unsigned int place = threadIdx.x;
  // each thread loads one element from global to shared memory
  const unsigned int diplacement = blockIdx.y * len;
  unsigned int i = (numThreads * 2) * blockIdx.x + place + diplacement;
  const unsigned int gridSize = (numThreads * 2) * gridDim.x;
  // the first element is in blockIdx.y*len, the last element to sum in
  // (blockIdx.y+1)*len-1
  const unsigned int trgt = len + diplacement;

  sdata[threadIdx.x] = T(0);
  while (i + numThreads < trgt) {
    sdata[threadIdx.x] += inputArray[i] + inputArray[i + numThreads];
    i += gridSize;
  }
  // The double while is for preventig to lose the last data (the
  // https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
  // implementation sums over 2^n integers, here we are not using a number of
  // data that is power of 2)
  while (i < trgt) {
    sdata[threadIdx.x] += inputArray[i];
    i += gridSize;
  }

  __syncthreads();
  // do reduction in shared memory
  reductor<numThreads>(sdata, outputArray, blockIdx.x + blockIdx.y * gridDim.x);
}

template <unsigned numThreads, typename T>
__global__ void reduction1D(T *inputArray, T *outputArray,
                            const unsigned int len) {
  auto sdata = shared_memory_proxy<T>();
  const unsigned int place = threadIdx.x;
  // each thread sums some elements from global to shared memory
  unsigned int i = (2 * numThreads) * blockIdx.x + place;
  const unsigned int gridSize = (2 * numThreads) * gridDim.x;
  sdata[place] = T(0);
  while (i + numThreads < len) {
    sdata[place] += inputArray[i] + inputArray[i + numThreads];
    i += gridSize;
  }
  while (i < len) {
    sdata[place] += inputArray[i];
    i += gridSize;
  }

  __syncthreads();
  // do reduction in shared memory
  reductor<numThreads>(sdata, outputArray, blockIdx.x);
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

template <typename T>
void callReduction1D(T *inputArray, T *outputArray, const unsigned int len,
                     const unsigned blocks, const unsigned nthreads) {
  switch (nthreads) {
  case 1024:
    reduction1D<1024, T>
        <<<blocks, 1024, 1024 * sizeof(T)>>>(inputArray, outputArray, len);
  case 512:
    reduction1D<512, T>
        <<<blocks, 512, 512 * sizeof(T)>>>(inputArray, outputArray, len);
    break;
  case 256:
    reduction1D<256, T>
        <<<blocks, 256, 256 * sizeof(T)>>>(inputArray, outputArray, len);
    break;
  case 128:
    reduction1D<128, T>
        <<<blocks, 128, 128 * sizeof(T)>>>(inputArray, outputArray, len);
    break;
  case 64:
    reduction1D<64, T>
        <<<blocks, 64, 64 * sizeof(T)>>>(inputArray, outputArray, len);
    break;
  case 32:
    reduction1D<32, T>
        <<<blocks, 32, 32 * sizeof(T)>>>(inputArray, outputArray, len);
    break;
  default:
    plumed_merror(
        "Reduction can be called only with 512, 256, 128, 64 or 32 threads.");
  }
}

template <typename T>
void callReductionND(T *inputArray, T *outputArray, const unsigned int len,
                     const dim3 blocks, const unsigned nthreads) {
  switch (nthreads) {
  case 1024:
    reductionND<1024, T>
        <<<blocks, 1024, 1024 * sizeof(T)>>>(inputArray, outputArray, len);
  case 512:
    reductionND<512, T>
        <<<blocks, 512, 512 * sizeof(T)>>>(inputArray, outputArray, len);
    break;
  case 256:
    reductionND<256, T>
        <<<blocks, 256, 256 * sizeof(T)>>>(inputArray, outputArray, len);
    break;
  case 128:
    reductionND<128, T>
        <<<blocks, 128, 128 * sizeof(T)>>>(inputArray, outputArray, len);
    break;
  case 64:
    reductionND<64, T>
        <<<blocks, 64, 64 * sizeof(T)>>>(inputArray, outputArray, len);
    break;
  case 32:
    reductionND<32, T>
        <<<blocks, 32, 32 * sizeof(T)>>>(inputArray, outputArray, len);
    break;
  default:
    plumed_merror(
        "Reduction can be called only with 512, 256, 128, 64 or 32 threads.");
  }
}

template <typename T>
void callReduction1D(T *inputArray, T *outputArray, const unsigned int len,
                     const unsigned blocks, const unsigned nthreads,
                     cudaStream_t stream = 0) {
  switch (nthreads) {
  case 1024:
    reduction1D<1024, T>
        <<<blocks, 1024, 1024 * sizeof(T)>>>(inputArray, outputArray, len);
  case 512:
    reduction1D<512, T><<<blocks, 512, 512 * sizeof(T), stream>>>(
        inputArray, outputArray, len);
    break;
  case 256:
    reduction1D<256, T><<<blocks, 256, 256 * sizeof(T), stream>>>(
        inputArray, outputArray, len);
    break;
  case 128:
    reduction1D<128, T><<<blocks, 128, 128 * sizeof(T), stream>>>(
        inputArray, outputArray, len);
    break;
  case 64:
    reduction1D<64, T>
        <<<blocks, 64, 64 * sizeof(T), stream>>>(inputArray, outputArray, len);
    break;
  case 32:
    reduction1D<32, T>
        <<<blocks, 32, 32 * sizeof(T), stream>>>(inputArray, outputArray, len);
    break;
  default:
    plumed_merror(
        "Reduction can be called only with 512, 256, 128, 64 or 32 threads.");
  }
}

template <typename T>
void callReductionND(T *inputArray, T *outputArray, const unsigned int len,
                     const dim3 blocks, const unsigned nthreads,
                     cudaStream_t stream = 0) {
  switch (nthreads) {
  case 1024:
    reductionND<1024, T>
        <<<blocks, 1024, 1024 * sizeof(T)>>>(inputArray, outputArray, len);
  case 512:
    reductionND<512, T><<<blocks, 512, 512 * sizeof(T), stream>>>(
        inputArray, outputArray, len);
    break;
  case 256:
    reductionND<256, T><<<blocks, 256, 256 * sizeof(T), stream>>>(
        inputArray, outputArray, len);
    break;
  case 128:
    reductionND<128, T><<<blocks, 128, 128 * sizeof(T), stream>>>(
        inputArray, outputArray, len);
    break;
  case 64:
    reductionND<64, T>
        <<<blocks, 64, 64 * sizeof(T), stream>>>(inputArray, outputArray, len);
    break;
  case 32:
    reductionND<32, T>
        <<<blocks, 32, 32 * sizeof(T), stream>>>(inputArray, outputArray, len);
    break;
  default:
    plumed_merror(
        "Reduction can be called only with 512, 256, 128, 64 or 32 threads.");
  }
}

void doReduction1D(double *inputArray, double *outputArray,
                   const unsigned int len, const unsigned blocks,
                   const unsigned nthreads, cudaStream_t stream) {
  callReduction1D(inputArray, outputArray, len, blocks, nthreads, stream);
}

void doReductionND(double *inputArray, double *outputArray,
                   const unsigned int len, const dim3 blocks,
                   const unsigned nthreads, cudaStream_t stream) {
  callReductionND(inputArray, outputArray, len, blocks, nthreads, stream);
}

void doReduction1D(float *inputArray, float *outputArray,
                   const unsigned int len, const unsigned blocks,
                   const unsigned nthreads, cudaStream_t stream) {
  callReduction1D(inputArray, outputArray, len, blocks, nthreads, stream);
}

void doReductionND(float *inputArray, float *outputArray,
                   const unsigned int len, const dim3 blocks,
                   const unsigned nthreads, cudaStream_t stream) {
  callReductionND(inputArray, outputArray, len, blocks, nthreads, stream);
}
} // namespace CUDAHELPERS
