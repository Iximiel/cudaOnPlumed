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
#ifndef __PLUMED_cuda_ndReduction_h
#define __PLUMED_cuda_ndReduction_h
#include "cudaHelpers.cuh"
#include <vector>

namespace CUDAHELPERS {

/** @brief reduce the input 1D cudaArray
 * @param inputArray the data to be reduced, this must point to a memory
 * address into the GPU
 * @param outputArray the address for the result of the reduction, this must
 * point to a memory address into the GPU
 * @param len the lenght of the array
 * @param blocks the dimension of the group to use
 * @param nthreads the number of threads per group (must be a power of 2)
 * @param stream the optional stream for concurrency
 */
void doReduction1D(double *inputArray, double *outputArray,
                   const unsigned int len, const unsigned blocks,
                   const unsigned nthreads, cudaStream_t stream = 0);

/** @brief reduce the input ND cudaArray
 * @param inputArray the data to be reduced, this must point to a memory address
 * into the GPU
 * @param outputArray the address for the result of the reduction, this must
 * point to a memory address into the GPU
 * @param len the lenght of the array
 * @param blocks the dimension of the group to use (must be (ngroup,ndim))
 * @param nthreads the number of threads per group (must be a power of 2)
 * @param stream the optional stream for concurrency
 *
 * The componensts of inputArray in memory shoud be stored organized in N
 * sequential blocks of len, represneting each dimension like: [x0, y0, z0, x1,
 * y1, z1 ... x(N-1), y(N-1), z(N-1)]
 */
void doReductionND(double *inputArray, double *outputArray,
                   const unsigned int len, const dim3 blocks,
                   const unsigned nthreads, cudaStream_t stream = 0);

void doReduction1D(float *inputArray, float *outputArray,
                   const unsigned int len, const unsigned blocks,
                   const unsigned nthreads, cudaStream_t stream = 0);

void doReductionND(float *inputArray, float *outputArray,
                   const unsigned int len, const dim3 blocks,
                   const unsigned nthreads, cudaStream_t stream = 0);

size_t idealGroups(size_t numberOfElements, size_t runningThreads);
size_t threadsPerBlock(unsigned N, unsigned maxNumThreads = 512);
} // namespace CUDAHELPERS

#endif //__PLUMED_cuda_ndReduction_h
