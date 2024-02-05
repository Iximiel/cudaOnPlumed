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

#include "plumed/core/ActionRegister.h"
#include "plumed/core/Colvar.h"
#include "plumed/tools/NeighborList.h"
#include "plumed/tools/SwitchingFunction.h"

#include "cudaHelpers.cuh"
// #include "ndReduction.h"

#include "Coordination.cuh"

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// cfloat for DLB_EPSILON and FLT_EPSILON
#include <cfloat>

#include <iostream>
#include <limits>
#include <numeric>

using std::cerr;

// #define vdbg(...) std::cerr << __LINE__ << ":" << #__VA_ARGS__ << " " <<
// (__VA_ARGS__) << '\n'
#define vdbg(...)

namespace PLMD {
namespace colvar {
//+PLUMEDOC COLVAR CUDACOORDINATION
/*
Calculate coordination numbers. Like coordination, but on nvdia gpu and with
limited switching.

CUDACOORDINATION can be invoked with CUDACOORDINATIONFLOAT, but that version
will use single floating point precision, while being faster and compatible with
desktop-based Nvidia cards.

This keyword can be used to calculate the number of contacts between two groups
of atoms and is defined as \f[ \sum_{i\in A} \sum_{i\in B} s_{ij} \f] where
\f$s_{ij}\f$ is 1 if the contact between atoms \f$i\f$ and \f$j\f$ is formed,
zero otherwise.
In actuality, \f$s_{ij}\f$ is replaced with a switching function so as to ensure
that the calculated CV has continuous derivatives. The default switching
function is: \f[ s_{ij} = \frac{ 1 - \left(\frac{{\bf r}_{ij}}{r_0}\right)^n
} { 1 - \left(\frac{{\bf r}_{ij}}{r_0}\right)^m } \f].


\par Examples

Here's an example that shows what happens when providing COORDINATION with
a single group:
\plumedfile
# define some huge group:
group: GROUP ATOMS=1-1000
# Here's coordination within a single group:
CUDACOORDINATION GROUPA=group R_0=0.3

\endplumedfile

*/
//+ENDPLUMEDOC

// does not inherit from coordination base because nl is private
template <typename calculateFloat> class CudaCoordinationWB : public Colvar {
  /// the pointer to the coordinates on the GPU
  thrust::device_vector<calculateFloat> cudaPositions;
  /// the pointer to the nn list on the GPU
  thrust::device_vector<calculateFloat> cudaCoordination;
  thrust::device_vector<calculateFloat> cudaDerivatives;
  thrust::device_vector<calculateFloat> cudaVirial;
  thrust::device_vector<calculateFloat> reductionMemoryVirial;
  thrust::device_vector<calculateFloat> reductionMemoryCoord;
  thrust::device_vector<unsigned> cudaTrueIndexes;

  cudaStream_t streamDerivatives;
  cudaStream_t streamVirial;
  cudaStream_t streamCoordination;

  unsigned maxNumThreads = 512;
  unsigned atomsInA;
  unsigned atomsInB;
  PLMD::GPU::rationalSwitchParameters<calculateFloat> switchingParameters;
  PLMD::GPU::ortoPBCs<calculateFloat> myPBC;

  bool pbc{true};
  void setUpPermanentGPUMemory();

public:
  explicit CudaCoordinationWB (const ActionOptions &);
  virtual ~CudaCoordinationWB();
  // active methods:
  static void registerKeywords (Keywords &keys);
  void calculate() override;
};
using CudaCoordinationWB_d = CudaCoordinationWB<double>;
using CudaCoordinationWB_f = CudaCoordinationWB<float>;
PLUMED_REGISTER_ACTION (CudaCoordinationWB_d, "CUDACOORDINATIONWB")
PLUMED_REGISTER_ACTION (CudaCoordinationWB_f, "CUDACOORDINATIONWBFLOAT")

template <typename calculateFloat>
void CudaCoordinationWB<calculateFloat>::setUpPermanentGPUMemory() {
  auto nat = getPositions().size();
  cudaPositions.resize (3 * nat);
  cudaDerivatives.resize (3 * nat);
  cudaTrueIndexes.resize (nat);
  std::vector<unsigned> trueIndexes (nat);
  for (size_t i = 0; i < nat; ++i) {
    trueIndexes[i] = getAbsoluteIndex (i).index();
  }
  cudaTrueIndexes = trueIndexes;
}

template <typename calculateFloat>
void CudaCoordinationWB<calculateFloat>::registerKeywords (Keywords &keys) {
  Colvar::registerKeywords (keys);

  keys.add ("optional", "THREADS", "The upper limit of the number of threads");
  keys.add ("atoms", "GROUPA", "First list of atoms");
  keys.add ("atoms", "GROUPB", "Second list of atoms");

  keys.add (
      "compulsory", "NN", "6", "The n parameter of the switching function ");
  keys.add ("compulsory",
            "MM",
            "0",
            "The m parameter of the switching function; 0 implies 2*NN");
  keys.add ("compulsory", "R_0", "The r_0 parameter of the switching function");
  keys.add (
      "compulsory", "D_MAX", "0.0", "The cut off of the switching function");
}

template <typename calculateFloat>
CudaCoordinationWB<calculateFloat>::CudaCoordinationWB (const ActionOptions &ao)
    : PLUMED_COLVAR_INIT (ao) {
  std::vector<AtomNumber> GroupA;
  parseAtomList ("GROUPA", GroupA);
  atomsInA = GroupA.size();

  std::vector<AtomNumber> GroupB;
  parseAtomList ("GROUPB", GroupB);
  atomsInB = GroupB.size();

  // todo: improve this
  GroupA.insert (GroupA.end(), GroupB.begin(), GroupB.end());

  bool nopbc = !pbc;
  parseFlag ("NOPBC", nopbc);
  pbc = !nopbc;

  parse ("THREADS", maxNumThreads);
  if (maxNumThreads <= 0)
    error ("THREADS should be positive");
  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms (GroupA);

  log.printf ("  \n");
  if (pbc)
    log.printf ("  using periodic boundary conditions\n");
  else
    log.printf ("  without periodic boundary conditions\n");

  std::string sw, errors;

  { // loading data to the GPU
    int nn_ = 6;
    int mm_ = 0;

    calculateFloat r0_ = 0.0;
    parse ("R_0", r0_);
    if (r0_ <= 0.0) {
      error ("R_0 should be explicitly specified and positive");
    }

    parse ("NN", nn_);
    parse ("MM", mm_);
    if (mm_ == 0) {
      mm_ = 2 * nn_;
    }
    if (mm_ % 2 != 0 || mm_ % 2 != 0)
      error (" this implementation only works with both MM and NN even");

    switchingParameters.nn = nn_;
    switchingParameters.mm = mm_;
    switchingParameters.stretch = 1.0;
    switchingParameters.shift = 0.0;

    calculateFloat dmax = 0.0;
    parse ("D_MAX", dmax);
    if (dmax == 0.0) { // TODO:check for a "non present flag"
      // set dmax to where the switch is ~0.00001
      dmax = r0_ * std::pow (0.00001, 1.0 / (nn_ - mm_));
      // ^This line is equivalent to:
      // SwitchingFunction tsw;
      // tsw.set(nn_,mm_,r0_,0.0);
      // dmax=tsw.get_dmax();
    }

    switchingParameters.dmaxSQ = dmax * dmax;
    calculateFloat invr0 = 1.0 / r0_;
    switchingParameters.invr0_2 = invr0 * invr0;
    constexpr bool dostretch = true;
    if (dostretch) {
      std::vector<calculateFloat> inputs = {0.0, dmax * invr0};

      thrust::device_vector<calculateFloat> inputZeroMax (2);
      inputZeroMax = inputs;
      thrust::device_vector<calculateFloat> dummydfunc (2);
      thrust::device_vector<calculateFloat> resZeroMax (2);

      PLMD::GPU::getpcuda_Rational<<<1, 2>>> (
          thrust::raw_pointer_cast (inputZeroMax.data()),
          nn_,
          mm_,
          thrust::raw_pointer_cast (dummydfunc.data()),
          thrust::raw_pointer_cast (resZeroMax.data()));

      switchingParameters.stretch = 1.0 / (resZeroMax[0] - resZeroMax[1]);
      switchingParameters.shift = -resZeroMax[1] * switchingParameters.stretch;
    }
  }

  checkRead();
  cudaStreamCreate (&streamDerivatives);
  cudaStreamCreate (&streamVirial);
  cudaStreamCreate (&streamCoordination);
  setUpPermanentGPUMemory();

  log << "  contacts are counted with cutoff (dmax)="
      << sqrt (switchingParameters.dmaxSQ)
      << ", with a rational switch with parameters: d0=0.0, r0="
      << 1.0 / sqrt (switchingParameters.invr0_2)
      << ", N=" << switchingParameters.nn << ", M=" << switchingParameters.mm
      << ".\n";
}

template <typename calculateFloat>
CudaCoordinationWB<calculateFloat>::~CudaCoordinationWB() {
  cudaStreamDestroy (streamDerivatives);
  cudaStreamDestroy (streamVirial);
  cudaStreamDestroy (streamCoordination);
}

#define X(I) 3 * I
#define Y(I) 3 * I + 1
#define Z(I) 3 * I + 2

template <bool usePBC, typename calculateFloat>
__global__ void
getCoordwB (const unsigned natActive,
            const unsigned natLoop,
            const PLMD::GPU::rationalSwitchParameters<calculateFloat>
                switchingParameters,
            const PLMD::GPU::ortoPBCs<calculateFloat> myPBC,
            const calculateFloat *coordActive,
            const calculateFloat *coordLoop,
            const unsigned *trueIndexes,
            calculateFloat *ncoordOut,
            calculateFloat *devOut,
            calculateFloat *virialOut) {
  // blockDIm are the number of threads in your block
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= natActive) { // blocks are initializated with 'ceil (nat/threads)'
    return;
  }
  // we try working with less global memory possible, so we set up a bunch of
  // temporary variables
  const unsigned idx = trueIndexes[i];
  // local results
  calculateFloat mydevX = 0.0;
  calculateFloat mydevY = 0.0;
  calculateFloat mydevZ = 0.0;
  calculateFloat mycoord = 0.0;
  // the previous version used static array for myVirial and d
  // using explicit variables guarantees that this data will be stored in
  // registers
  calculateFloat myVirial_0 = 0.0;
  calculateFloat myVirial_1 = 0.0;
  calculateFloat myVirial_2 = 0.0;
  calculateFloat myVirial_3 = 0.0;
  calculateFloat myVirial_4 = 0.0;
  calculateFloat myVirial_5 = 0.0;
  calculateFloat myVirial_6 = 0.0;
  calculateFloat myVirial_7 = 0.0;
  calculateFloat myVirial_8 = 0.0;
  // local calculation aid
  const calculateFloat x = coordActive[X (i)];
  const calculateFloat y = coordActive[Y (i)];
  const calculateFloat z = coordActive[Z (i)];
  calculateFloat d_0, d_1, d_2;
  calculateFloat t;
  calculateFloat dfunc;
  calculateFloat coord;
  for (unsigned j = 0; j < natLoop; ++j) {
    // const unsigned j = threadIdx.y + blockIdx.y * blockDim.y;

    // Safeguard
    if (idx == trueIndexes[j])
      continue;
    // or may be better to set up an
    // const unsigned xyz = threadIdx.z
    // where the third dim is 0 1 2 ^
    // ?
    if constexpr (usePBC) {
      d_0 = PLMD::GPU::pbcClamp ((coordLoop[X (j)] - x) * myPBC.invX) * myPBC.X;
      d_1 = PLMD::GPU::pbcClamp ((coordLoop[Y (j)] - y) * myPBC.invY) * myPBC.Y;
      d_2 = PLMD::GPU::pbcClamp ((coordLoop[Z (j)] - z) * myPBC.invZ) * myPBC.Z;
    } else {
      d_0 = coordLoop[X (j)] - x;
      d_1 = coordLoop[Y (j)] - y;
      d_2 = coordLoop[Z (j)] - z;
    }

    dfunc = 0.;
    coord = calculateSqr (
        d_0 * d_0 + d_1 * d_1 + d_2 * d_2, switchingParameters, dfunc);
    mycoord += coord;

    t = dfunc * d_0;
    mydevX -= t;

    myVirial_0 -= t * d_0;
    myVirial_1 -= t * d_1;
    myVirial_2 -= t * d_2;

    t = dfunc * d_1;
    mydevY -= t;

    myVirial_3 -= t * d_0;
    myVirial_4 -= t * d_1;
    myVirial_5 -= t * d_2;

    t = dfunc * d_2;
    mydevZ -= t;

    myVirial_6 -= t * d_0;
    myVirial_7 -= t * d_1;
    myVirial_8 -= t * d_2;
  }
  // working in global memory ONLY at the end
  devOut[X (i)] = mydevX;
  devOut[Y (i)] = mydevY;
  devOut[Z (i)] = mydevZ;
  ncoordOut[i] = mycoord;
  virialOut[natActive * 0 + i] = myVirial_0;
  virialOut[natActive * 1 + i] = myVirial_1;
  virialOut[natActive * 2 + i] = myVirial_2;
  virialOut[natActive * 3 + i] = myVirial_3;
  virialOut[natActive * 4 + i] = myVirial_4;
  virialOut[natActive * 5 + i] = myVirial_5;
  virialOut[natActive * 6 + i] = myVirial_6;
  virialOut[natActive * 7 + i] = myVirial_7;
  virialOut[natActive * 8 + i] = myVirial_8;
}

template <bool usePBC, typename calculateFloat>
__global__ void
getDerivB (const unsigned natLoop,
           const unsigned natActive,
           const PLMD::GPU::rationalSwitchParameters<calculateFloat>
               switchingParameters,
           const PLMD::GPU::ortoPBCs<calculateFloat> myPBC,
           const calculateFloat *coordLoop,
           const calculateFloat *coordActive,
           const unsigned *trueIndexes,
           calculateFloat *devOut) {
  // blockDIm are the number of threads in your block
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= natActive) { // blocks are initializated with 'ceil (nat/threads)'
    return;
  }
  // we try working with less global memory possible, so we set up a bunch of
  // temporary variables
  const unsigned idx = trueIndexes[i];
  // local results
  calculateFloat mydevX = 0.0;
  calculateFloat mydevY = 0.0;
  calculateFloat mydevZ = 0.0;
  calculateFloat mycoord = 0.0;
  // the previous version used static array for myVirial and d
  // using explicit variables guarantees that this data will be stored in
  // registers
  calculateFloat myVirial_0 = 0.0;
  calculateFloat myVirial_1 = 0.0;
  calculateFloat myVirial_2 = 0.0;
  calculateFloat myVirial_3 = 0.0;
  calculateFloat myVirial_4 = 0.0;
  calculateFloat myVirial_5 = 0.0;
  calculateFloat myVirial_6 = 0.0;
  calculateFloat myVirial_7 = 0.0;
  calculateFloat myVirial_8 = 0.0;
  // local calculation aid
  const calculateFloat x = coordActive[X (i)];
  const calculateFloat y = coordActive[Y (i)];
  const calculateFloat z = coordActive[Z (i)];
  calculateFloat d_0, d_1, d_2;
  calculateFloat t;
  calculateFloat dfunc;
  calculateFloat coord;
  for (unsigned j = 0; j < natLoop; ++j) {
    // const unsigned j = threadIdx.y + blockIdx.y * blockDim.y;

    // Safeguard
    if (idx == trueIndexes[j])
      continue;
    // or may be better to set up an
    // const unsigned xyz = threadIdx.z
    // where the third dim is 0 1 2 ^
    // ?
    if constexpr (usePBC) {
      d_0 = PLMD::GPU::pbcClamp ((coordLoop[X (j)] - x) * myPBC.invX) * myPBC.X;
      d_1 = PLMD::GPU::pbcClamp ((coordLoop[Y (j)] - y) * myPBC.invY) * myPBC.Y;
      d_2 = PLMD::GPU::pbcClamp ((coordLoop[Z (j)] - z) * myPBC.invZ) * myPBC.Z;
    } else {
      d_0 = coordLoop[X (j)] - x;
      d_1 = coordLoop[Y (j)] - y;
      d_2 = coordLoop[Z (j)] - z;
    }

    dfunc = 0.;
    t = calculateSqr (
        d_0 * d_0 + d_1 * d_1 + d_2 * d_2, switchingParameters, dfunc);

    mydevX -= dfunc * d_0;

    mydevY -= dfunc * d_1;
    mydevZ -= dfunc * d_2;
  }
  // working in global memory ONLY at the end
  devOut[X (i)] = mydevX;
  devOut[Y (i)] = mydevY;
  devOut[Z (i)] = mydevZ;
}

#define getCoordOrthoPBC getCoordwB<true>
#define getCoordNoPBC getCoordwB<false>

#define getDerivOrthoPBC getDerivB<true>
#define getDerivNoPBC getDerivB<false>

template <typename calculateFloat>
void CudaCoordinationWB<calculateFloat>::calculate() {
  constexpr unsigned dataperthread = 4;
  auto positions = getPositions();

  /***************************copying data on the GPU**************************/
  CUDAHELPERS::plmdDataToGPU (cudaPositions, positions, streamDerivatives);
  /***************************copying data on the GPU**************************/

  Tensor virial;
  double coordination;
  auto deriv = std::vector<Vector> (positions.size());

  // constexpr unsigned nthreads = 512;

  unsigned ngroupsA = ceil (double (atomsInA) / maxNumThreads);
  unsigned ngroupsB = ceil (double (atomsInB) / maxNumThreads);

  /**********************allocating the memory on the GPU**********************/
  cudaCoordination.resize (atomsInA);
  cudaVirial.resize (atomsInA * 9);
  /**************************starting the calculations*************************/
  // this calculates the derivatives and prepare the coordination and the
  // virial for the accumulation
  if (pbc) {
    // Only ortho as now
    auto box = getBox();

    myPBC.X = box (0, 0);
    myPBC.Y = box (1, 1);
    myPBC.Z = box (2, 2);
    myPBC.invX = 1.0 / myPBC.X;
    myPBC.invY = 1.0 / myPBC.Y;
    myPBC.invZ = 1.0 / myPBC.Z;

    getCoordOrthoPBC<<<ngroupsA, maxNumThreads, 0, streamDerivatives>>> (
        atomsInA,
        atomsInB,
        switchingParameters,
        myPBC,
        thrust::raw_pointer_cast (cudaPositions.data()),
        thrust::raw_pointer_cast (cudaPositions.data()) + 3 * atomsInA,
        thrust::raw_pointer_cast (cudaTrueIndexes.data()),
        thrust::raw_pointer_cast (cudaCoordination.data()),
        thrust::raw_pointer_cast (cudaDerivatives.data()),
        thrust::raw_pointer_cast (cudaVirial.data()));

    getDerivOrthoPBC<<<ngroupsB, maxNumThreads, 0, streamDerivatives>>> (
        atomsInA,
        atomsInB,
        switchingParameters,
        myPBC,
        thrust::raw_pointer_cast (cudaPositions.data()),
        thrust::raw_pointer_cast (cudaPositions.data()) + 3 * atomsInA,
        thrust::raw_pointer_cast (cudaTrueIndexes.data()),
        thrust::raw_pointer_cast (cudaDerivatives.data()) + 3 * atomsInA);
  } else {
    getCoordNoPBC<<<ngroupsA, maxNumThreads, 0, streamDerivatives>>> (
        atomsInA,
        atomsInB,
        switchingParameters,
        myPBC,
        thrust::raw_pointer_cast (cudaPositions.data()),
        thrust::raw_pointer_cast (cudaPositions.data()) + 3 * atomsInA,
        thrust::raw_pointer_cast (cudaTrueIndexes.data()),
        thrust::raw_pointer_cast (cudaCoordination.data()),
        thrust::raw_pointer_cast (cudaDerivatives.data()),
        thrust::raw_pointer_cast (cudaVirial.data()));

    getDerivNoPBC<<<ngroupsB, maxNumThreads, 0, streamDerivatives>>> (
        atomsInA,
        atomsInB,
        switchingParameters,
        myPBC,
        thrust::raw_pointer_cast (cudaPositions.data()),
        thrust::raw_pointer_cast (cudaPositions.data()) + 3 * atomsInA,
        thrust::raw_pointer_cast (cudaTrueIndexes.data()),
        thrust::raw_pointer_cast (cudaDerivatives.data()) + 3 * atomsInA);
  }

  /**************************accumulating the results**************************/

  cudaDeviceSynchronize();

  CUDAHELPERS::plmdDataFromGPU (cudaDerivatives, deriv, streamDerivatives);

  auto N = atomsInA;

  while (N > 1) {
    size_t runningThreads = CUDAHELPERS::threadsPerBlock (
        ceil (double (N) / dataperthread), maxNumThreads);

    unsigned nGroups = ceil (double (N) / (runningThreads * dataperthread));

    reductionMemoryVirial.resize (9 * nGroups);
    reductionMemoryCoord.resize (nGroups);

    dim3 ngroupsVirial (nGroups, 9);
    CUDAHELPERS::doReductionND<dataperthread> (
        thrust::raw_pointer_cast (cudaVirial.data()),
        thrust::raw_pointer_cast (reductionMemoryVirial.data()),
        N,
        ngroupsVirial,
        runningThreads,
        streamVirial);

    CUDAHELPERS::doReduction1D<dataperthread> (
        thrust::raw_pointer_cast (cudaCoordination.data()),
        thrust::raw_pointer_cast (reductionMemoryCoord.data()),
        N,
        nGroups,
        runningThreads,
        streamCoordination);

    if (nGroups == 1) {
      CUDAHELPERS::plmdDataFromGPU (
          reductionMemoryVirial, virial, streamVirial);
      // TODO:find a way to stream this
      coordination = reductionMemoryCoord[0];
    } else {
      reductionMemoryVirial.swap (cudaVirial);
      reductionMemoryCoord.swap (cudaCoordination);
    }

    N = nGroups;
  }

  // in this way we do not resize with additional memory allocation
  if (reductionMemoryCoord.size() > cudaCoordination.size())
    reductionMemoryCoord.swap (cudaCoordination);
  if (reductionMemoryVirial.size() > cudaVirial.size())
    reductionMemoryVirial.swap (cudaVirial);
  // this ensures that the memory is fully in the host ram
  cudaDeviceSynchronize();
  for (unsigned i = 0; i < deriv.size(); ++i)
    setAtomsDerivatives (i, deriv[i]);

  setValue (coordination);
  setBoxDerivatives (virial);
}
#undef getCoordOrthoPBC
#undef getCoordNoPBC

} // namespace colvar
} // namespace PLMD