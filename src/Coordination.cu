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
#include "plumed/colvar/CoordinationBase.h"
#include "plumed/core/ActionRegister.h"
#include "plumed/tools/NeighborList.h"
#include "plumed/tools/SwitchingFunction.h"

#include "cudaHelpers.cuh"
#include "ndReduction.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

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

// these constant will be used within the kernels
template <typename calculateFloat> struct rationalSwitchParameters {
  calculateFloat dmaxSQ = std::numeric_limits<calculateFloat>::max();
  calculateFloat invr0_2 = 1.0; // r0=1
  calculateFloat stretch = 1.0;
  calculateFloat shift = 0.0;
  int nn = 6;
  int mm = 12;
};

template <typename calculateFloat> struct ortoPBCs {
  calculateFloat invX = 1.0;
  calculateFloat invY = 1.0;
  calculateFloat invZ = 1.0;
  calculateFloat X = 1.0;
  calculateFloat Y = 1.0;
  calculateFloat Z = 1.0;
};

template <typename calculateFloat>
__device__ calculateFloat pbcClamp(calculateFloat x) {
  return 0.0;
}

template <> __device__ double pbcClamp<double>(double x) {
  // convert a double to a signed int in round-to-nearest-even mode.
  return __double2int_rn(x) - x;
  // return x - floor(x+0.5);
  // Round argument x to an integer value in single precision floating-point
  // format.
  // Uses round to nearest rounding, with ties rounding to even.
  // return nearbyint(x) - x;
}

template <> __device__ float pbcClamp<float>(float x) {
  // convert a double to a signed int in round-to-nearest-even mode.
  return __float2int_rn(x) - x;
  // return x - floorf(x+0.5f);
  // return nearbyintf(x) - x;
}

// does not inherit from coordination base because nl is private
template <typename calculateFloat> class CudaCoordination : public Colvar {
  /// the pointer to the coordinates on the GPU
  CUDAHELPERS::memoryHolder<calculateFloat> cudaPositions;
  /// the pointer to the nn list on the GPU
  CUDAHELPERS::memoryHolder<calculateFloat> cudaCoordination;
  CUDAHELPERS::memoryHolder<calculateFloat> cudaDerivatives;
  CUDAHELPERS::memoryHolder<calculateFloat> cudaVirial;
  CUDAHELPERS::memoryHolder<calculateFloat> reductionMemoryVirial;
  CUDAHELPERS::memoryHolder<calculateFloat> reductionMemoryCoord;
  CUDAHELPERS::memoryHolder<unsigned> cudaTrueIndexes;

  cudaStream_t streamDerivatives;
  cudaStream_t streamVirial;
  cudaStream_t streamCoordination;

  unsigned maxNumThreads = 512;
  rationalSwitchParameters<calculateFloat> switchingParameters;
  ortoPBCs<calculateFloat> myPBC;

  bool pbc{true};
  void setUpPermanentGPUMemory();

public:
  explicit CudaCoordination(const ActionOptions &);
  virtual ~CudaCoordination();
  // active methods:
  static void registerKeywords(Keywords &keys);
  void calculate() override;
};
using CudaCoordination_d = CudaCoordination<double>;
using CudaCoordination_f = CudaCoordination<float>;
PLUMED_REGISTER_ACTION(CudaCoordination_d, "CUDACOORDINATION")
PLUMED_REGISTER_ACTION(CudaCoordination_f, "CUDACOORDINATIONFLOAT")

template <typename calculateFloat>
void CudaCoordination<calculateFloat>::setUpPermanentGPUMemory() {
  auto nat = getPositions().size();
  cudaPositions.resize(3 * nat);
  cudaDerivatives.resize(3 * nat);
  cudaTrueIndexes.resize(nat);
  std::vector<unsigned> trueIndexes(nat);
  for (size_t i = 0; i < nat; ++i) {
    trueIndexes[i] = getAbsoluteIndex(i).index();
  }
  cudaTrueIndexes.copyToCuda(trueIndexes.data(), streamDerivatives);
}

template <typename calculateFloat>
void CudaCoordination<calculateFloat>::registerKeywords(Keywords &keys) {
  Colvar::registerKeywords(keys);

  keys.add("optional", "THREADS", "The upper limit of the number of threads");
  keys.add("atoms", "GROUPA", "First list of atoms");

  keys.add("compulsory", "NN", "6",
           "The n parameter of the switching function ");
  keys.add("compulsory", "MM", "0",
           "The m parameter of the switching function; 0 implies 2*NN");
  keys.add("compulsory", "R_0", "The r_0 parameter of the switching function");
  keys.add("compulsory", "D_MAX", "0.0",
           "The cut off of the switching function");
}

template <typename calculateFloat>
__device__ calculateFloat pcuda_fastpow(calculateFloat base, int expo) {
  if (expo < 0) {
    expo = -expo;
    base = 1.0 / base;
  }
  calculateFloat result = 1.0;
  while (expo) {
    if (expo & 1)
      result *= base;
    expo >>= 1;
    base *= base;
  }
  return result;
}

template <typename calculateFloat>
__device__ calculateFloat pcuda_Rational(const calculateFloat rdist,
                                         const int NN, const int MM,
                                         calculateFloat &dfunc) {
  calculateFloat result;
  if (2 * NN == MM) {
    // if 2*N==M, then (1.0-rdist^N)/(1.0-rdist^M) = 1.0/(1.0+rdist^N)
    calculateFloat rNdist = pcuda_fastpow(rdist, NN - 1);
    result = 1.0 / (1 + rNdist * rdist);
    dfunc = -NN * rNdist * result * result;
  } else {
    if (rdist > (1. - 10.0 * std::numeric_limits<calculateFloat>::epsilon()) &&
        rdist < (1 + 10.0 * std::numeric_limits<calculateFloat>::epsilon())) {
      result = NN / MM;
      dfunc = 0.5 * NN * (NN - MM) / MM;
    } else {
      calculateFloat rNdist = pcuda_fastpow(rdist, NN - 1);
      calculateFloat rMdist = pcuda_fastpow(rdist, MM - 1);
      calculateFloat num = 1. - rNdist * rdist;
      calculateFloat iden = 1.0 / (1.0 - rMdist * rdist);
      result = num * iden;
      dfunc = ((-NN * rNdist * iden) + (result * (iden * MM) * rMdist));
    }
  }
  return result;
}

template <typename calculateFloat>
__global__ void getpcuda_Rational(const calculateFloat *rdists, const int NN,
                                  const int MM, calculateFloat *dfunc,
                                  calculateFloat *res) {
  const int i = threadIdx.x + blockIdx.x * blockDim.x;
  if (rdists[i] <= 0.) {
    res[i] = 1.;
    dfunc[i] = 0.0;
  } else
    res[i] = pcuda_Rational(rdists[i], NN, MM, dfunc[i]);
  // printf("stretch: %i: %f -> %f\n",i,rdists[i],res[i]);
}

// __global__ void getConst() {
//   printf("Cuda: cu_epsilon = %f\n", cu_epsilon);
// }

template <typename calculateFloat>
CudaCoordination<calculateFloat>::CudaCoordination(const ActionOptions &ao)
    : PLUMED_COLVAR_INIT(ao) {
  std::vector<AtomNumber> GroupA;
  parseAtomList("GROUPA", GroupA);

  bool nopbc = !pbc;
  parseFlag("NOPBC", nopbc);
  pbc = !nopbc;

  parse("THREADS", maxNumThreads);
  if (maxNumThreads <= 0)
    error("THREADS should be positive");
  addValueWithDerivatives();
  setNotPeriodic();
  requestAtoms(GroupA);

  log.printf("  \n");
  if (pbc)
    log.printf("  using periodic boundary conditions\n");
  else
    log.printf("  without periodic boundary conditions\n");

  std::string sw, errors;

  { // loading data to the GPU
    int nn_ = 6;
    int mm_ = 0;

    calculateFloat r0_ = 0.0;
    parse("R_0", r0_);
    if (r0_ <= 0.0) {
      error("R_0 should be explicitly specified and positive");
    }

    parse("NN", nn_);
    parse("MM", mm_);
    if (mm_ == 0) {
      mm_ = 2 * nn_;
    }
    if (mm_ % 2 != 0 || mm_ % 2 != 0)
      error(" this implementation only works with both MM and NN even");

    switchingParameters.nn = nn_;
    switchingParameters.mm = mm_;
    switchingParameters.stretch = 1.0;
    switchingParameters.shift = 0.0;

    calculateFloat dmax = 0.0;
    parse("D_MAX", dmax);
    if (dmax == 0.0) { // TODO:check for a "non present flag"
      // set dmax to where the switch is ~0.00001
      dmax = r0_ * std::pow(0.00001, 1.0 / (nn_ - mm_));
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

      CUDAHELPERS::memoryHolder<calculateFloat> inputZeroMax(2);
      inputZeroMax.copyToCuda(inputs.data());
      CUDAHELPERS::memoryHolder<calculateFloat> dummydfunc(2);
      CUDAHELPERS::memoryHolder<calculateFloat> resZeroMax(2);

      getpcuda_Rational<<<1, 2>>>(inputZeroMax.pointer(), nn_, mm_,
                                  dummydfunc.pointer(), resZeroMax.pointer());
      std::vector<calculateFloat> s = {0.0, 0.0};
      resZeroMax.copyFromCuda(s.data());

      switchingParameters.stretch = 1.0 / (s[0] - s[1]);
      switchingParameters.shift = -s[1] * switchingParameters.stretch;
    }
  }

  checkRead();
  cudaStreamCreate(&streamDerivatives);
  cudaStreamCreate(&streamVirial);
  cudaStreamCreate(&streamCoordination);
  setUpPermanentGPUMemory();

  log << "  contacts are counted with cutoff (dmax)="
      << sqrt(switchingParameters.dmaxSQ)
      << ", with a rational switch with parameters: d0=0.0, r0="
      << 1.0 / sqrt(switchingParameters.invr0_2)
      << ", N=" << switchingParameters.nn << ", M=" << switchingParameters.mm
      << ".\n";
}

template <typename calculateFloat>
CudaCoordination<calculateFloat>::~CudaCoordination() {
  cudaStreamDestroy(streamDerivatives);
  cudaStreamDestroy(streamVirial);
  cudaStreamDestroy(streamCoordination);
}

template <typename calculateFloat>
__device__ calculateFloat
calculateSqr(const calculateFloat distancesq,
             const rationalSwitchParameters<calculateFloat> switchingParameters,
             calculateFloat &dfunc) {
  calculateFloat result = 0.0;
  dfunc = 0.0;
  if (distancesq < switchingParameters.dmaxSQ) {
    const calculateFloat rdist_2 = distancesq * switchingParameters.invr0_2;
    result = pcuda_Rational(rdist_2, switchingParameters.nn / 2,
                            switchingParameters.mm / 2, dfunc);
    // chain rule:
    dfunc *= 2 * switchingParameters.invr0_2;
    // cu_stretch:
    result = result * switchingParameters.stretch + switchingParameters.shift;
    dfunc *= switchingParameters.stretch;
  }
  return result;
}

#define X(I) 3 * I
#define Y(I) 3 * I + 1
#define Z(I) 3 * I + 2

template <bool usePBC, typename calculateFloat>
__global__ void
getCoord(const unsigned nat,
         const rationalSwitchParameters<calculateFloat> switchingParameters,
         const ortoPBCs<calculateFloat> myPBC,
         const calculateFloat *coordinates, const unsigned *trueIndexes,
         calculateFloat *ncoordOut, calculateFloat *devOut,
         calculateFloat *virialOut) {
  // blockDIm are the number of threads in your block
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= nat) // blocks are initializated with 'ceil (nat/threads)'
    return;
  // we try working with less global memory possible
  const unsigned idx = trueIndexes[i];
  // local results
  calculateFloat mydevX = 0.0;
  calculateFloat mydevY = 0.0;
  calculateFloat mydevZ = 0.0;
  calculateFloat mycoord = 0.0;
  calculateFloat myVirial[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  // local calculation aid
  calculateFloat x = coordinates[X(i)];
  calculateFloat y = coordinates[Y(i)];
  calculateFloat z = coordinates[Z(i)];
  calculateFloat d[3];
  calculateFloat dfunc;
  calculateFloat coord;
  for (unsigned j = 0; j < nat; ++j) {
    // const unsigned j = threadIdx.y + blockIdx.y * blockDim.y;

    // Safeguard
    if (idx == trueIndexes[j])
      continue;
    // or may be better to set up an
    // const unsigned xyz = threadIdx.z
    // where the third dim is 0 1 2 ^
    if (usePBC) {
      d[0] = pbcClamp((coordinates[X(j)] - x) * myPBC.invX) * myPBC.X;
      d[1] = pbcClamp((coordinates[Y(j)] - y) * myPBC.invY) * myPBC.Y;
      d[2] = pbcClamp((coordinates[Z(j)] - z) * myPBC.invZ) * myPBC.Z;

    } else {
      d[0] = coordinates[X(j)] - x;
      d[1] = coordinates[Y(j)] - y;
      d[2] = coordinates[Z(j)] - z;
    }

    dfunc = 0.;
    coord = calculateSqr(d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
                         switchingParameters, dfunc);
    mydevX -= dfunc * d[0];
    mydevY -= dfunc * d[1];
    mydevZ -= dfunc * d[2];
    if (i < j) {
      mycoord += coord;
      myVirial[0] -= dfunc * d[0] * d[0];
      myVirial[1] -= dfunc * d[0] * d[1];
      myVirial[2] -= dfunc * d[0] * d[2];
      myVirial[3] -= dfunc * d[1] * d[0];
      myVirial[4] -= dfunc * d[1] * d[1];
      myVirial[5] -= dfunc * d[1] * d[2];
      myVirial[6] -= dfunc * d[2] * d[0];
      myVirial[7] -= dfunc * d[2] * d[1];
      myVirial[8] -= dfunc * d[2] * d[2];
    }
  }
  // working in global memory ONLY at the end
  devOut[X(i)] = mydevX;
  devOut[Y(i)] = mydevY;
  devOut[Z(i)] = mydevZ;
  ncoordOut[i] = mycoord;
  virialOut[nat * 0 + i] = myVirial[0];
  virialOut[nat * 1 + i] = myVirial[1];
  virialOut[nat * 2 + i] = myVirial[2];
  virialOut[nat * 3 + i] = myVirial[3];
  virialOut[nat * 4 + i] = myVirial[4];
  virialOut[nat * 5 + i] = myVirial[5];
  virialOut[nat * 6 + i] = myVirial[6];
  virialOut[nat * 7 + i] = myVirial[7];
  virialOut[nat * 8 + i] = myVirial[8];
}

#define getCoordOrthoPBC getCoord<true>
#define getCoordNoPBC getCoord<false>

template <typename calculateFloat>
void CudaCoordination<calculateFloat>::calculate() {
  auto positions = getPositions();
  /***************************copying data on the GPU**************************/
  cudaPositions.copyToCuda(&positions[0][0], streamDerivatives);
  /***************************copying data on the GPU**************************/
  auto nat = positions.size();

  Tensor virial;
  double coordination;
  auto deriv = std::vector<Vector>(nat);

  constexpr unsigned nthreads = 512;

  unsigned ngroups = ceil(double(nat) / nthreads);

  /**********************allocating the memory on the GPU**********************/
  cudaCoordination.resize(nat);
  cudaVirial.resize(nat * 9);
  /**************************starting the calculations*************************/
  // this calculates the derivatives and prepare the coordination and the virial
  // for the accumulation
  if (pbc) {
    // Only ortho as now
    auto box = getBox();

    myPBC.X = box(0, 0);
    myPBC.Y = box(1, 1);
    myPBC.Z = box(2, 2);
    myPBC.invX = 1.0 / myPBC.X;
    myPBC.invY = 1.0 / myPBC.Y;
    myPBC.invZ = 1.0 / myPBC.Z;

    getCoordOrthoPBC<<<ngroups, nthreads, 0, streamDerivatives>>>(
        nat, switchingParameters, myPBC, cudaPositions.pointer(),
        cudaTrueIndexes.pointer(), cudaCoordination.pointer(),
        cudaDerivatives.pointer(), cudaVirial.pointer());
  } else {
    getCoordNoPBC<<<ngroups, nthreads, 0, streamDerivatives>>>(
        nat, switchingParameters, myPBC, cudaPositions.pointer(),
        cudaTrueIndexes.pointer(), cudaCoordination.pointer(),
        cudaDerivatives.pointer(), cudaVirial.pointer());
  }

  /**************************accumulating the results**************************/

  cudaDeviceSynchronize();

  cudaDerivatives.copyFromCuda(&deriv[0][0], streamDerivatives);
  auto N = nat;

  while (N > 1) {
    size_t runningThreads = CUDAHELPERS::threadsPerBlock(N, maxNumThreads);
    unsigned nGroups = CUDAHELPERS::idealGroups(N, runningThreads);

    reductionMemoryCoord.resize(nGroups);
    reductionMemoryVirial.resize(9 * nGroups);

    dim3 ngroupsVirial(nGroups, 9);
    CUDAHELPERS::doReductionND(cudaVirial.pointer(),
                               reductionMemoryVirial.pointer(), N,
                               ngroupsVirial, runningThreads, streamVirial);

    CUDAHELPERS::doReduction1D(cudaCoordination.pointer(),
                               reductionMemoryCoord.pointer(), N, nGroups,
                               runningThreads, streamCoordination);

    if (nGroups == 1) {
      reductionMemoryVirial.copyFromCuda(&virial[0][0], streamVirial);
      reductionMemoryCoord.copyFromCuda(&coordination, streamCoordination);
    } else {
      reductionMemoryCoord.swap(cudaCoordination);
      reductionMemoryVirial.swap(cudaVirial);
    }

    N = nGroups;
  }
  // in this way we do not resize with additional memory allocation
  if (reductionMemoryCoord.reserved() > cudaCoordination.reserved())
    reductionMemoryCoord.swap(cudaCoordination);
  if (reductionMemoryVirial.reserved() > cudaVirial.reserved())
    reductionMemoryVirial.swap(cudaVirial);
  // this ensures that the memory is fully in the host ram
  cudaDeviceSynchronize();
  for (unsigned i = 0; i < deriv.size(); ++i)
    setAtomsDerivatives(i, deriv[i]);

  setValue(coordination);
  setBoxDerivatives(virial);
}
#undef getCoordOrthoPBC
#undef getCoordNoPBC

} // namespace colvar
} // namespace PLMD
