# Implementation

## The switching function
The coordination is more or less calculated with 
$\frac{1}{2} \sum^{i=0}_{i<nat}\sum^{j=0}_{j<nat,j\neq i}f(d_{ij})$
, where $d_{ij}$ is the distances between atom $i$ and $j$, and $f(x)$ is a function that usually has the form

$$
f(x)=\begin{cases}
1       & x \leq d_0\\
s(x)    & d_0<x\leq d_{max}\\
0       & x > d_{max} 
\end{cases}
$$

and $s(x)$ is a switching function that links smoothly 1 to 0 within $d_0$ and $d_{max}$

In this case I used the RATIONAL function like in the default plumed implementation: $s(r)=\frac{ 1 - \left(\frac{ r - d_0 }{ r_0 }\right)^{n} }{ 1 - \left(\frac{ r - d_0 }{ r_0 }\right)^{m} }$.

The implementation of the rational function follows the one in plumed.
But for simplicity, it is not implemented with the parameter $d_0$, so $s(r)=\frac{ 1 - \left(\frac{ r }{ r_0 }\right)^{n} }{ 1 - \left(\frac{ r }{ r_0 }\right)^{m} }$.

Like in standard plumed, the switch has a stretch parameter: if $s(d_{max}) = shift $ and $stretch=\frac{1}{s(0)-s(d_{max})}$, $s(x)$ becames $s^s(x)=s(x)*stretch+shift$

## The kernel and the  \_\_device\_\_ functions
### The coord kernel

The simplest kernel for calculating the coordination in a group of atoms is:
```c++
#define X(I) 3 * I
#define Y(I) 3 * I + 1
#define Z(I) 3 * I + 2
__global__ void getSelfCoord(
    const unsigned nat,
    const rationalSwitchParameters switchingParameters,
    const ortoPBCs myPBC, const float *coordinates,
    const unsigned *trueIndexes, float *ncoordOut,
    float *devOut, float *virialOut) {
        
  // i is the index of the atoms that will be confronted with all the others
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;

  // blocks are initializated with 'ceil (nat/threads)'
  if (i >= nat) 
    return;
  // we try working with less global memory possible
  // so we set up some temporary variables private to this kernel
  const unsigned idx = trueIndexes[i];
  //local results
  float mydevX = 0.0;
  float mydevY = 0.0;
  float mydevZ = 0.0;
  float mycoord = 0.0;
  float myVirial[9] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  //local calculation aid
  float x = coordinates[X(i)];
  float y = coordinates[Y(i)];
  float z = coordinates[Z(i)];
  float d[3];
  float dfunc;
  float coord;
  for (unsigned j = 0; j < nat; ++j) {
    // Safeguard against self interaction
    if (idx == trueIndexes[j])
      continue;
    
    d[0] = coordinates[X(j)] - x;
    d[1] = coordinates[Y(j)] - y;
    d[2] = coordinates[Z(j)] - z;
    

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
  // updating global memory ONLY at the end
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
```
This kernel is the inner body of the nested for loop that can be used to write the most naive serial implementation of the coordination.

The use of global memory should be limited in a kernel, to limit the global memory access (and hence use local memory within the kernel or shared memory within the group). Here within a kernel I have accumulated the values in local variables and not on the global memory and then update the global memory only at the end of the calculation, for example I use `mycoord += coord;` within the loop and `ncoordOut[i] = mycoord;` only at the end of kernel.

The distance is calculated by using the stored x,y,z and only accessing the global memory on the "`j` side" within the llop.

### The switching function

The switching parameters are stored in a simple struct, when called kernels accepts struct input, hence is easier to pass collection of parameters:
```c++
struct rationalSwitchParameters {
  float dmaxSQ = std::numeric_limits<float>::max();
  float invr0_2 = 1.0; // r0=1
  float stretch = 1.0;
  float shift = 0.0;
  int nn = 6;
  int mm = 12;
};
```

`calculateSqr` is a simple interface to the actual switching function, and calculates it for the squared distance, and it works exacly as the `PLMD::SwitchingFunction::calculateSqr` method:
```c++
__device__ float
calculateSqr(const float distancesq,
             const rationalSwitchParameters switchingParameters,
             float &dfunc) {
  float result = 0.0;
  dfunc = 0.0;
  if (distancesq < switchingParameters.dmaxSQ) {
    const float rdist_2 = distancesq * switchingParameters.invr0_2;
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
``` 
so as the rational, that it is exacly like `PLMD::SwitchingFunction::do_rational`:
```c++
__device__ float pcuda_Rational(const float rdist, const int NN,
                                         const int MM, float &dfunc) {
  float result;
  if (2 * NN == MM) {
    // if 2*N==M, then (1.0-rdist^N)/(1.0-rdist^M) = 1.0/(1.0+rdist^N)
    float rNdist = pcuda_fastpow(rdist, NN - 1);
    result = 1.0 / (1 + rNdist * rdist);
    dfunc = -NN * rNdist * result * result;
    
  } else {
    if (rdist > (1. - 100.0 * cu_epsilon) && rdist < (1 + 100.0 * cu_epsilon)) {
      result = NN / MM;
      dfunc = 0.5 * NN * (NN - MM) / MM;
    } else {
      float rNdist = pcuda_fastpow(rdist, NN - 1);
      float rMdist = pcuda_fastpow(rdist, MM - 1);
      float num = 1. - rNdist * rdist;
      float iden = 1.0 / (1.0 - rMdist * rdist);
      result = num * iden;
      dfunc = ((-NN * rNdist * iden) + (result * (iden * MM) * rMdist));
    }
  }
  return result;
}
```

## Interfacing to cuda: helpers, invoking the kernel and the reductions

### cudaHelpers.cuh
cudaHelpers.cuh is an very simple template-only library (this interface is made ad hoc for this project) that contains a simple interface for working with the memory in cuda.

`template <typename T> class memoryHolder;` will ease the memory management with an RAII approach (so no need to remember to call `cudaFree`).
It `memoryHolder` is set up as a move-only object in a `std::unique_ptr` style. It initializes the wanted quantity of memory on construction. And has the following methods:
 - **size** returns the size of the memory allocated on the gpu that is accessible with the `copyTo` and `copyFrom` methods (in numbers of `T`, like in the STL)
 - **reserved** returns the size of the memory allocated on the gpu (in numbers of `T`, like in the STL), it may be bigger than the number returned by size
 - **resize** changes the size of the memory accessible, if the asked size is greater than the reserved memory, the memory will be reallocated, otherwise will only be changed the avaiable memory to the `copyTo` and `copyFrom` methods. A parameter regulates if the new array should contain the old data during a reallocation
 - **reserve** reallocate the memory on the GPU (after calling this method `size==reserved`). A parameter regulates if the new array should contain the old data during a reallocation
 - **copyToCuda** copies `size` `T` elements from the CPU to the GPU memory (if called with a type different from `T` a conversion will be made before copying the memory)
 - **copyFromCuda**  copies `size` `T` elements from the GPU to the CPU memory (if called with a type different from `T` a conversion will be made after copying the memory)

 both `copyTo` and `copyFrom` have an overloaded async version that is invoked by calling the function with specifying a stream as second argument