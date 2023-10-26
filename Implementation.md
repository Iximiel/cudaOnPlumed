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

## The coord kernel

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
