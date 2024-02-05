## Implement the couple coordination

This should be the easiest implementation: 
If each kernel will run only on a couple (but here we are losing time in spinning up the various kernels) it could be written like this:

```c++
template <bool usePBC, typename calculateFloat>
__global__ void
getCoord (const unsigned couples,
          const PLMD::GPU::rationalSwitchParameters<calculateFloat>
              switchingParameters,
          const PLMD::GPU::ortoPBCs<calculateFloat> myPBC,
          const calculateFloat *coordinates,
          const unsigned *trueIndexes,
          calculateFloat *ncoordOut,
          calculateFloat *devOut,
          calculateFloat *virialOut) {
  // blockDIm are the number of threads in your block
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned j = i + couples;
  if (i >= couples) { // blocks are initializated with 'ceil (nat/threads)'
    return;
  }
  // we try working with less global memory possible, so we set up a bunch of
  // temporary variables
  const unsigned idx = trueIndexes[i];
  const unsigned jdx = trueIndexes[j];
  if (idx == jdx) {
    return; 
  }

  // local calculation aid
  calculateFloat d_0, d_1, d_2;
  calculateFloat dfunc;

  if constexpr (usePBC) {
    d_0 = PLMD::GPU::pbcClamp ((coordinates[X (j)] - coordinates[X (i)]) * myPBC.invX) * myPBC.X;
    d_1 = PLMD::GPU::pbcClamp ((coordinates[Y (j)] - coordinates[Y (i)]) * myPBC.invY) * myPBC.Y;
    d_2 = PLMD::GPU::pbcClamp ((coordinates[Z (j)] - coordinates[Z (i)]) * myPBC.invZ) * myPBC.Z;
  } else {
    d_0 = coordinates[X (j)] - coordinates[X (i)];
    d_1 = coordinates[Y (j)] - coordinates[Y (i)];
    d_2 = coordinates[Z (j)] - coordinates[Z (i)];
  }

  dfunc = 0.;
  ncoordOut[i] = calculateSqr (
      d_0 * d_0 + d_1 * d_1 + d_2 * d_2, switchingParameters, dfunc);

  mydevX = dfunc * d_0;
  mydevY = dfunc * d_1;
  mydevZ = dfunc * d_2;

  devOut[X (i)] = mydevX;
  devOut[Y (i)] = mydevY;
  devOut[Z (i)] = mydevZ;
  devOut[X (j)] = -mydevX;
  devOut[Y (j)] = -mydevY;
  devOut[Z (j)] = -mydevZ;
  
  virialOut[couples * 0 + i] = -dfunc * d[0] * d[0];
  virialOut[couples * 1 + i] = -dfunc * d[0] * d[1];
  virialOut[couples * 2 + i] = -dfunc * d[0] * d[2];
  virialOut[couples * 3 + i] = -dfunc * d[1] * d[0];
  virialOut[couples * 4 + i] = -dfunc * d[1] * d[1];
  virialOut[couples * 5 + i] = -dfunc * d[1] * d[2];
  virialOut[couples * 6 + i] = -dfunc * d[2] * d[0];
  virialOut[couples * 7 + i] = -dfunc * d[2] * d[1];
  virialOut[couples * 8 + i] = -dfunc * d[2] * d[2];
}
```

