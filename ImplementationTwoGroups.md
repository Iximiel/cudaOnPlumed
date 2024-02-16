## Premise

We now try to change what we made in the [auto-coordination](Implementation.md) and determine the coordination between two groups.

I am assuming that you have already read the details about how the coordination is implemented in the [other chapter](Implementation.md).
So here you will see some more abstract pseudo code. Always refer to my (evolving) [repository]() for a more accurate and performant code.

## Wait, and the derivatives?
First of all let's hastily rewrite the core kernel, stripped by the declarations 

```c++
__global__ void getSelfCoord(
    const unsigned nat,
    //you can pass structs directly to kernels
    const rationalSwitchParameters switchingParameters,
    const float *coordinates,
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
  /* declaring calculation and output variables
  ...
  */
  for (unsigned j = 0; j < nat; ++j) {
    // Safeguard against self interaction
    if (idx == trueIndexes[j])
      continue;
    
    d[0] = coordinates[3*j] - x;
    d[1] = coordinates[3*j+1] - y;
    d[2] = coordinates[3*j+2] - z;
    

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
/* setting the data to the output
  ...
  */
}
```
You see what is the problem here?

Actwally two: `for (unsigned j = 0; j < nat; ++j)` is a loop running on ALL the atoms and that `if (i < j)` makes this loop working "only" on the upper triange of the "square matrix of calculations".
This affermation is actually not true: the loop works for all atoms, but for the upper triangle matrix the loop also sums the coordination and the virial (that are cnotribuited once for each couple) and the derivative for everithing else.

How can we work around this?

Let's start to aknowledge that we will need to have different specialized kernels (since branching within a kernel will generate loss of performaces).

Then we change the signature of the new kernel:
```c++
__global__ void getABCoord(
    const unsigned natA,
    const unsigned natB,
    const rationalSwitchParameters switchingParameters,
    const float *coordinatesA,
    const float *coordinatesB,
    const unsigned *trueIndexes, float *ncoordOut,
    float *devOut, float *virialOut)
```
So that we pass the parameters of the two groups separately.

we then make the first part of the kernel around this change ()
```c++
  // i is the index of the atoms that will be confronted with all the others
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;

  // blocks are initializated with 'ceil (nat/threads)'
  if (i >= nat) 
    return;
    const unsigned idx = trueIndexes[i];
  /* declaring calculation and output variables
  ...
  */
  for (unsigned j = 0; j < natB; ++j) {
    /*things*/
  }
```
So now each atom of group A will ONLY interact with the atoms in group B, then we rewrote the core of the loop, discarding the "triangular things" of before:
```c++
  for (unsigned j = 0; j < natB; ++j) {
    // Safeguard against self interaction
    if (idx == trueIndexes[j])
      continue;
    
    d[0] = coordinatesB[3*j] - x;
    d[1] = coordinatesB[3*j+1] - y;
    d[2] = coordinatesB[3*j+2] - z;
    

    dfunc = 0.;
    coord = calculateSqr(d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
     switchingParameters, dfunc);
    mydevX -= dfunc * d[0];
    mydevY -= dfunc * d[1];
    mydevZ -= dfunc * d[2];
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

``` 
And we end the kernel exacly like before, so no need to specify.

But now we still miss a piece.

The derivatives of the b atoms.

## Calculationg the derivatives of the atoms B

So, what we do?

We might calculate the derivative for the group B along the ones of group A,
but we will end up with needing `size(groupA) * size(groupB)` extra triplette of floats
to be stored during the execution of the first kernel because we can't sum on the same
piece of data unless we sto the parallel calculation to avoid a data race.
And so we need extra memory to store the temporary result and reduce them after.

But in that case we will need `size(groupA) * size(groupB) * 8` bytes of memory for each dimension (that are 3).
That means, for example that for a system of 10000 atoms, in two groups of 5000 will be 200000000 bytes, 
that are roughly 200 MB, but since the relatioship is quadratic and the memory on a gpu is usually scarce,
is not a scalable option.

A "easy" solution that I found is to redo the same loop, but with inverted groups and keeping ONLY
the derivative calculations
This solution as the same big O of the one before and do not cost extra space, apart from the one stored before.


```c++
//We use a similar signature without coordination and virial
__global__ void getABdevB(
    const unsigned natA,
    const unsigned natB,
    const rationalSwitchParameters switchingParameters,
    const float *coordinatesA,
    const float *coordinatesB,
    float *devOut) {
  // i is the index of the atoms that will be confronted with all the others
  const unsigned i = threadIdx.x + blockIdx.x * blockDim.x;

  // blocks are initializated with 'ceil (nat/threads)'
  if (i >= natB) 
    return;
  // we try working with less global memory possible
  // so we set up some temporary variables private to this kernel
  const unsigned idx = trueIndexes[i];
  //local results
  float mydevX = 0.0;
  float mydevY = 0.0;
  float mydevZ = 0.0;
  //local calculation aid
  float x = coordinatesB[3*i];
  float y = coordinatesB[3*i+1];
  float z = coordinatesB[3*i+2];
  float d[3];
  float dfunc;
  float coord;
  for (unsigned j = 0; j < natA; ++j) {
    // Safeguard against self interaction
    if (idx == trueIndexes[j])
      continue;
    
    d[0] = coordinatesA[3*j] - x;
    d[1] = coordinatesA[3*j+1] - y;
    d[2] = coordinatesA[3*j+2] - z;
    

    dfunc = 0.;
    coord = calculateSqr(d[0] * d[0] + d[1] * d[1] + d[2] * d[2],
     switchingParameters, dfunc);
    mydevX -= dfunc * d[0];
    mydevY -= dfunc * d[1];
    mydevZ -= dfunc * d[2];
  }
  // updating global memory ONLY at the end, to access to global memory fewer times per kernel
  devOut[X(i)] = mydevX;
  devOut[Y(i)] = mydevY;
  devOut[Z(i)] = mydevZ;
}
```


### Remarks
The problem of this approach is that if there is a great imbalance in numbers between 
the two groups the approach may scale not so efficiently.
But this may be solved by trying to balance how much work each thread will need to do.

I need to test these ideas (WIP):

 - run an atom per block, use threads to work on nat/threads atoms in parallel and reduce the derivatives for each atom in the first kernel and proceed as in the standard idea.
