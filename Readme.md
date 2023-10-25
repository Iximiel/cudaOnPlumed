# Rewriting the coordination for plumed in Cuda

Here I am showing how to set up a plug-in that can be LOADed in plumed that is compiled with the cuda compiler

The set up has a 'base module' that contains some helper functions and a "header-only" that contains a few function that I think ease the memory management in cuda along with an implementation of the reduction sum taken from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf

I have aslo a couple of pages in which I list some information that my be useful when writing cuda code.