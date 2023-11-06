# How to compile

You must have plumed2 installed, this is tested with 2.8, but since it is not using peculiar plumed calls it should be compatible with newer versions.

I have set up two way for compiling this code:
- the `make` approach is faster when you tinker with the code and you do not need to compile all the time `ndReduction.cu`
- the `mklib` approach is more direct and produces directly the shared object from the `Coordination.cu` file

## make
You should have plumed installed or compiled, and the environment variable `PLUMED_ROOT` defined on your active shell, pointing at the root foder of plumed (usually is something like `PLUMED_ROOT=/plumedprefix/lib/plumed`).

run 
```bash
nvcc-MakeFile.sh
```

this creates a make.in that will be included by the Makefile

after that you can simply call

```bash
make
```

You should get a `Coordination.so` that can be LOADed into your `plumed.dat`

**If you are on a mac you should manually change the `Makefile` for changing the `.so` into a `.dlyb`**

## mklib

In this directory there is also a `nvcc-mklib.sh`, it works like `plumed mklib` but for the nvcc. For example:
```bash
nvcc-mklib.sh Coordination.cu
```

Apart using nvcc the difference from the standard `plumed mklib` is that the `ndReduction.cu` file is **always** compiled along with the file specified in the argument.

This approach automatically changes the shared object extension to `.so` or a `.dlyb` depending on the plumed settings
