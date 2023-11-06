include make.inc
#-dc adds relocatable device code 
#-dlto Perform link-time optimization of device code. 
#tested with nvcc with :"Build cuda_11.7.r11.7/compiler.31442593_0"
NVFLAGS   = -dc -dlto 
#NVFLAGS   = -g -dc -dlto 
NVLDFLAGS = -dlto 
NV        = nvcc
soext     = so

all: Coordination.$(soext)

%.o: %.cu ndReduction.h cudaHelpers.cuh
	$(NV) $(NVFLAGS) -ccbin $(compile) -o $@  $<

Coordination.$(soext): ndReduction.o Coordination.o
	$(NV) -shared $(NVLDFLAGS) $(link) -o $@ $^

clean:
	@rm -fv Coordination.$(soext) ndReduction.o Coordination.o 

