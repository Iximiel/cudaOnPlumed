include make.inc
#-dc adds relocatable device code 
#-dlto Perform link-time optimization of device code. 
#tested with nvcc with :"Build cuda_11.7.r11.7/compiler.31442593_0"
NVFLAGS   = -dc -dlto 
#NVFLAGS   = -g -dc -dlto 
NVLDFLAGS = -dlto 
NV        = nvcc
soext     = so

OBJS:= Coordination.o cudaHelpers.o

all: CudaCoordination.$(soext)

%.o: %.cu cudaHelpers.cuh Coordination.cuh
	$(NV) $(NVFLAGS) -ccbin $(compile) -o $@  $<

CudaCoordination.$(soext): $(OBJS)
	$(NV) -shared $(NVLDFLAGS) $(link) -o $@ $^

clean:
	@rm -fv CudaCoordination.$(soext) $(OBJS)

check: all
	$(MAKE) -C regtest
	$(MAKE) -C regtest checkfail
