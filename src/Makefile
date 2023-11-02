include make.inc
all: Coordination.so

%.o: %.cu ndReduction.h cudaHelpers.cuh
	$(compile) -o $@  $<

Coordination.so: ndReduction.o Coordination.o
	$(link) -o $@ $^

clean:
	rm Coordination.so ndReduction.o
