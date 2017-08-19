
NVCC = nvcc
NVFLAGS = -O5 -keep -Xcompiler -fopenmp -Xcompiler -march=native -Xptxas -warn-spills -Xptxas -Werror --std=c++11 --x cu

KEPLER_FLAGS=-gencode arch=compute_35,code=sm_35 -gencode arch=compute_37,code=sm_37
PASCAL_FLAGS=-gencode arch=compute_60,code=sm_60

all: poly-kepler poly-pascal poly-kepler.s poly-pascal.s

batch.h: batch-gen.pl
	perl batch-gen.pl > batch.h

poly-kepler: poly.cpp batch.h
	$(NVCC) $(NVFLAGS) $(KEPLER_FLAGS) -o poly-kepler poly.cpp

poly-kepler.s: poly-kepler
	cuobjdump -sass poly-kepler > poly-kepler.s

poly-pascal: poly.cpp batch.h
	$(NVCC) $(NVFLAGS) $(PASCAL_FLAGS) -o poly-pascal poly.cpp

poly-pascal.s: poly-pascal
	cuobjdump -sass poly-pascal > poly-pascal.s

clean:
	rm -f *~ poly poly-kepler poly-pascal poly-kepler.s poly-pascal.s batch.h
	rm -f *.o *.cudafe* *.fatbin* *.cubin* *.ptx *.reg* *.i *.ii *.module_id *.hash
