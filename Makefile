NVCC=nvcc

#DOUBLEPRECISION = -DDOUBLE_PRECISION
all: mobility_cuda.o

mobility_cuda.o: mobility_kernels.cu
	$(NVCC) -O3 -c -std=c++14 $(DOUBLEPRECISION) -Xcompiler "-fPIC -O3"  $^ -o $@

clean:
	rm -f mobility_cuda.o
