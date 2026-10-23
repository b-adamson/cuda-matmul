NVCC  ?= nvcc
ARCH  ?= -arch=sm_75
FLAGS  = $(ARCH) -std=c++14 -O3 -lineinfo -Xcompiler -Wall

matmul: matmul.cu baseline.cuh
	$(NVCC) $(FLAGS) $< -o $@ -lcublas

run: matmul
	./matmul

clean:
	rm -f matmul
