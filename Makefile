NVCC  ?= nvcc
ARCH  ?= -arch=sm_86          # RTX A6000 is sm_86; 4090/L4 = sm_89, A100 = sm_80
FLAGS  = $(ARCH) -O3 -lineinfo -Xcompiler -Wall

matmul: matmul.cu
	$(NVCC) $(FLAGS) $< -o $@

run: matmul
	./matmul

clean:
	rm -f matmul *.ncu-rep *.nsys-rep
