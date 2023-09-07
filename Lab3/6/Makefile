CUDA_CC=nvcc
ARGS=-O4
DFLAGS=-G -g

all: Convolution2D

Convolution2D: Convolution2D.cu
	$(CUDA_CC) $(ARGS) $(DFLAGS) $^ -o $@

run:
	./Convolution2D

clean: 
	rm -f Convolution2D