CUDA_CC=nvcc
ARGS=-O4
DFLAGS=-G -g

all: Convolution2D_pad Convolution2D_shared

Convolution2D_pad: Convolution2D_padding.cu
	$(CUDA_CC) $(ARGS) $(DFLAGS) $^ -o $@

Convolution2D_shared: Convolution2D_padding_shared.cu
	$(CUDA_CC) $(ARGS) $(DFLAGS) $^ -o $@

run_pad:
	./Convolution2D_pad

run_shared:
	./Convolution2D_shared

clean: 
	rm -f Convolution2D_pad Convolution2D_shared