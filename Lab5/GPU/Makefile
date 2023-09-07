LDFLAGS=-lm
GCOMPILER=nvcc 
GFLAGS=-arch=sm_37 -I. -ftz=true -DCHECK
GPU_TARGET=nbody_gpu

all: gpu

cpu: $(CPU_TARGET)

gpu: $(GPU_TARGET)

$(GPU_TARGET):
	$(GCOMPILER) $(GFLAGS) nbody.cu -o $@

clean: 
	rm -f $(GPU_TARGET)
