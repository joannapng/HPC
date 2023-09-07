CCOMPILER=icc
CFLAGS=-std=c99 -fast -fopenmp -D_DEFAULT_SOURCE
LDFLAGS=-lm
CPU_TARGET=nbody_cpu

all: cpu

cpu: $(CPU_TARGET)

$(CPU_TARGET): 
	$(CCOMPILER) $(CFLAGS) nbody.c -o $@ $(LDFLAGS)

clean: 
	rm -f $(CPU_TARGET)
