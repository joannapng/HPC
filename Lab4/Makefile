CC=nvcc
CFLAGS=-O4 -lm
LDFLAGS=
OBJFILES=contrast-enhancement.o main.o 
TARGET=hist_eq

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJFILES)
	$(CC) $(CFLAGS) $(OBJFILES) -o $@ $(LDFLAGS)

$(OBJFILES) : %.o : %.cu
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJFILES) $(TARGET)
