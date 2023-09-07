- Folder Code contains the original, sequential Code
- Folder OMP contains the fastest OMP parallel implementation
- Folder GPU contains the fastest CUDA GPU implementation

To run the OMP executable:
    1. cd OMP
    2. make 
    3. ./nbody_cpu <nBodies>

If you wish to check the code for errors:
    1. Uncomment the -DCHECK flag from the CFLAGS (line 2 in OMP/Makefile)
    2. ./run.sh nbody_cpu <nBodies>

The run.sh bash script compiles the code in the Code folder, runs the 
sequential baseline implementation, stores the body data after the first 
iteration in a file named results_<nBodies>.txt inside the Code folder, if 
the file does not already exist and then runs the OMP implementation, printing
the number of errors, if the actual errors, if such exist.

To run the CUDA executable:
    1. cd CUDA
    2. make
    3. ./nbody_gpu <nBodies>

If you wish to check the code for errors:
    1. Uncomment the -DCHECK flag from the GFLAGS (line 3 in GPU/Makefile)
    2. ./run.sh nbody_gpu <nBodies>

The run.sh script works exactly as described above, but runs the CUDA 
implementation to check for errors.

If you have compiled with -DCHECK, only run the code using the ./run.sh script

