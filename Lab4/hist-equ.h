#ifndef HIST_EQU_COLOR_H
#define HIST_EQU_COLOR_H

#include <stdarg.h>

#define BINS 256

#define RED "\033[0;31m"
#define GREEN   "\033[32m"
#define RESET  "\033[0m"

typedef struct{
    int w;
    int h;
    unsigned char * img;
} PGM_IMG;    


#define CHECK_CUDA_ERROR(err, num, ...) 	if (err != cudaSuccess) {\
											    printf("%s%s in %s at line %d%s\n", RED, cudaGetErrorString( err), __FILE__, __LINE__, RESET);\
											    free_cuda_mem(num, __VA_ARGS__);\
											    cudaDeviceReset();\
											    exit(EXIT_FAILURE);\
                                            }

void free_cuda_mem(int num, ...);
PGM_IMG read_pgm(const char * path);
void write_pgm(PGM_IMG img, const char * path);
void free_pgm(PGM_IMG img);

void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin);

__global__ void histogram_gpu(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin);
__global__ void compute_cdf_gpu(int *lut, int img_size, int nbr_bin);
__global__ void histogram_equalization_gpu(unsigned char *img_out, unsigned char *img_in, int img_size);
__global__ void add_histogram_vectors(int *hist1, int* hist2);

//Contrast enhancement for gray-scale images
PGM_IMG contrast_enhancement_g(PGM_IMG img_in);
PGM_IMG contrast_enhancement_gpu(PGM_IMG img_in); 

#endif
