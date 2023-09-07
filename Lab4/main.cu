// Histogram implemented on device
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdarg.h>
#include <math.h>
#include "hist-equ.h"

PGM_IMG img_obuf_cpu;
PGM_IMG img_obuf_gpu;

int MAX_THREADS_PER_BLOCK;
int MAX_GRID_DIM_X;

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename);
void run_gpu_gray_test(PGM_IMG img_in, char *out_filename);
double check_PSNR(int width, int height);

int main(int argc, char *argv[]){
    PGM_IMG img_ibuf_g;  
    double PSNR;

	if (argc != 4) {
		printf("Run with input file name and output file name for host and device as arguments\n");
		exit(1);
	}
	
    int deviceCount = 0, dev = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    CHECK_CUDA_ERROR(err, 0, NULL);

    cudaSetDevice(dev);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);

    MAX_THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
    MAX_GRID_DIM_X = deviceProp.maxGridSize[0];
    
    printf("Running contrast enhancement for gray-scale images.\n");
    img_ibuf_g = read_pgm(argv[1]);
    run_cpu_gray_test(img_ibuf_g, argv[2]);
    run_gpu_gray_test(img_ibuf_g, argv[3]);

    PSNR = check_PSNR(img_ibuf_g.w, img_ibuf_g.h);
   
    if (!isinf(PSNR)) {
        printf("%sCPU and GPU results differ. Exiting...%s\n", RED, RESET);
        exit(EXIT_FAILURE);
    }
    
    free_pgm(img_ibuf_g);
    free_pgm(img_obuf_cpu);
    free_pgm(img_obuf_gpu);

    cudaDeviceReset();

	return 0;
}

double check_PSNR(int width, int height) {

    double PSNR = 0,  t = 0;

    for (int i=0; i<height; i++) {
        for (int j=0; j<width; j++) {
            t = img_obuf_cpu.img[i * width + j] - img_obuf_gpu.img[i * width+ j];
            PSNR += (double)t*t;
        }
    }

    PSNR /= (double)(width*height);
    PSNR = 10*log10(65536/PSNR);

    return (PSNR);
}

void run_cpu_gray_test(PGM_IMG img_in, char *out_filename)
{
    
    printf("\n\nStarting CPU processing...\n");
    img_obuf_cpu = contrast_enhancement_g(img_in);
    write_pgm(img_obuf_cpu, out_filename);

}


void run_gpu_gray_test(PGM_IMG img_in, char *out_filename)
{

    float milliseconds = 0;

    printf("Starting GPU processing..\n");
    img_obuf_gpu = contrast_enhancement_gpu(img_in);
    write_pgm(img_obuf_gpu, out_filename);

    //cudaDeviceReset();
}

PGM_IMG read_pgm(const char * path){
    FILE * in_file;
    char sbuf[256];
    
    
    PGM_IMG result;
    int v_max;//, i;
    in_file = fopen(path, "r");
    if (in_file == NULL){
        printf("Input file not found!\n");
        exit(1);
    }
    
    fscanf(in_file, "%s", sbuf); /*Skip the magic number*/
    fscanf(in_file, "%d",&result.w);
    fscanf(in_file, "%d",&result.h);
    fscanf(in_file, "%d\n",&v_max);
    printf("Image size: %d x %d\n", result.w, result.h);
    

    cudaHostAlloc((void **)&result.img, result.w * result.h * sizeof(unsigned char), cudaHostAllocDefault);
 
    fread(result.img,sizeof(unsigned char), result.w*result.h, in_file);    
    fclose(in_file);
    
    return result;
}

void write_pgm(PGM_IMG img, const char * path){
    FILE * out_file;
    out_file = fopen(path, "wb");
    fprintf(out_file, "P5\n");
    fprintf(out_file, "%d %d\n255\n",img.w, img.h);
    fwrite(img.img,sizeof(unsigned char), img.w*img.h, out_file);
    fclose(out_file);
}

void free_pgm(PGM_IMG img)
{
    // allocated with cudaHostAlloc
    cudaFreeHost(img.img);
}

