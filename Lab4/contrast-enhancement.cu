#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/time.h>
#include "hist-equ.h"

extern int MAX_THREADS_PER_BLOCK;
extern int MAX_GRID_DIM_X;

texture <int, 1, cudaReadModeElementType> hist_tex;
texture <int, 1, cudaReadModeElementType> lut_tex;

void free_cuda_mem(int num, ...) {
	va_list valist;

	va_start(valist, num);

	for (int i=0; i < num; i++) {
		cudaFree(va_arg(valist, void *));
	}

	va_end(valist);

}

#define GRID_DIM (ceil((double)width*height/(double)MAX_THREADS_PER_BLOCK))
#define GRID_DIM_X ((GRID_DIM > MAX_GRID_DIM_X) ? MAX_GRID_DIM_X : GRID_DIM)
#define BLOCK_DIM_X MAX_THREADS_PER_BLOCK

struct timeval cpu_end_malloc, cpu_histogram_end, cpu_histogram_eq_end, cpu_start, cpu_end;
cudaEvent_t GPU_start, gpu_end_malloc, gpu_histogram_end, GPU_end;


void histogram(int * hist_out, unsigned char * img_in, int img_size, int nbr_bin){
    int i;
    for ( i = 0; i < nbr_bin; i ++){
        hist_out[i] = 0;
    }

    for ( i = 0; i < img_size; i ++){
        hist_out[img_in[i]] ++;
    }
}


__global__ void histogram_gpu(int *hist_out, unsigned char *img_in, int img_size, int nbr_bin) {

    int i = threadIdx.x + blockDim.x * blockIdx.x;
    extern __shared__ int private_hist[];
    
    private_hist[threadIdx.x] = 0;
    __syncthreads();

    if ( i < img_size) {
        int val = img_in[i];
        atomicAdd(&(private_hist[val]), 1);
    }

    __syncthreads();

    atomicAdd(&(hist_out[threadIdx.x]), private_hist[threadIdx.x]);

}

__global__ void histogram_equalization_gpu(unsigned char *img_out, unsigned char *img_in, int img_size) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;

    // do not copy to shared, fetch directly from texture memory

    img_out[i] = min(255, tex1Dfetch(lut_tex, img_in[i]));

}

__global__ void compute_cdf_gpu(int *lut, int img_size, int nbr_bin) {
    int min = 0, min_index = 0, d;
    __shared__ int cdf[BINS];

    cdf[threadIdx.x] = tex1Dfetch(hist_tex, threadIdx.x);
    __syncthreads();

    while (min == 0) {
        min = cdf[min_index++];
    }

    d = img_size - min;

    for (unsigned int stride = 1; stride < blockDim.x; stride*=2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * 2 * stride - 1;
        if (index < blockDim.x) {
            cdf[index] += cdf[index - stride];
        }
    }

    for (int stride = BINS/4; stride > 0; stride /=2) {
        __syncthreads();
        int index = (threadIdx.x + 1) * stride*2 - 1;
        if (index + stride < BINS) {
            cdf[index + stride] += cdf[index];
        }
    }
    __syncthreads();

    lut[threadIdx.x] = max((int)(((float)cdf[threadIdx.x] - min)*255/d + 0.5), 0);

}

void histogram_equalization(unsigned char * img_out, unsigned char * img_in, 
                            int * hist_in, int img_size, int nbr_bin){
    int *lut = (int *)malloc(sizeof(int)*nbr_bin);
    int i, cdf, min, d;
    /* Construct the LUT by calculating the CDF */
    cdf = 0;
    min = 0;
    i = 0;
    while(min == 0){
        min = hist_in[i++];
    }
    d = img_size - min;
    for(i = 0; i < nbr_bin; i ++){
        cdf += hist_in[i];
        //lut[i] = (cdf - min)*(nbr_bin - 1)/d;
        lut[i] = (int)(((float)cdf - min)*255/d + 0.5);
        if(lut[i] < 0){
            lut[i] = 0;
        }
        
        
    }
    
    /* Get the result image */
    for(i = 0; i < img_size; i ++){
        if(lut[img_in[i]] > 255){
            img_out[i] = 255;
        }
        else{
            img_out[i] = (unsigned char)lut[img_in[i]];
        }
        
    }
}



PGM_IMG contrast_enhancement_g(PGM_IMG img_in)
{
    PGM_IMG result;
    int hist[BINS];
    
    result.w = img_in.w;
    result.h = img_in.h;

    gettimeofday(&cpu_start, NULL);
    result.img = (unsigned char *)malloc(result.w * result.h * sizeof(unsigned char));
    gettimeofday(&cpu_end_malloc, NULL);

    histogram(hist, img_in.img, img_in.h * img_in.w, BINS);
    gettimeofday(&cpu_histogram_end, NULL);

    histogram_equalization(result.img,img_in.img,hist,result.w*result.h, BINS);
    gettimeofday(&cpu_end, NULL);

    printf("CPU Memory Allocation Execution Time: %lf seconds\n", \
    (cpu_end_malloc.tv_sec - cpu_start.tv_sec) + (cpu_end_malloc.tv_usec - cpu_start.tv_usec)/1000000.0);

    printf("CPU Histogram Execution Time: %lf seconds\n", \
    (cpu_histogram_end.tv_sec - cpu_end_malloc.tv_sec) + (cpu_histogram_end.tv_usec - cpu_end_malloc.tv_usec)/1000000.0);

    printf("CPU Histogram Equalization Execution Time: %lf seconds\n", \
    (cpu_end.tv_sec - cpu_histogram_end.tv_sec) + (cpu_end.tv_usec - cpu_histogram_end.tv_usec)/1000000.0);
    
    printf("CPU Total Execution Time: %lf seconds\n\n\n", \
    (cpu_end.tv_sec - cpu_start.tv_sec) + (cpu_end.tv_usec - cpu_start.tv_usec)/1000000.0);

    return result;
}

PGM_IMG contrast_enhancement_gpu(PGM_IMG img_in) 
{
    PGM_IMG result;
    int *hist_gpu, *d_lut; 
    int width, height;
    float milliseconds;
    unsigned char *img_in_gpu_1, *img_in_gpu_2;
    int img_size_1 = (int)ceil((double)img_in.w*img_in.h/2), img_size_2 = img_in.w*img_in.h - img_size_1;

    cudaError_t err;
    size_t offset = 0;

    cudaEventCreate(&GPU_start);
    cudaEventCreate(&gpu_end_malloc);
    cudaEventCreate(&gpu_histogram_end);
    cudaEventCreate(&GPU_end);

    width = result.w = img_in.w;
    height = result.h = img_in.h;
   
    dim3 blockDim(BINS, 1, 1);

    // For the texture memories 
    hist_tex.addressMode[0] = cudaAddressModeBorder;
    hist_tex.addressMode[1] = cudaAddressModeBorder;
    hist_tex.filterMode = cudaFilterModePoint;
    hist_tex.normalized = false;

    lut_tex.addressMode[0] = cudaAddressModeBorder;
    lut_tex.addressMode[1] = cudaAddressModeBorder;
    lut_tex.filterMode = cudaFilterModePoint;
    lut_tex.normalized = false;

    // Create 2 streams
    cudaStream_t stream0, stream1;
    cudaStreamCreate(&stream0);
    cudaStreamCreate(&stream1);

    cudaEventRecord(GPU_start);
    err = cudaMallocManaged((void **)&result.img, width*height*sizeof(unsigned char));
    CHECK_CUDA_ERROR(err, 0, NULL)
    
    // First half of the input image
    err = cudaMalloc((void ** )&img_in_gpu_1, img_size_1*sizeof(unsigned char));
    CHECK_CUDA_ERROR(err, 0, NULL)

    // Second Half of the input image
    cudaMalloc((void ** )&img_in_gpu_2, img_size_2*sizeof(unsigned char));
    CHECK_CUDA_ERROR(err, 1, img_in_gpu_1)

    err = cudaMalloc((void **) &hist_gpu, BINS*sizeof(int));
    CHECK_CUDA_ERROR(err, 2, img_in_gpu_1, img_in_gpu_2)

    err = cudaMemset(hist_gpu, 0, BINS*sizeof(int));
    CHECK_CUDA_ERROR(err, 3, img_in_gpu_1, img_in_gpu_2, hist_gpu)

    err = cudaMalloc((void **) &d_lut, BINS*sizeof(int));
    CHECK_CUDA_ERROR(err, 2, img_in_gpu_1, img_in_gpu_2)

    cudaEventRecord(gpu_end_malloc);

    // Start asynchronous transfers for the two halves of the input image
    // and the two kernels for the computation of the histogram 
    err = cudaMemcpyAsync(img_in_gpu_1, img_in.img, img_size_1, cudaMemcpyHostToDevice, stream0);
    CHECK_CUDA_ERROR(err, 4, img_in_gpu_1, img_in_gpu_2, hist_gpu, d_lut)
    
    dim3 gridDim((ceil((double)img_size_1/(double)BINS)), 1, 1);

    err = cudaMemcpyAsync(img_in_gpu_2, img_in.img+img_size_1, img_size_2, cudaMemcpyHostToDevice, stream1);
    CHECK_CUDA_ERROR(err, 4, img_in_gpu_1, img_in_gpu_2, hist_gpu, d_lut)

    // if the image not a multiple of 2, then we might need a different number of grids
    dim3 gridDim2((ceil((double)img_size_2/(double)BINS)), 1, 1);

    histogram_gpu<<<gridDim, blockDim, BINS*sizeof(int), stream0>>>(hist_gpu, img_in_gpu_1, img_size_1, BINS);
    err = cudaGetLastError();
    CHECK_CUDA_ERROR(err, 4, img_in_gpu_1, img_in_gpu_2, hist_gpu, d_lut)

    cudaDeviceSynchronize();

    histogram_gpu<<<gridDim2, blockDim, BINS*sizeof(int), stream1>>>(hist_gpu, img_in_gpu_2, img_size_2, BINS);
    err = cudaGetLastError();
    CHECK_CUDA_ERROR(err, 4, img_in_gpu_1, img_in_gpu_2, hist_gpu, d_lut)

    cudaEventRecord(gpu_histogram_end);
    cudaDeviceSynchronize();

    // bind the histogram to texture memory
    cudaBindTexture(&offset, hist_tex, hist_gpu, BINS*sizeof(int));

    compute_cdf_gpu<<<1, BINS>>>(d_lut, width*height, BINS);
    err = cudaGetLastError();
    CHECK_CUDA_ERROR(err, 4, img_in_gpu_1, img_in_gpu_2, hist_gpu, d_lut)

    err = cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(err, 4, img_in_gpu_1, img_in_gpu_2, hist_gpu, d_lut)
    
    // bind the lut to texture memory
    cudaBindTexture(&offset, lut_tex, d_lut, BINS*sizeof(int));
    
    histogram_equalization_gpu<<<gridDim, blockDim, BINS*sizeof(int), stream0>>>(result.img, img_in_gpu_1, img_size_1);
    err = cudaGetLastError();
    CHECK_CUDA_ERROR(err, 4, img_in_gpu_1, img_in_gpu_2, hist_gpu, d_lut)

    histogram_equalization_gpu<<<gridDim2, blockDim, BINS*sizeof(int), stream1>>>(result.img+img_size_1, img_in_gpu_2, img_size_2);
    err = cudaGetLastError();
    CHECK_CUDA_ERROR(err, 4, img_in_gpu_1, img_in_gpu_2, hist_gpu, d_lut)

    cudaEventRecord(GPU_end);
    cudaEventSynchronize(GPU_end);

    cudaEventElapsedTime(&milliseconds, GPU_start, gpu_end_malloc);
    printf("GPU Memory Allocation Execution Time: %f\n", milliseconds/1000.0);

    // Input transfers are interleaved with the histogram calculation
    printf("GPU Input Transfers Execution Time: 0.0\n");

    cudaEventElapsedTime(&milliseconds, gpu_end_malloc, gpu_histogram_end);
    printf("GPU Histogram Execution Time: %f\n", milliseconds/1000.0);

    cudaEventElapsedTime(&milliseconds, gpu_histogram_end, GPU_end);
    printf("GPU Histogram Equalization Execution Time: %f\n", milliseconds/1000.0);

    // No output transfers
    printf("GPU Output Transfers Execution Time: 0.0\n");

    cudaEventElapsedTime(&milliseconds, GPU_start, GPU_end);
    printf("GPU Total Execution Time: %f seconds\n", milliseconds/1000.0);

    cudaUnbindTexture(hist_tex);
    cudaUnbindTexture(lut_tex);

    cudaFree(hist_gpu);
    cudaFree(d_lut);
    cudaFree(img_in_gpu_1);
    cudaFree(img_in_gpu_2);

    cudaStreamDestroy(stream0);
    cudaStreamDestroy(stream1);
    
    return result;

}