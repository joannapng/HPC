/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <sys/time.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
float accuracy;

// Color definition for output formatting
#define RED "\033[0;31m"
#define GREEN   "\033[32m"
#define RESET  "\033[0m"

// Filter size should be a constant in to be placed in constant memory
#define MAX_FILTER_SIZE 8192
__constant__ float filter[MAX_FILTER_SIZE];

//  Set block and grid dimensions
#define MAX_THREADS_PER_BLOCK_DIM 32
#define GRID_DIM_X (imageW < MAX_THREADS_PER_BLOCK_DIM) ? 1 : imageW/MAX_THREADS_PER_BLOCK_DIM
#define GRID_DIM_Y (imageW < MAX_THREADS_PER_BLOCK_DIM) ? 1 : imageH/MAX_THREADS_PER_BLOCK_DIM
#define BLOCK_DIM_X (imageW < MAX_THREADS_PER_BLOCK_DIM) ? imageW : MAX_THREADS_PER_BLOCK_DIM
#define BLOCK_DIM_Y (imageH < MAX_THREADS_PER_BLOCK_DIM) ? imageH : MAX_THREADS_PER_BLOCK_DIM

// Macro to free cuda allocated memory //
#define FREE_CUDA_MEMORY  cudaFree(d_Input); \
                          cudaFree(d_Buffer); \
                          cudaFree(d_OutputGPU);


void free_cuda_mem(int num, ...) {
	va_list valist;

	va_start(valist, num);

	for (int i=0; i < num; i++) {
		cudaFree(va_arg(valist, void *));
	}

	va_end(valist);

}

#define CHECK_CUDA_ERROR(err, num, ...) 	if (err != cudaSuccess) {\
											printf("%s%s in %s at line %d%s\n", RED, cudaGetErrorString( err), __FILE__, __LINE__, RESET);\
											free_cuda_mem(num, __VA_ARGS__);\
											cudaDeviceReset();\
											exit(EXIT_FAILURE);\
											}

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionRowCPU(float *h_Dst, float *h_Src, float *h_Filter, 
                       int imageW, int imageH, int filterR) {

	int x, y, k;
						
	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			float sum = 0;

			for (k = -filterR; k <= filterR; k++) {
				int d = x + k;

				if (d >= 0 && d < imageW) {
					sum += h_Src[y * imageW + d] * h_Filter[filterR - k];
				}     

				h_Dst[y * imageW + x] = sum;
			}
		}
	}
}


////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
void convolutionColumnCPU(float *h_Dst, float *h_Src, float *h_Filter,
    			   int imageW, int imageH, int filterR) {

	int x, y, k;
		
	for (y = 0; y < imageH; y++) {
		for (x = 0; x < imageW; x++) {
			float sum = 0;

			for (k = -filterR; k <= filterR; k++) {
				int d = y + k;

				if (d >= 0 && d < imageH) {
					sum += h_Src[d * imageW + x] * h_Filter[filterR - k];
				}   

				h_Dst[y * imageW + x] = sum;
			}
		}
	}
    
}

////////////////////////////////////////////////////////////////////////////////
// Reference row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, 
                       int imageW, int imageH, int filterR) {

	int k;
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;

	float sum = 0;

	for (k = -filterR; k <= filterR; k++) {
		int d = idx + k;

		if (d >= 0 && d < imageW) {
			sum += d_Src[idy * imageW + d] * filter[filterR - k];
		}     
		
	}
	d_Dst[idy * imageW + idx] = sum;
}



////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src,
    			   int imageW, int imageH, int filterR) {

	int k;

	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	int idy = blockIdx.y * blockDim.y + threadIdx.y;
		
	float sum = 0;

	for (k = -filterR; k <= filterR; k++) {
		int d = idy + k;

		if (d >= 0 && d < imageH) {
			sum += d_Src[d * imageW + idx] * filter[filterR - k];
		}   

	}
	d_Dst[idy * imageW + idx] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
	float
	*h_Filter = NULL,
	*h_Input = NULL, *d_Input = NULL, 
	*h_Buffer = NULL, *d_Buffer = NULL,
	*h_OutputCPU = NULL, *d_OutputGPU = NULL, *h_OutputGPU =  NULL;

	int imageW;
	int imageH;
	unsigned int i;
	cudaError_t err;

	struct timeval CPU_start, CPU_end, CPU_row;
	
	printf("Enter filter radius : ");
	scanf("%d", &filter_radius);

	// Ta imageW, imageH ta dinei o xrhsths kai thewroume oti einai isa,
	// dhladh imageW = imageH = N, opou to N to dinei o xrhsths.
	// Gia aplothta thewroume tetragwnikes eikones.  

	printf("Enter image size. Should be a power of two and greater than %d : ", FILTER_LENGTH);
	scanf("%d", &imageW);
	imageH = imageW;

	printf("Enter accuracy : ");
	scanf("%f", &accuracy);

	// Set grid and block dimensions
	dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y, 1);
	dim3 gridDim(GRID_DIM_X, GRID_DIM_Y, 1);

	cudaEvent_t GPU_start, GPU_stop, GPU_row, GPU_input_transfers_stop, GPU_output_transfers_start;
	cudaEventCreate(&GPU_start);
	cudaEventCreate(&GPU_stop);
	cudaEventCreate(&GPU_row);
	cudaEventCreate(&GPU_input_transfers_stop);
	cudaEventCreate(&GPU_output_transfers_start);

	printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
	printf("Allocating and initializing host arrays...\n");
	// Tha htan kalh idea na elegxete kai to apotelesma twn malloc...
	h_Filter    = (float *)malloc(FILTER_LENGTH * sizeof(float));
	h_Input     = (float *)malloc(imageW * imageH * sizeof(float));
	h_Buffer    = (float *)malloc(imageW * imageH * sizeof(float));
	h_OutputCPU = (float *)malloc(imageW * imageH * sizeof(float));
	h_OutputGPU = (float *)malloc(imageW * imageH*sizeof(float));
	// to 'h_Filter' apotelei to filtro me to opoio ginetai to convolution kai
	// arxikopoieitai tuxaia. To 'h_Input' einai h eikona panw sthn opoia ginetai
	// to convolution kai arxikopoieitai kai auth tuxaia.

	srand(200);

	for (i = 0; i < FILTER_LENGTH; i++) {
    	h_Filter[i] = (float)(rand() % 16);
	}

	for (i = 0; i < imageW * imageH; i++) {
    	h_Input[i] = (float)rand() / ((float)RAND_MAX / 255) + (float)rand() / (float)RAND_MAX;
	}


	// To parakatw einai to kommati pou ekteleitai sthn CPU kai me vash auto prepei na ginei h sugrish me thn GPU.
	printf("CPU computation...\n");

	gettimeofday(&CPU_start, NULL);
	convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
	gettimeofday(&CPU_row, NULL);
	convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles
	gettimeofday(&CPU_end, NULL);
	// Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
	// pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

	printf("\n\nAllocating and Initializing device arrays...\n");

	err = cudaMalloc((void**) &d_Input, imageW * imageH * sizeof(float));
	CHECK_CUDA_ERROR(err, 1, d_Input)

	err = cudaMalloc((void**) &d_Buffer, imageW * imageH * sizeof(float));
	CHECK_CUDA_ERROR(err, 2, d_Buffer, d_Input)

	err = cudaMalloc((void**) &d_OutputGPU, imageW * imageH * sizeof(float));
	CHECK_CUDA_ERROR(err, 3, d_Buffer, d_Input, d_OutputGPU)

	printf("GPU Computation...\n");

	cudaEventRecord(GPU_start);
  	
	cudaMemcpy(d_Input, h_Input, imageW*imageH*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpyToSymbol(filter, h_Filter, FILTER_LENGTH*sizeof(float));
	cudaEventRecord(GPU_input_transfers_stop);

	convolutionRowGPU<<<gridDim, blockDim>>>(d_Buffer, d_Input, imageW, imageH, filter_radius);
	err = cudaGetLastError();
	CHECK_CUDA_ERROR(err, 3, d_Buffer, d_Input, d_OutputGPU)
	cudaEventRecord(GPU_row);
	cudaDeviceSynchronize();

	convolutionColumnGPU<<<gridDim, blockDim>>>(d_OutputGPU, d_Buffer, imageW, imageH, filter_radius);
	err = cudaGetLastError();
	CHECK_CUDA_ERROR(err, 3, d_Buffer, d_Input, d_OutputGPU)

	cudaEventRecord(GPU_output_transfers_start);
	cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW*imageH*sizeof(float), cudaMemcpyDeviceToHost);
	
	cudaEventRecord(GPU_stop);
	cudaEventSynchronize(GPU_stop);

	for (int x=0; x <imageW; x++) {
		for (int y=0; y<imageH; y++) {
			if (ABS(h_OutputGPU[y*imageW+x] - h_OutputCPU[y*imageW+x]) > accuracy) {
				printf("\n\n%sCPU and GPU results don't match, exiting...%s\n", RED, RESET);
				FREE_CUDA_MEMORY
				cudaDeviceReset();
				exit(EXIT_FAILURE);
			}
		}
	}


	printf("\n\nCPU Row Convolution time (sec): %lf\n", (double) (CPU_row.tv_usec - CPU_start.tv_usec) / 1000000 + \
			(double) (CPU_row.tv_sec - CPU_start.tv_sec));
	printf("CPU Col Convolution time (sec): %lf\n", (double) (CPU_end.tv_usec - CPU_row.tv_usec) / 1000000 + \
			(double) (CPU_end.tv_sec - CPU_row.tv_sec));
	 
	printf("CPU Total Convolution time (sec): %lf\n", (double) (CPU_end.tv_usec - CPU_start.tv_usec) / 1000000 + \
			(double) (CPU_end.tv_sec - CPU_start.tv_sec));

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, GPU_start, GPU_stop);
	printf("\n\nGPU Execution time: %f\n", milliseconds/1000.0);

	cudaEventElapsedTime(&milliseconds, GPU_start, GPU_input_transfers_stop);
	printf("Input Transfers time: %f\n", milliseconds/1000.0);

	cudaEventElapsedTime(&milliseconds, GPU_input_transfers_stop, GPU_row);
	printf("GPU Row Convolution Execution time: %f\n", milliseconds/1000.0);

	cudaEventElapsedTime(&milliseconds, GPU_row, GPU_output_transfers_start);
	printf("GPU Col Convolution Execution time: %f\n", milliseconds/1000.0);

	cudaEventElapsedTime(&milliseconds, GPU_output_transfers_start, GPU_stop);
	printf("Output Transfers time: %f\n", milliseconds/1000.0);
	printf("\n\n%sTest passed%s\n", GREEN, RESET);
	// free all the allocated memory
	free(h_OutputCPU);
	free(h_OutputGPU);
	free(h_Buffer);
	free(h_Input);
	free(h_Filter);

	FREE_CUDA_MEMORY

	// Do a device reset just in case... Bgalte to sxolio otan ylopoihsete CUDA
	// cudaDeviceReset();

	cudaDeviceReset();	
  	return 0;
}
