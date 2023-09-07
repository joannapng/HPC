/*
* This sample implements a separable convolution 
* of a 2D image with an arbitrary filter.
*/

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

unsigned int filter_radius;

#define FILTER_LENGTH 	(2 * filter_radius + 1)
#define ABS(val)  	((val)<0.0 ? (-(val)) : (val))
float accuracy;

// Color definitions for output formatting
#define RED "\033[0;31m"
#define GREEN   "\033[32m"
#define RESET  "\033[0m"

// Set block dimensions
#define BLOCK_DIM_X imageW
#define BLOCK_DIM_Y imageH


// Macro to free cuda allocated memory //
#define FREE_CUDA_MEMORY  cudaFree(d_Filter);\
                          cudaFree(d_Input); \
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
// Macro to check if cudaMalloc returned any errors //
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
__global__ void convolutionRowGPU(float *d_Dst, float *d_Src, float *d_Filter, 
                       int imageW, int imageH, int filterR) {

  	int k;
                      
	float sum = 0;

	for (k = -filterR; k <= filterR; k++) {
		int d = threadIdx.x + k;

		if (d >= 0 && d < imageW) {
			sum += d_Src[threadIdx.y * imageW + d] * d_Filter[filterR - k];
		}     

	}
	d_Dst[threadIdx.y * imageW + threadIdx.x] = sum;
}



////////////////////////////////////////////////////////////////////////////////
// Reference column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnGPU(float *d_Dst, float *d_Src, float *d_Filter,
    			   int imageW, int imageH, int filterR) {

	int k;
		
	float sum = 0;

	for (k = -filterR; k <= filterR; k++) {
		int d = threadIdx.y + k;

		if (d >= 0 && d < imageH) {
			sum += d_Src[d * imageW + threadIdx.x] * d_Filter[filterR - k];
		}   

	}
	d_Dst[threadIdx.y * imageW + threadIdx.x] = sum;
}

////////////////////////////////////////////////////////////////////////////////
// Main program
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv) {
    
	float
	*h_Filter = NULL, *d_Filter = NULL,
	*h_Input = NULL, *d_Input = NULL, 
	*h_Buffer = NULL, *d_Buffer = NULL,
	*h_OutputCPU = NULL, *d_OutputGPU = NULL, *h_OutputGPU =  NULL;

	int imageW;
	int imageH;
	unsigned int i;
	cudaError_t err;
	
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
	dim3 gridDim(1, 1, 1);

	printf("Image Width x Height = %i x %i\n\n", imageW, imageH);
	printf("Allocating and initializing host arrays...\n");
	
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

	convolutionRowCPU(h_Buffer, h_Input, h_Filter, imageW, imageH, filter_radius); // convolution kata grammes
	convolutionColumnCPU(h_OutputCPU, h_Buffer, h_Filter, imageW, imageH, filter_radius); // convolution kata sthles

	// Kanete h sugrish anamesa se GPU kai CPU kai an estw kai kapoio apotelesma xeperna thn akriveia
	// pou exoume orisei, tote exoume sfalma kai mporoume endexomenws na termatisoume to programma mas  

	printf("\n\nAllocating and Initializing device arrays...\n");

	err = cudaMalloc((void**) &d_Filter, FILTER_LENGTH * sizeof(float));
	CHECK_CUDA_ERROR(err, 1, d_Filter)
	
	err = cudaMalloc((void**) &d_Input, imageW * imageH * sizeof(float));
	CHECK_CUDA_ERROR(err, 2, d_Filter, d_Input)
	
	err = cudaMalloc((void**) &d_Buffer, imageW * imageH * sizeof(float));
	CHECK_CUDA_ERROR(err, 3, d_Filter, d_Input, d_Buffer)

	err = cudaMalloc((void**) &d_OutputGPU, imageW * imageH * sizeof(float));
	CHECK_CUDA_ERROR(err, 4, d_Filter, d_Input, d_Buffer, d_OutputGPU)

	cudaMemcpy(d_Filter, h_Filter, FILTER_LENGTH*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_Input, h_Input, imageW*imageH*sizeof(float), cudaMemcpyHostToDevice);
	
	printf("GPU computation...\n");

	convolutionRowGPU<<<gridDim, blockDim>>>(d_Buffer, d_Input, d_Filter, imageW, imageH, filter_radius);
	err = cudaGetLastError();
	CHECK_CUDA_ERROR(err, 4, d_Filter, d_Input, d_Buffer, d_OutputGPU)
	
	cudaDeviceSynchronize();

	convolutionColumnGPU<<<gridDim, blockDim>>>(d_OutputGPU, d_Buffer, d_Filter, imageW, imageH, filter_radius);
	err = cudaGetLastError();
	CHECK_CUDA_ERROR(err, 4, d_Filter, d_Input, d_Buffer, d_OutputGPU)

	cudaMemcpy(h_OutputGPU, d_OutputGPU, imageW*imageH*sizeof(float), cudaMemcpyDeviceToHost);


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
	
	printf("\n\n%sTest passed%s\n", GREEN, RESET);
	// free all the allocated memory
	free(h_OutputCPU);
	free(h_OutputGPU);
	free(h_Buffer);
	free(h_Input);
	free(h_Filter);

	FREE_CUDA_MEMORY

	cudaDeviceReset();	
  	return 0;
}
