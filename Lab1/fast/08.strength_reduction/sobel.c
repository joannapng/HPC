// This will apply the sobel filter and return the PSNR between the golden sobel and the produced sobel
// sobelized image
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <errno.h>

#define SIZE	4096
#define SIZE_SQ 16777216
#define LOG10_65536 4.81647993062
#define INPUT_FILE	"input.grey"
#define OUTPUT_FILE	"output_sobel.grey"
#define GOLDEN_FILE	"golden.grey"


int convolution2D_ver(int posy, int posx, const unsigned char *input);
int convolution2D_hor(int posy, int posx, const unsigned char *input);
double sobel(unsigned char *input, unsigned char *output, unsigned char *golden);
//int convolution2D(int posy, int posx, const unsigned char *input, char operator[][3]);
/* The arrays holding the input image, the output image and the output used *
 * as golden standard. The luminosity (intensity) of each pixel in the      *
 * grayscale image is represented by a value between 0 and 255 (an unsigned *
 * character). The arrays (and the files) contain these values in row-major *
 * order (element after element within each row and row after row. 			*/
unsigned char input[SIZE*SIZE], output[SIZE*SIZE], golden[SIZE*SIZE];


/* Implement a 2D convolution of the matrix with the operator */
/* posy and posx correspond to the vertical and horizontal disposition of the *
 * pixel we process in the original image, input is the input image and       *
 * operator the operator we apply (horizontal or vertical). The function ret. *
 * value is the convolution of the operator with the neighboring pixels of the*
 * pixel we process.														  */
inline int convolution2D_ver(int posy, int posx, const unsigned char *input) {
	int res;
  
	res = 0;

	/* Replaced * SIZE (* 4096) with shift right by 12, */
	/* Returns res * res to avoid the costly pow operation in the sobel function */
	/* column #1 */
	res += input[((posy - 1) << 12 ) + posx - 1 ];
	res += input[((posy - 1) << 12 ) + posx ] << 1;
	res += input[((posy - 1) << 12 ) + posx + 1 ];

	/* column #3 */
	res -= input[((posy + 1) << 12 ) + posx - 1 ];
	res -= input[((posy + 1) << 12 ) + posx ] << 1;
	res -= input[((posy + 1) << 12 ) + posx + 1 ]; 

	return(res*res);
}

inline int convolution2D_hor(int posy, int posx, const unsigned char *input) {
	int res;
  
	res = 0;

	/* Replaced * SIZE (* 4096) with shift right by 12 */
	/* Returns res * res to avoid the costly pow operation in the sobel function */

	/* row #1 */
	res += -input[( ( posy - 1) << 12 ) + posx - 1 ]; 
	res += input[( ( posy - 1) << 12 ) + posx + 1 ]; 
	res += -(input[( posy  << 12 ) + posx - 1 ] << 1);
	
	/* row #3 */
	res += input[ ( posy << 12 ) + posx + 1 ] << 1; 
	res += -input[( ( posy + 1)<< 12 ) + posx - 1 ];
	res += input[( ( posy + 1)<< 12 ) + posx + 1 ]; 

	return(res*res);
}


/* The main computational function of the program. The input, output and *
 * golden arguments are pointers to the arrays used to store the input   *
 * image, the output produced by the algorithm and the output used as    *
 * golden standard for the comparisons.									 */
double sobel(unsigned char *input, unsigned char *output, unsigned char *golden)
{
	double PSNR = 0, t;
	int i, j;
	unsigned int p;
	struct timespec  tv1, tv2;
	FILE *f_in, *f_out, *f_golden;

	/* The first and last row of the output array, as well as the first  *
     * and last element of each column are not going to be filled by the *
     * algorithm, therefore make sure to initialize them with 0s.		 */
	memset(output, 0, SIZE*sizeof(unsigned char));
	memset(&output[SIZE*(SIZE-1)], 0, SIZE*sizeof(unsigned char));
	for (i = 1; i < SIZE-1; i++) {
		output[i*SIZE] = 0;
		output[i*SIZE + SIZE - 1] = 0;
	}

	/* Open the input, output, golden files, read the input and golden    *
     * and store them to the corresponding arrays.						  */
	f_in = fopen(INPUT_FILE, "r");
	if (f_in == NULL) {
		printf("File " INPUT_FILE " not found\n");
		exit(1);
	}
  
	f_out = fopen(OUTPUT_FILE, "wb");
	if (f_out == NULL) {
		printf("File " OUTPUT_FILE " could not be created\n");
		fclose(f_in);
		exit(1);
	}  
  
	f_golden = fopen(GOLDEN_FILE, "r");
	if (f_golden == NULL) {
		printf("File " GOLDEN_FILE " not found\n");
		fclose(f_in);
		fclose(f_out);
		exit(1);
	}    

	fread(input, sizeof(unsigned char), SIZE*SIZE, f_in);
	fread(golden, sizeof(unsigned char), SIZE*SIZE, f_golden);
	fclose(f_in);
	fclose(f_golden);
  
	/* This is the main computation. Get the starting time. */
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv1);
	/* For each pixel of the output image */
	for (i=1; i<SIZE-1; i+=1) {
		for (j=1; j<SIZE-1; j+=1 ) {
			/* Apply the sobel filter and calculate the magnitude *
			 * of the derivative.								  */
			p = convolution2D_hor(i, j, input) + 
				convolution2D_ver(i, j, input);
	
			/* Replaced * SIZE (* 4096) with shift right by 12 */
			/* The value of the output pixel is clampled to 255 if the floor of 
			 * the square root of p is greater than 255 or else if p >= 256^2.
			 * Therefore, we do not need to perform the sqrt operation if the output is 
			 * going to be clamped to 255 */
			if (p >= 65536)
				output[(i << 12) + j] = 255;      
			else
				output[(i << 12) + j] = (unsigned char)(int)sqrt(p);
		}
	}
	/* Now run through the output and the golden output to calculate *
	 * the MSE and then the PSNR.									 */
	for (i=1; i<SIZE-1; i++) {
		for ( j=1; j<SIZE-1; j++ ) {
			/* Replaced * SIZE (* 4096) with shift right by 12 */
			/* Replaced pow(, 2) with just a multiplication */
			t = output[(i << 12 ) +j] - golden[(i << 12) +j];
			PSNR += t * t;
		}
	}
  
	PSNR /= (double)(SIZE_SQ);
	/* Replaced division using log properties */
	PSNR = 10*(LOG10_65536 - log10(PSNR));

	/* This is the end of the main computation. Take the end time,  *
	 * calculate the duration of the computation and report it. 	*/
	clock_gettime(CLOCK_MONOTONIC_RAW, &tv2);

	printf ("Total time = %10g seconds\n",
			(double) (tv2.tv_nsec - tv1.tv_nsec) / 1000000000.0 +
			(double) (tv2.tv_sec - tv1.tv_sec));

  
	/* Write the output file */
	fwrite(output, sizeof(unsigned char), SIZE*SIZE, f_out);
	fclose(f_out);
  
	return PSNR;
}


int main(int argc, char* argv[])
{
	double PSNR;
	PSNR = sobel(input, output, golden);
	printf("PSNR of original Sobel and computed Sobel image: %g\n", PSNR);
	printf("A visualization of the sobel filter can be found at " OUTPUT_FILE ", or you can run 'make image' to get the jpg\n");

	return 0;
}

