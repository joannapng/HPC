#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "timer.h"

#define RED "\033[0;31m"
#define GREEN   "\033[32m"
#define RESET  "\033[0m"

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

#define THREADS_PER_BLOCK 512
#define GRID_DIM (int)(ceil((float)nBodies/THREADS_PER_BLOCK))

#define ACCURACY 0.01f
#define SOFTENING 1e-9f  /* Will guard against denormals */
#define ABS(x) (((x) > 0) ? (x) : (-(x)))

typedef struct Body { float *x, *y, *z, *vx, *vy, *vz;} Body;

int check_results(Body *p, int nBodies) {
  FILE *f;
  char temp[256];
  char size[7];
  char filename[28] = "../Code/results_";
  int errors = 0;
  sprintf(size, "%d", nBodies);
  strcat(filename, size);
  strcat(filename, ".txt");
  
  f = fopen(filename, "r");

  for (int i=0; i<nBodies; i++) {
    fgets(temp, 256, f);
    if (ABS(atof(temp) - p->x[i]) > ACCURACY) {
      printf("%f %f\n", atof(temp), p->x[i]);
      printf("Error. Exiting\n");
      errors++;
    }
    fgets(temp, 256, f);

    if (ABS(atof(temp) - p->y[i]) > ACCURACY) {
      printf("%f %f\n", atof(temp), p->y[i]);
      printf("Error. Exiting\n");
      errors++;
    }
    fgets(temp, 256, f);

    if (ABS(atof(temp) - p->z[i]) > ACCURACY) {
      printf("%f %f\n", atof(temp), p->z[i]);
      printf("Error. Exiting\n");
      errors++;
    }

    fgets(temp, 256, f);
    fgets(temp, 256, f);
    fgets(temp, 256, f);

  }

  printf("Errors: %d\n", errors);

  return(0);
}

void randomizeBodies(float *x, float *y, float *z, float *vx, float *vy, float *vz, int n) {

  for (int i=0; i<n; i++) {
    x[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    y[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    z[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;

    vx[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    vy[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    vz[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

__global__ void bodyForce(Body p, float dt, int nBodies, int tiles) {
  __align__(32) __shared__ float priv_x[THREADS_PER_BLOCK];
  __align__(32) __shared__ float priv_y[THREADS_PER_BLOCK];
  __align__(32) __shared__ float priv_z[THREADS_PER_BLOCK];

  float dx, dy, dz, distSqr, invDist3;
  float Fx = 0.0f, Fy=0.0f, Fz=0.0f;

  int tile;
  int id = threadIdx.x + blockIdx.x * blockDim.x;

  float x, y, z, vx, vy, vz;
  
  x = p.x[id];
  y = p.y[id];
  z = p.z[id];
  vx = p.vx[id];
  vy = p.vy[id];
  vz = p.vz[id];

  for (tile=0; tile < tiles-1; tile++) {
    __syncthreads();
    int idx = threadIdx.x + tile * blockDim.x;
    priv_x[threadIdx.x] = p.x[idx];
    priv_y[threadIdx.x] = p.y[idx];
    priv_z[threadIdx.x] = p.z[idx];
    __syncthreads();

    #pragma unroll 8
    for (int j=0; j<blockDim.x; j++) {
      dx = priv_x[j] - x;
      dy = priv_y[j] - y;
      dz = priv_z[j] - z;
      
      distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      invDist3 = __powf(distSqr, -3.0/2.0);

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }
  }

  __syncthreads();
  int idx = threadIdx.x + (tiles-1) * blockDim.x;
  priv_x[threadIdx.x] = p.x[idx];
  priv_y[threadIdx.x] = p.y[idx];
  priv_z[threadIdx.x] = p.z[idx];
  __syncthreads();

  #pragma unroll 8
  for (int k=(tiles-1)*blockDim.x, j=0; k<nBodies; k++, j++) {

    dx = priv_x[j] - x;
    dy = priv_y[j] - y;
    dz = priv_z[j] - z;
    
    distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
    invDist3 = __powf(distSqr, -3.0/2.0);

    Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
  
  }

  vx += dt*Fx; vy += dt*Fy; vz += dt*Fz;

  p.vx[id] = vx;
  p.vy[id] = vy;
  p.vz[id] = vz;

}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 13;  // simulation iterations
  cudaError_t err;
  Body h_body, d_bodies;
  Body *h_bodies = &h_body;

  h_bodies->x = (float *)malloc(nBodies*sizeof(float));
  h_bodies->y = (float *)malloc(nBodies*sizeof(float));
  h_bodies->z = (float *)malloc(nBodies*sizeof(float));
  h_bodies->vx = (float *)malloc(nBodies*sizeof(float));
  h_bodies->vy = (float *)malloc(nBodies*sizeof(float));
  h_bodies->vz = (float *)malloc(nBodies*sizeof(float));

  randomizeBodies(h_bodies->x, h_bodies->y, h_bodies->z, h_bodies->vx, \
                  h_bodies->vy, h_bodies->vz, nBodies); // Init pos / vel data

  err = cudaMalloc((void **)&d_bodies.x, nBodies*sizeof(float));
  CHECK_CUDA_ERROR(err, 0, NULL)

  err = cudaMalloc((void **)&d_bodies.y, nBodies*sizeof(float));
  CHECK_CUDA_ERROR(err, 1, d_bodies.x)
  
  err = cudaMalloc((void **)&d_bodies.z, nBodies*sizeof(float));
  CHECK_CUDA_ERROR(err, 2, d_bodies.x, d_bodies.y)
  
  err = cudaMalloc((void **)&d_bodies.vx, nBodies*sizeof(float));
  CHECK_CUDA_ERROR(err, 3, d_bodies.x, d_bodies.y, d_bodies.z)

  err = cudaMalloc((void **)&d_bodies.vy, nBodies*sizeof(float));
  CHECK_CUDA_ERROR(err, 4, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx)

  err = cudaMalloc((void **)&d_bodies.vz, nBodies*sizeof(float));
  CHECK_CUDA_ERROR(err, 5, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy)

  double totalTime = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {
    StartGPUTimer();

    if (iter == 1) {
      // copy all the data to the device
      err = cudaMemcpy(d_bodies.x, h_bodies->x, nBodies*sizeof(float), cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)
      
      err = cudaMemcpy(d_bodies.y, h_bodies->y, nBodies*sizeof(float), cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)
      
      err = cudaMemcpy(d_bodies.z, h_bodies->z, nBodies*sizeof(float), cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)

      err = cudaMemcpy(d_bodies.vx, h_bodies->vx, nBodies*sizeof(float), cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)
      
      err = cudaMemcpy(d_bodies.vy, h_bodies->vy, nBodies*sizeof(float), cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)

      err = cudaMemcpy(d_bodies.vz, h_bodies->vz, nBodies*sizeof(float), cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)
    }
    else {
      // copy only the position which have been update on the host side
      err = cudaMemcpy(d_bodies.x, h_bodies->x, nBodies*sizeof(float), cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)
      
      err = cudaMemcpy(d_bodies.y, h_bodies->y, nBodies*sizeof(float), cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)

      err = cudaMemcpy(d_bodies.z, h_bodies->z, nBodies*sizeof(float), cudaMemcpyHostToDevice);
      CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)
    }

    bodyForce<<<GRID_DIM, THREADS_PER_BLOCK>>>(d_bodies, dt, nBodies, GRID_DIM); // compute interbody forces
    err = cudaGetLastError();
    CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)

    cudaDeviceSynchronize();    

    // copy the new velocities to the host
    err = cudaMemcpy(h_bodies->vx, d_bodies.vx, nBodies*sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)
    
    err = cudaMemcpy(h_bodies->vy, d_bodies.vy, nBodies*sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)

    err = cudaMemcpy(h_bodies->vz, d_bodies.vz, nBodies*sizeof(float), cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(err, 6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz)

    // update the positions
    for (int i=0; i<nBodies; i++) {
      h_bodies->x[i] += h_bodies->vx[i] * dt;
      h_bodies->y[i] += h_bodies->vy[i] * dt;
      h_bodies->z[i] += h_bodies->vz[i] * dt;
    }

    const float tElapsed = GetGPUTimeElapsed();
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }

    #ifdef CHECK
    if (iter==1) {  
      // check results with golden output
      check_results(h_bodies, nBodies);
    }
    #endif

    printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
  }
  double avgTime = totalTime / (double)(nIters-1); 

  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

  // Free device memory
  free_cuda_mem(6, d_bodies.x, d_bodies.y, d_bodies.z, d_bodies.vx, \
                  d_bodies.vy, d_bodies.vz);

  cudaDeviceReset();

  // Free host memory
  free(h_bodies->x);
  free(h_bodies->y);
  free(h_bodies->z);
  free(h_bodies->vx);
  free(h_bodies->vy);
  free(h_bodies->vz);
}
