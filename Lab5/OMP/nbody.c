#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "timer.h"

#define ACCURACY 0.005f
#define SOFTENING 1e-9f  /* Will guard against denormals */
#define ABS(x) (((x) > 0) ? (x) : (-(x)))

typedef struct { float x, y, z, vx, vy, vz; } Body;

int check_results(float *buf, int nBodies) {
  FILE *f;
  char temp[256];
  char size[7];
  char filename[28] = "../Code/results_";
  int errors = 0;

  sprintf(size, "%d", nBodies);
  strcat(filename, size);
  strcat(filename, ".txt");
  
  f = fopen(filename, "r");

  for (int i=0; i<6*nBodies; i++) {
    fgets(temp, 256, f);
    if (ABS(atof(temp)-buf[i]) > ACCURACY) {
      printf("%f %f", atof(temp), buf[i]);
      printf("Error. Exiting\n");
    }
  }
  
  printf("Errors: %d\n", errors);

  return(0);
}

void randomizeBodies(float *data, int n) {
  for (int i = 0; i < n; i++) {
    data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
  }
}

void bodyForce(Body *p, float dt, int n) {

  #pragma omp parallel for 
  for (int i = 0; i < n; i++) { 
    float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

    for (int j = 0; j < n; j++) {
      float dx = p[j].x - p[i].x;
      float dy = p[j].y - p[i].y;
      float dz = p[j].z - p[i].z;
      float distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
      float invDist = 1.0f / sqrtf(distSqr);
      float invDist3 = invDist * invDist * invDist;

      Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
    }

    p[i].vx += dt*Fx; p[i].vy += dt*Fy; p[i].vz += dt*Fz;
  }
}

int main(const int argc, const char** argv) {
  
  int nBodies = 30000;
  if (argc > 1) nBodies = atoi(argv[1]);

  const float dt = 0.01f; // time step
  const int nIters = 13;  // simulation iterations

  int bytes = nBodies*sizeof(Body);
  float *buf = (float*)malloc(bytes);
  Body *p = (Body*)buf;

  randomizeBodies(buf, 6*nBodies); // Init pos / vel data

  double totalTime = 0.0;

  for (int iter = 1; iter <= nIters; iter++) {
    StartTimer();

    bodyForce(p, dt, nBodies); // compute interbody forces

    for (int i = 0 ; i < nBodies; i++) { // integrate position
      p[i].x += p[i].vx*dt;
      p[i].y += p[i].vy*dt;
      p[i].z += p[i].vz*dt;
    }

    const double tElapsed = GetTimer() / 1000.0;
    if (iter > 1) { // First iter is warm up
      totalTime += tElapsed; 
    }
    
    #ifdef CHECK
    if (iter==1) {  
      check_results(buf, nBodies);
    }
    #endif

    printf("Iteration %d: %.3f seconds\n", iter, tElapsed);
  }
  double avgTime = totalTime / (double)(nIters-1); 

  printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

  free(buf);
}
