/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         seq_kmeans.c  (sequential version)                        */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*                                                                           */
/*   Copyright (C) 2005, Northwestern University                             */
/*   See COPYRIGHT notice in top-level directory.                            */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <immintrin.h>

#include "kmeans.h"

typedef struct dist_t {
    double dist;
    int index;
} dist_t;

__forceinline float hadd_ps( __m128 r4 )
{

    /* r2 is the result of the addition of the input vector and the input vector
     * with the 2 lower and 2 higher floats interchanged */
    const __m128 r2 = _mm_add_ps( r4, _mm_movehl_ps( r4, r4 ) );
	
    /* r1 is the result of the addition of r2 and r2 with the floats in
     * in odd indexed duplicated */
    const __m128 r1 = _mm_add_ss( r2, _mm_movehdup_ps( r2 ) );
	
    /* the  result is in float 0 of vector r1 */
    return _mm_cvtss_f32( r1 );
}

/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
__inline static
float euclid_dist_2(int    numdims,  /* no. dimensions */
                    const float *coord1,   /* [numdims] */
                    const float *coord2)   /* [numdims] */
{
    int i;
    float ans=0.0;
    const float *p1 = coord1;
    const float *p2 = coord2;
    const float *const p1End = p1 + numdims;
    __m128 acc;

    const __m128 a = _mm_load_ps(p1);
    const __m128 b = _mm_load_ps(p2);
    const __m128 c = _mm_sub_ps(a, b);
    acc = _mm_mul_ps(c, c);

    p1+=4;
    p2+=4;

    for (; p1 < p1End; p1+=4, p2+=4) {
    	const __m128 a = _mm_load_ps(p1);
	const __m128 b = _mm_load_ps(p2);
	const __m128 c = _mm_sub_ps(a, b);
	acc = _mm_fmadd_ps(c, c, acc);
    }
 
    ans = hadd_ps(acc);    
    return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         float  *object,      /* [numCoords] */
                         float **clusters)    /* [numClusters][numCoords] */
{
    int   i;
    float dist;

    dist_t min_dist = {DBL_MAX, 0};

    /* find the cluster id that has min distance to object */
    for (i=0; i<numClusters; i++) {
        dist = euclid_dist_2(numCoords, object, clusters[i]);
        /* no need square root */
        if (dist < min_dist.dist) { /* find the min and its array index */
            min_dist.dist = dist;
            min_dist.index  = i;
        }
     
    }
    return(min_dist.index);
}

/*----< seq_kmeans() >-------------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
int seq_kmeans(float **objects,      /* in: [numObjs][numCoords] */
               int     numCoords,    /* no. features */
               int     numObjs,      /* no. objects */
               int     numClusters,  /* no. clusters */
               float   threshold,    /* % objects change membership */
               int    *membership,   /* out: [numObjs] */
               float **clusters)     /* out: [numClusters][numCoords] */

{
    int      i, j, index, loop=0;
    int     *newClusterSize; /* [numClusters]: no. objects assigned in each
                                new cluster */
    float    delta;          /* % of objects change their clusters */
    float  *newClusters;    /* [numClusters][numCoords] */

    /* initialize membership[] */
    for (i=0; i<numObjs; i++) membership[i] = -1;

    /* need to initialize newClusterSize and newClusters[0] to all 0 */
    newClusterSize = (int*) calloc(numClusters, sizeof(int));
    assert(newClusterSize != NULL);

    newClusters = (float*)calloc(numClusters*numCoords, sizeof(float));
    assert(newClusterSize != NULL);

    do {
        delta = 0.0;
        #pragma omp parallel for private(j, index), reduction(+:delta, newClusterSize[:numClusters], \
        newClusters[:numClusters*numCoords]), schedule(auto)
        for (i=0; i<numObjs; i++) {
            /* find the array index of nestest cluster center */
            index = find_nearest_cluster(numClusters, numCoords, objects[i],
                                         clusters);

            /* if membership changes, increase delta by 1 */
            if (membership[i] != index) delta += 1.0;

            /* assign the membership to object i */
            membership[i] = index;

            /* update new cluster center : sum of objects located within */
            newClusterSize[index]++;
            for (j=0; j<numCoords; j++)
                newClusters[index * numCoords + j] += objects[i][j];
        }
 
        /* average the sum and replace old cluster center with newClusters */
        for (i=0; i<numClusters; i++) {
            for (j=0; j<numCoords; j++) {
                if (newClusterSize[i] > 0)
                    clusters[i][j] = newClusters[i * numCoords + j] / newClusterSize[i];
                newClusters[i * numCoords + j] = 0.0;   /* set back to 0 */
            }
            newClusterSize[i] = 0;   /* set back to 0 */
        } 
        delta /= numObjs;
    
    } while (delta > threshold && loop++ < 500);
    //free(newClusters[0]);
    free(newClusters);
    free(newClusterSize);

    return 1;
}

