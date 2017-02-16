/*
    fcm.cpp
*/

#include "fcm.h"


float __random(float min, float max){
	return min + (float)(rand() * (max - min) / RAND_MAX);
}

void __fprint_2df_r(FILE* fp, float *raw, int width, int height){
    for (int k = 0; k < height; k++){
		for (int i = 0; i < width; i++){
		    fprintf(fp2, "%f ", h_ds[n].uik[i*height + k]);
		}
		fprintf(fp2, "\n");
	}
}




void fprintf_uik(FILE* fp, float *uik, int iSize, int kSize){
    __fprint_2df_r(fp, uik, iSize, kSize);
}

void make_datasets(float *xk, int size, float min=0.0, float max=1.0){
    for(int i=0; i<size; i++){
        xk[i] = __random(min, max);
    }
}

void make_sample_sets(float *xk, int size, float min=-1.0, float max=1.0){
    for(int i=0; i<size; i++){
        xk[i] = (abs(max)+abs(min))*i/size + min;
    }
}

void make_first_centroids(float *vi, int size, float min=0.0, float max=1.0){
    make_datasets(vi, size, min, max);
}


void make_iris_150_targes(int *targets){
    for (int i = 0; i < 50; i++) targets[i] = 0;
	for (int i = 50; i < 100; i++) targets[i] = 1;
	for (int i = 100; i < 150; i++) targets[i] = 2;
}

