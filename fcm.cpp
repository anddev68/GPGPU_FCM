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
		    fprintf(fp, "%f ", raw[i*height + k]);
		}
		fprintf(fp, "\n");
	}
}

void __fprint_2df(FILE* fp, float *raw, int width, int height){
	for (int k = 0; k < height; k++){
		for (int i = 0; i <width; i++){
			fprintf(fp, "%f ", raw[k*width + i]);
		}
		fprintf(fp, "\n");
	}
}


void __fprint_2dd_r(FILE *fp, int *raw, int width, int height){
	for (int k = 0; k < height; k++){
		for (int i = 0; i < width; i++){
			fprintf(fp, "%d ", raw[i*height + k]);
		}
		fprintf(fp, "\n");
	}
}

void fprintf_results(FILE *fp, int *results, int size){
	__fprint_2dd_r(fp, results, size, 1);
}

void fprintf_uik(FILE* fp, float *uik, int iSize, int kSize){
    __fprint_2df_r(fp, uik, iSize, kSize);
}

void fprintf_xk(FILE* fp, float *xk, int kSize, int pSize){
	__fprint_2df(fp, xk, pSize, kSize);
}

void make_datasets(float *xk, int size, float min, float max){
    for(int i=0; i<size; i++){
        xk[i] = __random(min, max);
    }
}

void make_sample_sets(float *xk, int size, float min, float max){
    for(int i=0; i<size; i++){
        xk[i] = (abs(max)+abs(min))*i/size + min;
    }
}

void make_first_centroids(float *vi, int size, float min, float max){
    make_datasets(vi, size, min, max);
}


void make_iris_150_targes(int *targets){
    for (int i = 0; i < 50; i++) targets[i] = 0;
	for (int i = 50; i < 100; i++) targets[i] = 1;
	for (int i = 100; i < 150; i++) targets[i] = 2;
}

