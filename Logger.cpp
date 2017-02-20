#include "Logger.h"

void fprintf_pair_f(FILE *fp, float *A, float *B, int size, char div){
	for (int i = 0; i < size; i++){
		fprintf(fp, "%f%c%f", A[i], div, B[i]);
		fprintf(fp, "\n");
	}
}


void fprintf_pair_d(FILE *fp, int *A, int *B, int size, char div){
	for (int i = 0; i < size; i++){
		fprintf(fp, "%d%c%d", A[i], div, B[i]);
		fprintf(fp, "\n");
	}
}

void fprintf_pair_df(FILE *fp, int *A, float *B, int size, char div){
	for (int i = 0; i < size; i++){
		fprintf(fp, "%d%c%f", A[i], div, B[i]);
		fprintf(fp, "\n");
	}
}