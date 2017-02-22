#include "PFCM.h"

void __pfcm_fprint_2df(FILE* fp, float *raw, int width, int height){
	for (int k = 0; k < height; k++){
		for (int i = 0; i <width; i++){
			fprintf(fp, "%f ", raw[k*width + i]);
		}
		fprintf(fp, "\n");
	}
}

void __pfcm_fprint_2dd(FILE* fp, int *raw, int width, int height){
	for (int k = 0; k < height; k++){
		for (int i = 0; i <width; i++){
			fprintf(fp, "%d ", raw[k*width + i]);
		}
		fprintf(fp, "\n");
	}
}

void __pfcm_fprint_2dd_over0(FILE* fp, int *raw, int width, int height){
	for (int k = 0; k < height; k++){
		for (int i = 0; i <width; i++){
			if (0 <= raw[k*width + i])
				fprintf(fp, "%d ", raw[k*width + i]);
		}
		fprintf(fp, "\n");
	}
}


/*
エラー遷移配列を出力する
*/
void fprintf_error(FILE *fp, int *v, int size){
	__pfcm_fprint_2dd_over0(fp, v, 1, size);
}


/*
温度遷移配列を出力する
*/
void fprintf_T(FILE *fp, float *v, int size){
	__pfcm_fprint_2df(fp, v, 1, size);
}


void fprintf_objfunc(FILE *fp, float *v, int size){
	__pfcm_fprint_2df(fp, v, 1, size);
}