/*
    fcm.cpp
*/

#include "fcm.h"
#include <vector>
#include <algorithm>

using namespace std;

#define IRIS_FILE "data/iris.txt"

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

void __deepcopy(vector<int> *src, vector<int> *dst){
	for (int i = 0; i < src->size(); i++){
		(*dst)[i] = (*src)[i];
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

int make_iris_datasets(float *xk, int kSize, int pSize){
	if (kSize != 150) return -1;
	if (pSize != 4) return -1;
	FILE *fp = fopen(IRIS_FILE, "r");
	for (int k = 0; k < kSize; k++){
		for (int p = 0; p < pSize; p++){
			float tmp;
			fscanf(fp, "%f", &tmp);
			xk[k * pSize + p] = tmp;
		}
		char buf[32];
		fscanf(fp, "%s", buf);
	}
	fclose(fp);
	return 0;
}

void make_first_centroids(float *vi, int size, float min, float max){
    make_datasets(vi, size, min, max);
}

void make_iris_150_targes(int *targets){
    for (int i = 0; i < 50; i++) targets[i] = 0;
	for (int i = 50; i < 100; i++) targets[i] = 1;
	for (int i = 100; i < 150; i++) targets[i] = 2;
}

int compare(int *target, int *sample, int size){
	//	[0,1,2]の組み合わせの作成用配列と正解パターン
	vector<int> pattern = vector<int>();
	vector<int> good_pattern = vector<int>();
	for (int i = 0; i < 3; i++){
		pattern.push_back(i);
		good_pattern.push_back(0);
	}

	//	エラー最小値
	int min_error = INT_MAX;

	//	すべての置換パターンでマッチング
	do{
		//	エラー数
		int error = 0;
		//	すべてのデータについて、
		for (int i = 0; i < size; i++){
			if (2 < sample[i]) return -2;
			int index = pattern[sample[i]];	//	置換する
			if (target[i] != index) error++;	//	誤った分類
		}
		//	誤分類数が少なければ入れ替える
		if (error < min_error){
			min_error = error;
			__deepcopy(&pattern, &good_pattern);
		}

	} while (next_permutation(pattern.begin(), pattern.end()));

	//	置換パターンを利用して、インデックスを置換する
	for (int i = 0; i < size; i++){
		sample[i] = good_pattern[sample[i]];
	}
	return min_error;
}

void deepcopy(float *src, float *dst, int size){
	for (int i = 0; i < size; i++){
		dst[i] = src[i];
	}
}