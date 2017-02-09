/*
	温度ごとにスレッドを立てて並列処理を行う
*/


// CUDA runtime
#include <cuda_runtime.h>

// includes
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization

#include <cuda.h>

#include <memory>
#include <iostream>
#include <cassert>
#include <algorithm>

#include <stdio.h>

#include <time.h>

#include <vector>

#include "Timer.h"
#include "CpuGpuData.cuh"


#define DATA_TYPE float
#define OK 1
#define NG 0

#define N 32	//	一度に実行するスレッド/温度の数

using namespace std;



/**
	クラスタリング結果をGPUから受け取るための構造体
*/
typedef struct{
	int iterations;
	float q;
	float T;
}Result;


/**
	srcをdstにコピーする
*/
__device__ void gpu_array_copy(DATA_TYPE *src, DATA_TYPE *dst, int len){
	for (int i = 0; i < len; i++){
		dst[i] = src[i];
	}

}


/**
q-FCMでuik-matrixを作成する
CPUと同様に単スレッドで作成する
@param q q値
@param T 温度
@param dik dik-matrix
*/
__device__ void gpu_update_uik(double q, double T, DATA_TYPE *uik, DATA_TYPE* dik, int iLen, int kLen){

	double powered = 1.0 / (1.0 - q);
	DATA_TYPE beta = 1.0 / T;

	for (int i = 0; i < iLen; i++){
		for (int k = 0; k < kLen; k++){

			DATA_TYPE up = pow(double(1.0 - beta * (1.0 - q) * dik[i*kLen + k]), powered);

			DATA_TYPE sum = 0;
			for (int j = 0; j < iLen; j++){
				sum += pow(double(1.0 - beta * (1.0 - q) * dik[j*kLen + k]), double(powered));
			}
			uik[i*kLen + k] = up / sum;
		}
	}
}


/**
q-FCMでviを作成する
cpuと同様に単スレッドで作成する
*/
__device__ void gpu_update_vi(float q, DATA_TYPE *uik, DATA_TYPE *xk, DATA_TYPE *vi, int iLen, int kLen, int pLen){
	for (int i = 0; i < iLen; i++){
		//	独立しているため、分母に利用する合計値を出しておく
		DATA_TYPE sum_down = 0;
		for (int k = 0; k < kLen; k++){
			sum_down += pow(uik[i*kLen + k], q);
		}
		//	分子を計算する	
		for (int p = 0; p < pLen; p++){
			DATA_TYPE sum_up = 0;
			for (int k = 0; k < kLen; k++){
				sum_up += pow(uik[i*kLen + k], q) * xk[p*kLen + k];
			}
			vi[p*iLen + i] = sum_up / sum_down;
		}
	}
}


/**
	dikを作成する
*/
__device__ void gpu_make_dik(DATA_TYPE *dik, DATA_TYPE *vi, DATA_TYPE *xk, int iLen, int kLen, int pLen){
	for (int i = 0; i < iLen; i++){
		for (int k = 0; k < kLen; k++){
			DATA_TYPE sum = 0.0;
			for (int p = 0; p < pLen; p++){
				sum += //(xk[p*kLen + k] - vi[p*iLen + i]);
					pow(DATA_TYPE(xk[p*kLen + k] - vi[p*iLen + i]), (DATA_TYPE)2.0);
			}
			//	dik->setValue(k, i, sqrt(sum));
			dik[i*kLen + k] = sum;
		}
	}
}


/**
	収束判定
	@return 0 収束 それ以外 収束していない
*/
__device__ void gpu_judgement_convergence(DATA_TYPE *vi, DATA_TYPE *vi_bak, int iLen, int pLen, int *result, DATA_TYPE epsiron = 0.001){
	DATA_TYPE max_error = 0;
	for (int i = 0; i < iLen; i++){
		DATA_TYPE sum = 0.0;				//	クラスタ中心の移動量を計算する
		for (int p = 0; p < pLen; p++){
			sum += pow(double(vi[p*iLen + i] - vi_bak[p*iLen + i]), 2.0);
		}
		max_error = MAX(max_error, sum);	//	最も大きい移動量を判断基準にする
	}

	if (max_error < epsiron) *result = OK;	//	クラスタ中心の移動がなくなったら終了
	//else *result = 0;
}



/*
	クラスタリングを行う
*/
__global__ void gpu_clustering(DATA_TYPE *g_dik, DATA_TYPE *g_uik, DATA_TYPE *g_vi, DATA_TYPE *g_vi_bak, DATA_TYPE *g_xk, Result *g_results, float *g_T, int iLen, int kLen, int pLen){

	//	スレッド番号を取得
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	//	q値設定
	const float q = 2.0;

	//	xkはどれでも一緒なので変換処理は必要なし
	//	同時アクセスできない場合は考える
	DATA_TYPE *xk = &g_xk[0];

	//	配列を3次元で確保しているので
	//	アドレスの変換処理を行う
	DATA_TYPE *dik = &g_dik[iLen*kLen*index];
	DATA_TYPE *uik = &g_uik[iLen*kLen*index];
	DATA_TYPE *vi = &g_vi[iLen*pLen*index];
	DATA_TYPE *vi_bak = &g_vi_bak[iLen*pLen*index];
	Result *result_t = &g_results[index];


	//	温度も1次元で確保しているので
	//	スレッド番号から変換処理を行う
	float T = g_T[index];

	//	初期uikを作成する
	gpu_make_dik(dik, vi, xk, iLen, kLen, pLen);
	gpu_update_uik(q, T, uik, dik, iLen, kLen);


	//	収束するまで繰り返し行う
	//int *repeat_num = &tmp2[0];
	int iterations;
	for (iterations = 1; iterations < 8; iterations++){

		//	viのバックアップを取る
		gpu_array_copy(vi, vi_bak, iLen*pLen);

		//	viを更新する
		gpu_update_vi(q, uik, xk, vi, iLen, kLen, pLen);

		//	dikを作成する
		gpu_make_dik(dik, vi, xk, iLen, kLen, pLen);

		//	uikを作成する
		gpu_update_uik(q, T, uik, dik, iLen, kLen);

		//	収束判定
		int result = NG;
		gpu_judgement_convergence(vi, vi_bak, iLen, pLen, &result);
		if (result == OK) break;
	}
	
	result_t->iterations = iterations;
	result_t->q = q;
	result_t->T = T;

}




/*
	配列操作関数
*/
void init_random(){
	srand(time(NULL));
	for (int i = 0; i < 100; i++){
		rand();
	}
}
void fill_random(DATA_TYPE *data, int len, float min, float max){
	//	min-maxでランダムに値を埋める
	for (int i = 0; i < len; i++){
		data[i] = (rand() % 100) * (max - min) / 100.0;
	}
}
void print_float_array(float *data, int width, int divider = 10){
	for (int i = 0; i < width; i++){
		printf("%3.4f ", data[i]);
		if ((i + 1) % divider == 0) printf("\n");
	}
	if ( width % divider !=0 )
		printf("\n");
}
void print_int_array(int *data, int width, int divider = 10){
	for (int i = 0; i < width; i++){
		printf("%d ", data[i]);
		if ((i + 1) % divider == 0) printf("\n");
	}
	if (width % divider != 0)
		printf("\n");
}
void deepcopy_vector(vector<int> *src, vector<int> *dst){
	for (int i = 0; i < src->size(); i++){
		(*dst)[i] = (*src)[i];
	}
}
int get_max(float *data, int size){
	int max = data[0];
	for (int i = 1; i < size; i++) max = MAX(data[i], max);
	return max;
}
int get_min(float *data, int size){
	int min = data[0];
	for (int i = 1; i < size; i++) min = MIN(data[i], min);
	return min;
}
void copy_array(DATA_TYPE *src, DATA_TYPE *dst, int len){
	for (int i = 0; i < len; i++){
		dst[i] = src[i];
	}
}

/*
	irisのデータを読み込む
*/
const int IRIS_P = 4;	//	4次元
const int IRIS_K = 150;	//	150個のデータ
const int IRIS_I = 3;	//	3個のクラスタ
const char IRIS_FILE_NAME[16] = "data/iris.txt";
const char IRIS_CLUSTER_NAME[3][32] = {
	"Iris-setosa",
	"Iris-versicolor",
	"Iris-virginica"
};
void load_iris(float *xk, int *target){
	int kLen = IRIS_K;
	int pLen = IRIS_P;

	FILE *fp = fopen(IRIS_FILE_NAME, "r");
	for (int k = 0; k < kLen; k++){
		for (int p = 0; p < pLen; p++){
			float tmp;
			fscanf(fp, "%f", &tmp);
			xk[p*kLen + k] = tmp;
		}
		char buf[32];
		int index = 0;
		fscanf(fp, "%s", &buf);	//	名前
		for (int i = 1; i < 3; i++){
			if (strcmp(buf, IRIS_CLUSTER_NAME[i]) == 0){
				index = i;
			}
		}
		target[k] = index;
	}
	fclose(fp);
}


/*
	正分類データ＝ターゲットとクラスタリング結果＝サンプルを比較し
	誤った数を計算する
	クラスタ番号は順不同とする
*/
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
			int index = pattern[sample[i]];	//	置換する
			if (target[i] != index) error++;	//	誤った分類
		}
		//	誤分類数が少なければ入れ替える
		if (error < min_error){
			min_error = error;
			deepcopy_vector(&pattern, &good_pattern);
		}
	} while (next_permutation(pattern.begin(), pattern.end()));

	//	置換パターンを利用して、インデックスを置換する
	for (int i = 0; i < size; i++){
		sample[i] = good_pattern[sample[i]];
	}

	return min_error;
}


/**
	uikから帰属しているクラスターの番号を取得する
*/
void belongs(DATA_TYPE *uik, int *sample, int kLen, int iLen){
	for (int k = 0; k < kLen; k++){
		DATA_TYPE maxValue = 0;
		int maxIndex = 0;
		for (int i = 0; i < iLen; i++){
			DATA_TYPE value = uik[i*kLen+k];
			if (maxValue < value){
				maxIndex = i;
				maxValue = value;
			}
		}
		//	もっとも高い帰属度を持つクラスタに変更する
		sample[k] = maxIndex;
	}

}


/**
	温度を設定する
*/
float VFA(float Thigh, int k, float D, float Cd=2.0){
	return Thigh * exp(-Cd * pow(k-1, 1.0/D));
}



//	====================================================================================================================
//	メイン関数
//
//	定数値
//		iLen	クラスタ数
//		kLen	データ数
//		pLen	クラスタとデータの次元
//		N		同時に実行するスレッドの数
//
//
//	CPUとGPUで共有する配列
//	配列は(スレッド数xデータ数x次元)で3次元的に確保し、GPU側でアクセスするアドレスを設定する。
//	xkは変化することはないので、そのまま作成する
//		dik		|| vi-xk ||^2		
//		uik		帰属度関数
//		vi		クラスタ中心
//		vi_bak	収束判定用に利用する一時的な配列
//		xk		データセット
//
//
//
//	====================================================================================================================
int main(void){

	const int iLen = 3;
	const int kLen = 150;
	const int pLen = 4;
	CpuGpuData<DATA_TYPE> dik(iLen*kLen*N);
	CpuGpuData<DATA_TYPE> uik(iLen*kLen*N);
	CpuGpuData<DATA_TYPE> vi(iLen*pLen*N);
	CpuGpuData<DATA_TYPE> vi_bak(iLen*pLen*N);
	CpuGpuData<DATA_TYPE> vi2_bak(iLen*pLen*N);
	CpuGpuData<DATA_TYPE> xk(kLen*pLen);

	//	結果保存用
	int target[kLen] = { 0 };
	int sample[kLen] = { 0 };
	CpuGpuData<Result> results(N);	//	結果保存用

	//	温度配列はCPUで確保する
	CpuGpuData<float> T_array(N);	//	温度配列
	float Thigh_array[N] = { 0 };				//	初期温度

	//	xkはアイリスで設定
	load_iris(xk.m_data, target);

	//	viはランダム値で設定
	init_random();
	fill_random(vi.m_data, vi.m_size, get_min(xk.m_data, xk.m_size), get_max(xk.m_data, xk.m_size));

	//	初期温度Thighを設定する
	//	各スレッドごとに温度を設定する
	for (int i = 0; i < N; i++){
		T_array.m_data[i] = 2.0 + 0.01 * i + 0.01;
		Thigh_array[i] = 2.0 + 0.01 * i + 0.01;
	}


	//	収束するまで繰り返す
	for (int i = 0; i < 10; i++){

		//	gpuで(その温度で）クラスタリングを行う
		gpu_clustering << <1, N >> >(dik.m_data, uik.m_data, vi.m_data, vi_bak.m_data, xk.m_data, results.m_data, T_array.m_data, iLen, kLen, pLen);
		cudaDeviceSynchronize();

		cudaError_t cudaErr = cudaGetLastError();
		if (cudaErr != cudaSuccess){
			printf("%s\n", cudaGetErrorString(cudaErr));
		}

		//	結果の出力
		printf("---------------result--------------\n");
		printf("index\tq\tT\titerations\terror\n");
		for (int j = 0; j < N; j++){
			belongs(&uik.m_data[j*kLen*iLen], sample, kLen, iLen);
			int error = compare(target, sample, kLen);
			printf("%d\t%3.2f\t%3.4f\t%d\t%d\n", j, results.m_data[j].q, results.m_data[j].T, results.m_data[j].iterations, error);
		}

		//	温度を下げる
		for (int j = 0; j < N; j++){
			T_array.m_data[j] = VFA(Thigh_array[j], i + 2, pLen);
		}


	}



	//print_float_array(T_array.m_data, T_array.m_size);










	/*
	for (int i = 0; i < N; i++){
		printf("<Threads No.%d>\n", i);
		belongs(uik.m_data, sample, kLen, iLen);
		int error = compare(target, sample, kLen);
		printf("error=%d\n", error);
		printf("iterations=%d\n", results.m_data[i].iterations);
		printf("q=%f\n", results.m_data[i].q);
		printf("T=%f\n", results.m_data[i].T);
		printf("sample=\n");
		print_int_array(sample, kLen, 25);
		printf("\ntarget=\n");
		print_int_array(target, kLen, 25);
		printf("\n");

	}
	*/



	

	//print_float_array(dik.m_data, dik.m_size);
	//print_float_array(vi_bak.m_data, vi_bak.m_size);



	cudaDeviceReset();
}