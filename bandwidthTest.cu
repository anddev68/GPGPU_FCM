/*
	WELCOME TO ようこそ GPGPUの闇へ
	極力，関数にはマクロや構造体に依存しないような形にしていますが，
	一部手抜き工事でそのまま使ってる部分があります．他のソースコードに転用する場合は注意してください．
	@author Hideki.Kano
	@updated 2018/01/24

	# 命名規則について
	## device/__device
	接頭辞が__deviceとつくものに関しては，カーネル関数からのみ呼び出し可能です．
	deviceはカーネル関数です．CPU側からアクセスできます．

	# マクロ定義について
	ひとつのソースコードで複数の手法を取り扱ってる都合上，マクロ定義により切り替えています．
	マクロ定義を有効に利用してください．
	
	# 配列の確保方法について
	GPUに渡せないため，配列は全て一次元配列で確保しています．
	本来はメモリをCPUとGPUで別々に確保し，コピーする処理が必要ですが，
	thrustライブラリを使うと確保やコピーが非常に楽です．
	
	## vi
	横軸に次元pを取り，縦軸にクラスタ中心番号iを設定しています．
	i番目のクラスタのp次元目にアクセスする場合はvi[i*P+p]です．
	```
	v0.x, v0.y ..., 
	v1.x, v1.y ...
	```

	## xk
	横軸に次元pを取り，縦軸にデータ番号kを設定しています．
	k番目のクラスタのp次元目にアクセスするためにはxk[k*P+p]です．
	```
	x0.x, x0.y, ...,
	x1.x, x1.y, ...,
	```


*/

#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared functions common to CUDA Samples
#include <helper_cuda.h>       // helper functions for CUDA error checking and initialization
#include <cuda.h>
#include <memory>
#include <iostream>
#include <cassert>
#include <algorithm>
#include <stdio.h>
#include <time.h>
#include <math.h>
#include <vector>
#include <string>
#include <sstream>
#include <list>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <time.h>
#include <direct.h>
# include <random>
#include <Windows.h>

#define MAX3(a,b,c) ((a<b)? ((b<c)? c: b):  ((a<c)? c: a))
#define CASE break; case
typedef unsigned  int uint;
using namespace std;


//	=====================================================================
//		実行するクラスタリング手法を#defineで切り替える
//	====================================================================
//#define PFCM 1 // 提案手法1 帰属度関数並列化(感覚的には一番優しいやつ)
//#define TPFCM 1 // 提案手法2 温度並列化
//#define TPFCM2 1
#define HYBRID 1  // 提案手法3 ハイブリッド法
//#define DHYBRID 1 
 //#define AUTO_THIGH 1  // 従来手法 Thighを自動的に決定する方法
//#define DA_FCM 1 // 従来手法 DA_FCM法

#define CD 2.0
#define Q 2.0 // q値は全ての方法で2.0を使用
#define THIGH 7.0 // Thighに固定値を利用する場合に使用
#define TMAX 20.0 // TPFCMで用いる最大値
#define G_IT 1000  /* 繰り返し実験を行う回数 */
#define IT 100 /* 最大帰属度更新回数(打ち切る回数) */
#define EPSIRON 0.001 /* 収束判定に用いる許容誤差 */
#define N 512  /* スレッド数*/
#define INIT_RAND_MIN 0.0 /* 初期クラスタ中心の位置 */
#define INIT_RAND_MAX 7.0

//#define IRIS 1
//#define RANDOM2D 1

#ifdef IRIS
#define CLUSTER_NUM 3 /*クラスタ数*/
#define DATA_NUM 150 /*データ数*/
#define DS_FILENAME "data/iris_formatted.txt" /* データセットファイルをしようする場合のファイル名 */
#define P 4 /* 次元数 */
#elif RANDOM2D
#define CLUSTER_NUM 2
#define DATA_NUM 200 // =2^16=66536
#define P 2
#else
#define CLUSTER_NUM 4 /*クラスタ数*/
#define DATA_NUM 4000 /*データ数*/
//#define DS_FILENAME "data/c4k200p2.txt" /* データセットファイルをしようする場合のファイル名 */
#define DS_FILENAME "data/c4k4000.csv" /* データセットファイルをしようする場合のファイル名 */
#define P 2 /* 次元数 */
#endif


/*
	=================================================
	汎用Util系メソッドとクラス
	使いまわしが聞くように分離
	=================================================
*/

void deepcopy(float *src, float *dst, int size){
	for (int i = 0; i < size; i++){
		dst[i] = src[i];
	}
}

void deepcopy(vector<int> *src, vector<int> *dst){
	for (int i = 0; i < src->size(); i++){
		(*dst)[i] = (*src)[i];
	}
}

void swap(float *a, float *b){
	float tmp = *a;
	*a = *b;
	*b = tmp;
}

void sort(float *src, int size){
	for (int i = 0; i < size; i++){
		for (int j = i + 1; j < size; j++){
			if (src[i] > src[j]) {
				swap(&src[i], &src[j]);
			}
		}
	}
}

void zero_clear(float *src, int size){
	for (int i = 0; i < size; i++){
		src[i] = 0;
	}
}

class MyTimer{
	LARGE_INTEGER nFreq, nBefore, nAfter;
	DWORD dwTime;

public:
	MyTimer(){
	}

	void start(){
		memset(&nFreq, 0x00, sizeof nFreq);
		memset(&nBefore, 0x00, sizeof nBefore);
		memset(&nAfter, 0x00, sizeof nAfter);
		dwTime = 0;
		QueryPerformanceFrequency(&nFreq);
		QueryPerformanceCounter(&nBefore);
	}

	void stop(int *ms){
		QueryPerformanceCounter(&nAfter);
		dwTime = (DWORD)((nAfter.QuadPart - nBefore.QuadPart) * 1000 / nFreq.QuadPart);
		*ms = (int)dwTime;
	}
};

/*
	連番フォルダを作る関数
	path_to_dir/[head]1,
	path_to_dir/[head]2,
	path_to_dir/[head]3,
	というように作ります．
	arg0: 接頭
	return: 成功した場合，連番，失敗した場合-1
*/
int make_seq_dir(char path[], char head[], int max = 10000){
	char textbuf[256];
	for (int i = 0; i < max; i++){
		sprintf(textbuf, "%s%s%d", head, path, i);
		if (_mkdir(textbuf) == 0){
			//	フォルダ作成成功
			return i;
		}
	}
	return -1;
}

/*
=================================================
	ここからはソースコード依存性が高い関数
	iSize,kSizeがマクロ定義されてるので分離できません
	一時的に関数化したと思ってください
=================================================
*/

void print_results(float *uik){
	//	帰属先を出力する
	for (int k = 0; k < DATA_NUM; k++){
		float max = 0.0;
		int index = 0;
		for (int j = 0; j < CLUSTER_NUM; j++){
			if (uik[j*DATA_NUM + k] > max){
				max = uik[j*DATA_NUM + k];
				index = j;
			}
		}
		printf("%d ", index);
	}
	printf("\n");
}

void print_vi(FILE *fp, float *vi){
	for (int i = 0; i < CLUSTER_NUM; i++){
		for (int p = 0; p < P; p++){
			fprintf(fp, "%.2f  ", vi[i*P + p]);
		}
	}
	fprintf(fp, "\n");
}

void print_label(FILE *fp, int *label){
	for (int i = 0; i < DATA_NUM; i++){
		fprintf(fp, "%d ", label[i]);
	}
	fprintf(fp, "\n");
}

void print_entropy(FILE *fp, float *entropy){
	for (int i = 0; i < IT; i++){
		fprintf(fp, "%f\n", entropy[i]);
	}
}

//	ただのエイリアスなんですけど一応準備しておく
void print_jfcm(FILE *fp, float *jfcm){
	for (int i = 0; i < IT; i++){
		fprintf(fp, "%f\n", jfcm[i]);
	}
}

void calc_current_entropy(float *uik, float *entropy){
	float ent = 0.0;
	for (int k = 0; k < DATA_NUM; k++){
		for (int i = 0; i < CLUSTER_NUM; i++){
			//fprintf(stdout, "%.3f  ", h_ds[0].uik[i*DATA_NUM+k]);
			float _uik = uik[i*DATA_NUM + k];
			ent += pow(_uik, Q) * (pow(_uik, 1.0 - Q) - 1.0) / (1.0 - Q);
		}
	}
	*entropy = ent;
}

void calc_current_jfcm(float *uik, float *dik, float *jfcm){
	float org = 0.0;
	for (int k = 0; k < DATA_NUM; k++){
		for (int i = 0; i < CLUSTER_NUM; i++){
			float _uik = uik[i*DATA_NUM + k];
			org += pow(_uik, Q) * dik[i*DATA_NUM + k];
		}
	}
	*jfcm = org;
}



/*
	=================================================
	ここからFCM法専用関数
	恐らくそのまま分離できるとは思うが，
	TODO: FCM法で使いまわせるように分離したい
	=================================================
*/

typedef struct{
public:
	float vi[CLUSTER_NUM*P];
	float vi_bak[CLUSTER_NUM*P];
	float uik[DATA_NUM*CLUSTER_NUM];
	float xk[DATA_NUM*P];
	float dik[DATA_NUM*CLUSTER_NUM];
	float Thigh;
	int iterations; /* 帰属度更新回数(トータル) */
	BOOL finished;

	/* 温度並列化でのみ使用 */
	int t_update_cnt; /* 温度更新回数 */
	float Vi_bak[CLUSTER_NUM*P]; /* 前温度でののvi */
	float entropy[IT];
	float jfcm[IT];

}DataFormat;

template <typename T>
class EvalFormat{
private:
	vector<T> data;
public:
	EvalFormat(){
	}
	void add(T value){
		this->data.push_back(value);
	}

	void statics(T *min, T*max, float *ave, float *conv){
		*min = 10000000;
		*max = 0;
		*ave = 0;
		*conv = 0;
		for (auto it = this->data.begin(); it != this->data.end(); it++){
			*min = MIN(*min, *it);
			*max = MAX(*max, *it);
			*ave += *it;
		}
		*ave /= data.size();
		
		for (auto it = this->data.begin(); it != this->data.end(); it++){
			*conv += *it * *it;
		}
		*conv = *conv / data.size() - (*ave * *ave);
	}

};

void VFA(float *T, float Thigh, int k, float D, float Cd = 2.0){
	*T = Thigh * exp(-Cd*pow((float)k, 1.0f / D));
}

void update_vi(float *uik, float *xk, float *vi, int iSize, int kSize, int pSize, float m){
	for (int i = 0; i < iSize; i++){
		//	独立しているため、分母に利用する合計値を出しておく
		float sum_down = 0;
		for (int k = 0; k < kSize; k++){
			sum_down += pow(uik[i*kSize + k], m);
		}
		//	分子を計算する	
		for (int p = 0; p < pSize; p++){
			float sum_up = 0;
			for (int k = 0; k < kSize; k++){
				sum_up += pow(uik[i*kSize + k], m) * xk[k*pSize + p];
			}
			vi[i*pSize + p] = sum_up / sum_down;
		}
	}
}

void update_uik(float *uik, float *dik, int iSize, int kSize, float m){
	for (int i = 0; i < iSize; i++){
		for (int k = 0; k < kSize; k++){
			float sum = 0;
			for (int j = 0; j < iSize; j++){
				sum += pow((float)(dik[i*kSize + k] / dik[j*kSize + k]), float(1.0 / (m - 1.0)));
			}
			uik[i*kSize + k] = 1.0 / sum;
		}
	}
}

void update_uik_with_T(float *uik, float *dik, int iSize, int kSize, float q, float T){
	for (int i = 0; i < iSize; i++){
		for (int k = 0; k < kSize; k++){
			float sum = 0;
			for (int j = 0; j < iSize; j++){
				sum += pow((1.0f - (1.0f / T)*(1.0f - q)*dik[j*kSize + k]), 1.0f / (1.0f - q));
			}
			float up = pow((1.0f - (1.0f / T)*(1.0f - q)*dik[i*kSize + k]), 1.0f / (1.0f - q));
			uik[i*kSize + k] = up / sum;
		}
	}
}

void update_dik(float *dik, float *vi, float *xk, int iSize, int kSize, int pSize){
	for (int i = 0; i < iSize; i++){
		for (int k = 0; k < kSize; k++){
			float sum = 0.0;
			for (int p = 0; p < pSize; p++){
				sum += pow(float(xk[k*pSize + p] - vi[i*pSize + p]), 2.0f);
			}
			//	dik->setValue(k, i, sqrt(sum));
			//dik[k*iSize + i] = sum;
			dik[i*kSize + k] = sum;
		}
	}
}

void calc_convergence(float *vi, float *vi_bak, int iSize, int pSize, float *err){
	float max_error = 0;
	for (int i = 0; i < iSize; i++){
		float sum = 0.0;				//	クラスタ中心の移動量を計算する
		for (int p = 0; p < pSize; p++){
			sum += pow(vi[i*pSize + p] - vi_bak[i*pSize + p], 2.0f);
		}
		max_error = MAX(max_error, sum);	//	最も大きい移動量を判断基準にする
	}
	*err = max_error;
}

int load_dataset(char *filename, float *dst, int xsize, int ysize){
	FILE *fp = fopen(filename, "r");
	if (fp == NULL) return -1;
	for (int k = 0; k < ysize; k++){
		for (int p = 0; p < xsize; p++){
			float tmp;
			fscanf(fp, "%f", &tmp);
			dst[k * xsize + p] = tmp;
		}
	}
	fclose(fp);
	return 0;
}

float __random(float min, float max){
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> score(min, max);
	//return min + (float)(rand() * (max - min) / RAND_MAX);
	return score(mt);
}

void make_random(float *xk, int size, float min, float max){
	for (int i = 0; i<size; i++){
		xk[i] = __random(min, max);
	}
}

float calc_L1k(float *xk, float *vi, int iSize, int kSize, int pSize){
	float sum = 0.0;
	for (int i = 0; i < iSize; i++){
		for (int k = 0; k < kSize; k++){
			float tmp = 0.0;
			for (int p = 0; p < pSize; p++){
				tmp += pow(float(xk[k*pSize + p] - vi[i*pSize + p]), 2.0f);
			}
			sum += sqrt(tmp);
			//sum += tmp;
		}
	}
	return sum / kSize;
}

int compare(int *target, int *sample, int kSize, int iSize){
	//	[0,1,2]の組み合わせの作成用配列と正解パターン
	vector<int> pattern = vector<int>();
	vector<int> good_pattern = vector<int>();
	for (int i = 0; i < iSize; i++){
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
		for (int i = 0; i < kSize; i++){
			//if (2 < sample[i]) return -2;
			int index = pattern[sample[i]];	//	置換する
			if (target[i] != index) error++;	//	誤った分類
		}
		//	誤分類数が少なければ入れ替える
		if (error < min_error){
			min_error = error;
			deepcopy(&pattern, &good_pattern);
		}

	} while (next_permutation(pattern.begin(), pattern.end()));

	//	置換パターンを利用して、インデックスを置換する
	for (int i = 0; i < kSize; i++){
		sample[i] = good_pattern[sample[i]];
	}
	return min_error;
}

void do_labeling(float *uik, int *results, int iSize, int kSize){
	for (int k = 0; k < kSize; k++){
		results[k] = 0;
		float maxValue = uik[0 * kSize + k];
		for (int i = 1; i < iSize; i++){
			if (maxValue < uik[i*kSize + k]){
				maxValue = uik[i*kSize + k];
				results[k] = i;
			}
		}
	}

}

void sort_xyz(float *src){
	//	一旦ベクトルにします
	vector<vector<float>> tmp;
	tmp.resize(CLUSTER_NUM);
	for (int i = 0; i < CLUSTER_NUM; i++){
		tmp[i].resize(P);
		for (int p = 0; p < P; p++){
			tmp[i][p] = src[i*P + p];
		}
	}
	//		x,y,z...軸の順でそれぞれソートする
	//		規定値のものは同じものとしてカウントする
	/*
	sortの中でforを繰り返すとうまくいかない
	for (int p = 0; p < 1; p++)
	for (int p = 1; p < 2; p++)だとうまくいく．
	*/
	stable_sort(tmp.begin(), tmp.end(), [](const vector<float> a, const vector<float> b){
		return a[0] < b[0];
		/*
		if ((b[0] - a[0]) > 1.0) return true;
		else if ((b[1] - a[1]) > 1.0) return true;
		else return false;
		*/
		/*
		for (int p = 0; p < 2; p++){
		//
		if( (b[p]-a[p]) > 1.0) return true;
		}
		return false;
		*/
	});

	//	ソート結果を代入する
	for (int i = 0; i < CLUSTER_NUM; i++){
		for (int p = 0; p < P; p++){
			src[i*P + p] = tmp[i][p];
		}
	}
	print_vi(stdout, src);

}

void sort_label(const float *xk, float *vi, int *label){
	float _vi[CLUSTER_NUM*P];
	deepcopy(vi, _vi, CLUSTER_NUM*P);

	/* viがどのxkと最も近いか調べて，そいつをラベルとする */
	for (int i = 0; i < CLUSTER_NUM; i++){
		float min_distance = 100000;
		int min_k = 1;
		for (int k = 0; k < DATA_NUM; k++){
			float distance = 0.0;
			for (int p = 0; p < P; p++){
				distance += pow(_vi[i*P + p] - xk[k*P + p], 2.0f);
			}
			if (distance < min_distance){
				min_distance = distance;
				min_k = k;
			}
		}
		/* 最も近いラベル(インデックス)を取得する */
		int index = label[min_k];
		/* ラベルの位置にviをコピーする */
		for (int p = 0; p < P; p++){
			vi[index*P + p] = _vi[i*P + p];
		}
	}
}

/*
	2Dデータセットを作成する関数
	@param xk
	@param label 正解ラベル
	@param minValue 取りえる最小値
	@param maxValue 取りえる最大値
	@param xkSize データ数
*/
void make_2d_random(float *xk, int *label, float minValue, float maxValue, int kSize){
	std::random_device rd;
	std::mt19937 mt(rd());
	
	/* x0 */
	float min = minValue;
	float max = minValue + (maxValue - minValue)*0.45;
	std::uniform_real_distribution<float> score(min, max);
	for (int i = 0; i < kSize/2; i++){
		xk[i*2] = score(mt);
		xk[i * 2+1] = score(mt);
		label[i] = 0;
	}

	/* x1 */
	min = (maxValue - minValue)*0.55;
	max = min + (maxValue - minValue)*0.45;
	std::uniform_real_distribution<float> score2(min, max);
	for (int i = kSize/2; i < kSize; i++){
		xk[i * 2] = score2(mt);
		xk[i * 2 + 1] = score2(mt);
		label[i] = 1;
	}
}





__device__ void __device_update_dik(float *dik, float *vi, float *xk, int iSize, int kSize, int pSize){
	for (int i = 0; i < iSize; i++){
		for (int k = 0; k < kSize; k++){
			float sum = 0.0;
			for (int p = 0; p < pSize; p++){
				sum += pow(float(xk[k*pSize + p] - vi[i*pSize + p]), 2.0f);
			}
			//	dik->setValue(k, i, sqrt(sum));
			//dik[k*iSize + i] = sum;
			dik[i*kSize + k] = sum;
		}
	}
}

__device__ void __device_update_uik(float *uik, float *dik, int iSize, int kSize, float m){
	for (int i = 0; i < iSize; i++){
		for (int k = 0; k < kSize; k++){
			float sum = 0;
			for (int j = 0; j < iSize; j++){
				sum += pow((float)(dik[i*kSize + k] / dik[j*kSize + k]), float(1.0 / (m - 1.0)));
			}
			uik[i*kSize + k] = 1.0 / sum;
		}
	}
}

__device__ void __device_update_uik_with_T(float *uik, float *dik, int iSize, int kSize, float q, float T){
	for (int i = 0; i < iSize; i++){
		for (int k = 0; k < kSize; k++){
			float sum = 0;
			for (int j = 0; j < iSize; j++){
				sum += pow((1.0f - (1.0f / T)*(1.0f - q)*dik[j*kSize + k]), 1.0f / (1.0f - q));
			}
			float up = pow((1.0f - (1.0f / T)*(1.0f - q)*dik[i*kSize + k]), 1.0f / (1.0f - q));
			uik[i*kSize + k] = up / sum;
		}
	}
}

__device__  void __device_update_vi(float *uik, float *xk, float *vi, int iSize, int kSize, int pSize, float m){
	for (int i = 0; i < iSize; i++){
		//	独立しているため、分母に利用する合計値を出しておく
		float sum_down = 0;
		for (int k = 0; k < kSize; k++){
			sum_down += pow(uik[i*kSize + k], m);
		}
		//	分子を計算する	
		for (int p = 0; p < pSize; p++){
			float sum_up = 0;
			for (int k = 0; k < kSize; k++){
				sum_up += pow(uik[i*kSize + k], m) * xk[k*pSize + p];
			}
			vi[i*pSize + p] = sum_up / sum_down;
		}
	}
}

__device__ void __device_VFA(float *T, float Thigh, int k, float D, float Cd = 2.0){
	*T = Thigh * exp(-Cd*pow((float)k, 1.0f / D));
}

__device__ void __device_calc_convergence(float *vi, float *vi_bak, int iSize, int pSize, float *err){
	float max_error = 0;
	for (int i = 0; i < iSize; i++){
		float sum = 0.0;				//	クラスタ中心の移動量を計算する
		for (int p = 0; p < pSize; p++){
			sum += pow(vi[i*pSize + p] - vi_bak[i*pSize + p], 2.0f);
		}
		max_error = MAX(max_error, sum);	//	最も大きい移動量を判断基準にする
	}
	*err = max_error;
}

__device__ void __device_deepcopy(float *src, float *dst, int size){
	for (int i = 0; i < size; i++){
		dst[i] = src[i];
	}
}

__device__ void __device_distance(float *distance, float *v1, float *v2, int pSize){
	float total = 0.0;
	for (int p = 0; p < pSize; p++){
			/* v1[p] * v2[p] */
			total += pow(*(v1 + p) - *(v2 + p), 2);
	}
	*distance = total;
}

__global__ void device_pre_FCM(DataFormat*dss){
	int i = threadIdx.x;
	DataFormat *ds = &dss[i];

	//	収束してたらクラスタリングしない
	if (ds->finished == TRUE)
		return;

	//	viのバックアップ
	__device_deepcopy(ds->vi, ds->vi_bak, CLUSTER_NUM*P);

	float T = ds->Thigh;
	ds->iterations++;
	//__device_VFA(&T, ds->Thigh, ds->iterations, P);
	__device_update_dik(ds->dik, ds->vi, ds->xk, CLUSTER_NUM, DATA_NUM, P);
	__device_update_uik_with_T(ds->uik, ds->dik, CLUSTER_NUM, DATA_NUM, Q, T);
	__device_update_vi(ds->uik, ds->xk, ds->vi, CLUSTER_NUM, DATA_NUM, P, Q);

	//	収束判定
	float err;
	__device_calc_convergence(ds->vi, ds->vi_bak, CLUSTER_NUM, P, &err);
	if (err < EPSIRON)
		ds->finished = TRUE;
}

__global__ void device_update_uik_with_T_parallel(float *uik, float *dik, int iSize, int kSize, float q, float T){
	uint tidx = threadIdx.x;
	uint k = blockIdx.x * blockDim.x + tidx;
	if (k < kSize){
		for (int i = 0; i < iSize; i++){
			float sum = 0;
			for (int j = 0; j < iSize; j++){
				sum += pow((1.0f - (1.0f / T)*(1.0f - q)*dik[j*kSize + k]), 1.0f / (1.0f - q));
			}
			float up = pow((1.0f - (1.0f / T)*(1.0f - q)*dik[i*kSize + k]), 1.0f / (1.0f - q));
			uik[i*kSize + k] = up / sum;
		}
	}
}

__global__ void device_update_dik_parallel(float *dik, float *vi, float *xk, int iSize, int kSize, int pSize){
	uint tidx = threadIdx.x;
	uint k = blockIdx.x * blockDim.x + tidx;
	if (k < kSize){
		for (int i = 0; i < iSize; i++){
			float sum = 0.0;
			for (int p = 0; p < pSize; p++){
				sum += pow(float(xk[k*pSize + p] - vi[i*pSize + p]), 2.0f);
			}
			//	dik->setValue(k, i, sqrt(sum));
			//dik[k*iSize + i] = sum;
			dik[i*kSize + k] = sum;
		}
	}
}

__global__ void device_TPFCM(DataFormat *ds){
	int i = threadIdx.x;

	//	クラスタリングしない
	if (ds[i].finished){
		return;
	}

	float err;
	float Thigh = ds[i].Thigh;
	float T;
	__device_VFA(&T, Thigh, ds[i].t_update_cnt, P, CD);

	//	uikを更新する
	__device_update_dik(ds[i].dik, ds[i].vi, ds[i].xk, CLUSTER_NUM, DATA_NUM, P);
	__device_update_uik_with_T(ds[i].uik, ds[i].dik, CLUSTER_NUM, DATA_NUM, Q, T);


	//	viのバックアップを取る
	__device_deepcopy(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM*P);

	//	vi(centroids)を更新する
	__device_update_vi(ds[i].uik, ds[i].xk, ds[i].vi, CLUSTER_NUM, DATA_NUM, P, Q);

	//	クラスタリング回数を増やしておく
	ds[i].iterations++;

	//	同一温度での収束を判定
	//	収束していなければそのままの温度で繰り返す
	__device_calc_convergence(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM, P, &err);
	//err= 0; // 温度を下げる
	if (EPSIRON < err){
		//	温度を下げずに関数を終了
		return;
	}

	//	前の温度との収束を判定
	//	収束していたら終了
	__device_calc_convergence(ds[i].vi, ds[i].Vi_bak, CLUSTER_NUM, P, &err);
	//err = 0; // 終了
	if (err < EPSIRON){
		//	この時点でクラスタリングを終了する
		ds[i].finished = TRUE;
		return;
	}

	//	バックアップ
	//	温度を下げる前のviを保存
	__device_deepcopy(ds[i].vi, ds[i].Vi_bak, CLUSTER_NUM*P);

	//	温度更新
	ds[i].t_update_cnt++;

}


//	=====================================================================
//	 帰属度関数並列化
//	====================================================================
#ifdef  PFCM
int main(){

	/* 評価用変数 */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;
	EvalFormat<float> eval_time;  // 時間だけはfloatで作成
	MyTimer timer;

	/* クラスタリング用の箱を用意する */
	float _Vi_bak[CLUSTER_NUM*P];
	float _vi_bak[CLUSTER_NUM*P];
	int _label[DATA_NUM];
	int _result[DATA_NUM];
	
	thrust::device_vector<float> d_dik(DATA_NUM*CLUSTER_NUM);
	thrust::device_vector<float> d_uik(DATA_NUM*CLUSTER_NUM);
	thrust::device_vector<float> d_xk(DATA_NUM*P);
	thrust::device_vector<float> d_vi(CLUSTER_NUM*P);

	thrust::host_vector<float> h_uik(DATA_NUM*CLUSTER_NUM);
	thrust::host_vector<float> h_xk(DATA_NUM*P);
	thrust::host_vector<float> h_vi(CLUSTER_NUM*P);

	float *_vi = thrust::raw_pointer_cast(h_vi.data());
	float *_xk = thrust::raw_pointer_cast(h_xk.data());
	float *_uik = thrust::raw_pointer_cast(h_uik.data());
	
	/*
	thrust::host_vector<float> h_dik(DATA_NUM*CLUSTER_NUM);
	float *_dik = thrust::raw_pointer_cast(h_dik.data());
	*/

	/* データロード */
	if (load_dataset(DS_FILENAME, _xk,  P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}

	/* 正解ラベル生成 */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}

	/* GPUにデータを渡す */
	d_xk = h_xk;

	/* FCM法を繰り返し行い，誤分類数(err)，帰属度更新回数(it)，実行時間(time)の最小，最大，平均値を求める */
	for (int g_it = 0; g_it < G_IT; g_it++){
		printf("Prosessing %d/%d\n", g_it, G_IT);
		timer.start();

		/* 初期クラスタ中心を決定 */
		make_random(_vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);

		/* 収束するまで繰り返す */
		float T = THIGH;
		float q = Q;
		int it = 0;
		int t_update_count = 0;
		for (; it < 50; it++){
			printf("T=%f Processing... %d/50\n", T, it);

			/* viをコピー */
			d_vi = h_vi;

			/* dikのテーブルを作成してからuikを更新 */
			device_update_dik_parallel << <1, N >> >(
				thrust::raw_pointer_cast(d_dik.data()),
				thrust::raw_pointer_cast(d_vi.data()),
				thrust::raw_pointer_cast(d_xk.data()),
				CLUSTER_NUM, DATA_NUM, P);
			cudaDeviceSynchronize();

			device_update_uik_with_T_parallel << <1, N >> >(
				thrust::raw_pointer_cast(d_uik.data()),
				thrust::raw_pointer_cast(d_dik.data()),
				CLUSTER_NUM, DATA_NUM, q, T);
			cudaDeviceSynchronize();
			h_uik = d_uik;


			//	viのバックアップを取る
			deepcopy(_vi, _vi_bak, CLUSTER_NUM*P);

			//	vi(centroids)を更新する
			update_vi(_uik, _xk, _vi, CLUSTER_NUM, DATA_NUM, P, q);

			//	同一温度での収束を判定
			//	収束していなければそのままの温度で繰り返す
			float err = 0.0;
			calc_convergence(_vi, _vi_bak, CLUSTER_NUM, P, &err);
			if (EPSIRON < err){
				//	温度を下げずにクラスタリングを継続
				continue;
			}

			//	前の温度との収束を判定
			//	収束していたら終了
			calc_convergence(_vi, _Vi_bak, CLUSTER_NUM, P, &err);
			//err = 0; // 終了
			if (err < EPSIRON){
				//	この時点でクラスタリングを終了する
				break;
			}

			//	温度を下げる前のviを保存
			deepcopy(_vi, _Vi_bak, CLUSTER_NUM*P);

			// 収束していなければ温度を下げて繰り返す
			t_update_count++;
			VFA(&T, THIGH, t_update_count, P);
		}

		//	繰り返し回数を記録
		eval_it.add(it);

		//	誤分類数を記録
		do_labeling(_uik, _result, CLUSTER_NUM, DATA_NUM);
		int err = compare(_label, _result, DATA_NUM, CLUSTER_NUM);
		eval_err.add(err);

		//	実行時間を記録
		int ms;
		timer.stop(&ms);
		eval_time.add(ms);

		printf("DONE: Err=%d, Itaration=%d Time=%d[ms]\n", err, it, ms);
	}

	/* 結果の表示 */
	int min, max;
	float ave, cov;
	float fmin, fmax;
	printf("Method: PFCM\n");
	eval_it.statics(&min, &max, &ave, &cov);
	printf("Update Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_err.statics(&min, &max, &ave, &cov);
	printf("Error Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_time.statics(&fmin, &fmax, &ave, &cov);
	printf("Execution Time: {min: %.3f, max: %.3f, ave: %.3f, cov: %.3f}\n", fmin, fmax, ave, cov);
	
	return 0;
}
#endif

//	=====================================================================
//	 温度並列化
//	====================================================================
#ifdef TPFCM
int main(){

	/* 評価用変数 */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;

	/* クラスタリング用の箱を用意する */
	int _label[DATA_NUM];
	int _result[DATA_NUM];
	float _xk[DATA_NUM*P];

	thrust::device_vector<DataFormat> d_ds(N);
	thrust::host_vector<DataFormat> h_ds(N);

#ifdef RANDOM2D
	/* データロードをやめてランダム生成 */
	make_2d_random(_xk, _label, INIT_RAND_MIN, INIT_RAND_MAX, DATA_NUM);
#else
	/* データロード */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}
	/* 正解ラベル生成 */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}
#endif

	/* 箱の初期化 */
	/* TODO: Tjの決定方法 */
	for (int i = 0; i < N; i++){
		h_ds[i].Thigh = pow(TMAX, (i + 1.0f - N / 2.0f) / (N / 2.0f));
		h_ds[i].iterations = 0;
		make_random(h_ds[i].vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
		deepcopy(_xk, h_ds[i].xk, DATA_NUM*P);
		h_ds[i].finished = FALSE;
		h_ds[i].t_update_cnt = 0;
	}

	/* 各スレッドでFCMを実行 */
	for (int it=0; it< IT; it++){
		// device_pre_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
		d_ds = h_ds;
		device_TPFCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
		cudaDeviceSynchronize();
		h_ds = d_ds;

		/* ここでエントロピーと目的関数を求める */
		for (int n = 0; n < N; n++){
			float* _uik = h_ds[n].uik;
			float * _dik = h_ds[n].dik;
			float entropy, jfcm;
			calc_current_jfcm(_uik, _dik, &jfcm);
			calc_current_entropy(_uik, &entropy);
			h_ds[n].entropy[it] = abs(entropy);
			h_ds[n].jfcm[it] = abs(jfcm);
		}

	}

	/* 結果を記録 */
	int best_it = INT_MAX;
	int best_error = INT_MAX;
	FILE *fp_result = fopen("T_it_err.txt", "w");

	for (int n = 0; n < N; n++){
		int it = h_ds[n].iterations;
		do_labeling(h_ds[n].uik, _result, CLUSTER_NUM, DATA_NUM);
		int err = compare(_label, _result, DATA_NUM, CLUSTER_NUM);
		
		//	最良解の誤分類数を取得
		if (it<best_it){
			best_it = it;
			best_error = err;
		}

		//	繰り返し回数を記録
		eval_it.add(it);

		//	誤分類数を記録
		eval_err.add(err);

		//	エントロピーと目的関数の変化を出力
		char path[32];
		FILE *fp;
		sprintf(path, "entropy/%d.txt", n);
		fp = fopen(path, "w");
		print_entropy(fp, h_ds[n].entropy);
		fclose(fp);

		sprintf(path, "jfcm/%d.txt", n);
		fp = fopen(path, "w");
		print_jfcm(fp, h_ds[n].jfcm);
		fclose(fp);

		//	初期温度，繰り返し回数, 誤分類数のペアを出力
		fprintf(fp_result, "%.4f,%d,%d\n", h_ds[n].Thigh, it, err);

	}
	fclose(fp_result);


	/* 結果の表示 */
	int min, max;
	float ave, cov;
	float fmin, fmax;
	printf("Mathod: 温度並列化アニーリング\n");
	eval_it.statics(&min, &max, &ave, &cov);
	printf("Update Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_err.statics(&min, &max, &ave, &cov);
	printf("Error Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	printf("Best It:%d, Best Error:%d\n", best_it, best_error);
	return 0;
}
#endif

//	=====================================================================
//	 温度並列化
// 繰り返しバージョン
//	====================================================================
#ifdef TPFCM2
int main(){

	/* 評価用変数 */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;

	/* クラスタリング用の箱を用意する */
	int _label[DATA_NUM];
	int _result[DATA_NUM];
	float _xk[DATA_NUM*P];

	thrust::device_vector<DataFormat> d_ds(N);
	thrust::host_vector<DataFormat> h_ds(N);

#ifdef RANDOM2D
	/* データロードをやめてランダム生成 */
	make_2d_random(_xk, _label, INIT_RAND_MIN, INIT_RAND_MAX, DATA_NUM);
#else
	/* データロード */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}
	/* 正解ラベル生成 */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}
#endif

	for(int g_it=0; g_it < G_IT; g_it++){
		printf("g_it=%d\n", g_it);
		
		/* 箱の初期化 */
		/* TODO: Tjの決定方法 */
		for (int i = 0; i < N; i++){
			h_ds[i].Thigh = pow(TMAX, (i + 1.0f - N / 2.0f) / (N / 2.0f));
			h_ds[i].iterations = 0;
			make_random(h_ds[i].vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			deepcopy(_xk, h_ds[i].xk, DATA_NUM*P);
			h_ds[i].finished = FALSE;
			h_ds[i].t_update_cnt = 0;
		}

		/* 各スレッドでFCMを実行 */
		d_ds = h_ds;
		for (int it = 0; it< 100; it++){
			// device_pre_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
			device_TPFCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
			cudaDeviceSynchronize();
		}
		h_ds = d_ds;

		/* 結果を記録 */
		int best_it = INT_MAX;
		int best_error = INT_MAX;
		for (int n = 0; n < N; n++){
			int it = h_ds[n].iterations;
			do_labeling(h_ds[n].uik, _result, CLUSTER_NUM, DATA_NUM);
			int err = compare(_label, _result, DATA_NUM, CLUSTER_NUM);

			//	最良解の誤分類数を取得
			if (it<best_it){
				best_it = it;
				best_error = err;
			}
		}

		//	繰り返し回数を記録
		eval_it.add(best_it);

		//	誤分類数を記録
		eval_err.add(best_error);


	}

	/* 結果の表示 */
	int min, max;
	float ave, cov;
	float fmin, fmax;
	printf("Mathod: 温度並列化アニーリング(1000回平均)\n");
	eval_it.statics(&min, &max, &ave, &cov);
	printf("Update Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_err.statics(&min, &max, &ave, &cov);
	printf("Error Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	return 0;
}
#endif

//	=====================================================================
//	 従来手法DA-FCM
//	====================================================================
#ifdef  DA_FCM
int main(){

	/* 評価用変数 */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;
	EvalFormat<float> eval_time;  // 時間だけはfloatで作成

	/* クラスタリング用の箱を用意する */
	float _Vi_bak[CLUSTER_NUM*P];
	float _vi_bak[CLUSTER_NUM*P];
	float _vi[CLUSTER_NUM*P];
	float _uik[DATA_NUM*CLUSTER_NUM];
	float _xk[DATA_NUM*P];
	float _dik[DATA_NUM*CLUSTER_NUM];
	int _label[DATA_NUM];
	int _result[DATA_NUM];
	MyTimer timer;

	/* データロード */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}

	/* 正解ラベル生成 */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}

	/* ハイブリッド法を繰り返し行い，誤分類数(err)，帰属度更新回数(it)，実行時間(time)の最小，最大，平均値を求める */
	for (int g_it = 0; g_it < G_IT; g_it++){
		printf("Prosessing %d/%d\n", g_it, G_IT);
		timer.start();
		
		make_random(_vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);

		float T = THIGH;
		float q = Q;
		int it = 0;
		int t_update_count = 0;
		for (; it < 50; it++){
			printf("T=%f Processing... %d/50\n", T, it);

			/* 帰属度uik計算を並列化して更新する */
			update_dik(_dik, _vi, _xk, CLUSTER_NUM, DATA_NUM, P);
			update_uik_with_T(_uik, _dik, CLUSTER_NUM, DATA_NUM, q, T);

			//	viのバックアップを取る
			deepcopy(_vi, _vi_bak, CLUSTER_NUM*P);

			//	vi(centroids)を更新する
			update_vi(_uik, _xk, _vi, CLUSTER_NUM, DATA_NUM, P, q);

			//	同一温度での収束を判定
			//	収束していなければそのままの温度で繰り返す
			float err = 0.0;
			calc_convergence(_vi, _vi_bak, CLUSTER_NUM, P, &err);
			if (EPSIRON < err){
				//	温度を下げずにクラスタリングを継続
				continue;
			}

			//	前の温度との収束を判定
			//	収束していたら終了
			calc_convergence(_vi, _Vi_bak, CLUSTER_NUM, P, &err);
			//err = 0; // 終了
			if (err < EPSIRON){
				//	この時点でクラスタリングを終了する
				break;
			}

			//	温度を下げる前のviを保存
			deepcopy(_vi, _Vi_bak, CLUSTER_NUM*P);

			// 収束していなければ温度を下げて繰り返す
			t_update_count++;
			VFA(&T, THIGH, t_update_count, P, CD);
		}

		//	繰り返し回数を記録
		eval_it.add(it);

		//	誤分類数を記録
		do_labeling(_uik, _result, CLUSTER_NUM, DATA_NUM);
		int err = compare(_label, _result, DATA_NUM);
		eval_err.add(err);

		//	実行時間を記録
		int ms;
		timer.stop(&ms);
		eval_time.add(ms);

	}


	/* 結果の表示 */
	
	int min, max;
	float ave, cov;
	float fmin, fmax;

	printf("Method: DA-FCM\n");
	eval_it.statics(&min, &max, &ave, &cov);
	printf("Update Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_err.statics(&min, &max, &ave, &cov);
	printf("Error Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_time.statics(&fmin, &fmax, &ave, &cov);
	printf("Execution Time: {min: %.3f, max: %.3f, ave: %.3f, cov: %.3f}\n", fmin, fmax, ave, cov);
	return 0;
}
#endif

//	=====================================================================
//	 ハイブリッド法
//	====================================================================
#ifdef HYBRID
int main(){

	/* 評価用変数 */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;
	EvalFormat<int> eval_phase1_it;
	EvalFormat<float> eval_time;  // 時間だけはfloatで作成

	/* クラスタリング用の箱を用意する */
	float _Vi_bak[CLUSTER_NUM*P];
	float _vi_bak[CLUSTER_NUM*P];
	int _label[DATA_NUM];
	int _result[DATA_NUM];
	thrust::device_vector<DataFormat> d_ds(N);
	thrust::host_vector<DataFormat> h_ds(N);
	MyTimer timer;

	thrust::device_vector<float> d_dik(DATA_NUM*CLUSTER_NUM);
	thrust::device_vector<float> d_uik(DATA_NUM*CLUSTER_NUM);
	thrust::device_vector<float> d_xk(DATA_NUM*P);
	thrust::device_vector<float> d_vi(CLUSTER_NUM*P);

	thrust::host_vector<float> h_uik(DATA_NUM*CLUSTER_NUM);
	thrust::host_vector<float> h_xk(DATA_NUM*P);
	thrust::host_vector<float> h_vi(CLUSTER_NUM*P);

	float *_vi = thrust::raw_pointer_cast(h_vi.data());
	float *_xk = thrust::raw_pointer_cast(h_xk.data());
	float *_uik = thrust::raw_pointer_cast(h_uik.data());

#ifdef RANDOM2D
	/* データロードをやめてランダム生成 */
	make_2d_random(_xk, _label, INIT_RAND_MIN, INIT_RAND_MAX, DATA_NUM);
#else
	/* データロード */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}
	/* 正解ラベル生成 */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}
#endif

	d_xk = h_xk;

	/* ハイブリッド法を繰り返し行い，誤分類数(err)，帰属度更新回数(it)，実行時間(time)の最小，最大，平均値を求める */
	for (int g_it = 0; g_it < G_IT; g_it++){
		printf("Prosessing %d/%d\n", g_it, G_IT);
		timer.start();

		/* T_baseを決定 */
		float Tbase = 0.0;
		for (int i = 0; i < 1000; i++){
			make_random(_vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			float L1k_bar = calc_L1k(_xk, _vi, CLUSTER_NUM, DATA_NUM, P);
			Tbase += CLUSTER_NUM / L1k_bar;
		}
		Tbase = 1000 / Tbase;
		printf("Tbase=%f\n", Tbase);
		

		/* 箱の初期化 */
		/* TODO: Tjの決定方法 */
		for (int i = 0; i < N; i++){
			h_ds[i].Thigh = pow(Tbase, (i + 1.0f - N / 2.0f) / (N / 2.0f)) + Tbase;
			h_ds[i].iterations = 0;
			make_random(h_ds[i].vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			deepcopy(_xk, h_ds[i].xk, DATA_NUM*P);
			h_ds[i].finished = FALSE;
		}

		/* 各スレッドでフェーズ1を実行 */
		for (int i = 0; i <40; i++){
			d_ds = h_ds;
			device_pre_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
			cudaDeviceSynchronize();
			h_ds = d_ds;
		}

		/* 平均値を利用して中心を決定 */
		/* h_ds[0].viを基準値として，最も距離が近いクラスタの平均値を利用することとするけどそれでいいのか？ */
		zero_clear(_vi, CLUSTER_NUM*P);
		for (int n = 0; n < N; n++){
			sort_label(_xk, h_ds[n].vi, _label);
			for (int i = 0; i < CLUSTER_NUM; i++){
				for (int p = 0; p < P; p++){
					_vi[i*P + p] += h_ds[n].vi[i*P + p];
				}
			}
		}
		for (int i = 0; i < CLUSTER_NUM; i++){
			for (int p = 0; p < P; p++){
				_vi[i*P + p] /= N;
			}
		}

		/* ココまでにかかった繰り返し回数(最大値)を計算する */
		int phase1_max_iteration = 0;
		for (int n = 0; n < N; n++){
			printf("%d ", h_ds[n].iterations);
			phase1_max_iteration = MAX(phase1_max_iteration, h_ds[n].iterations);
		}
		printf("\n");
		

		/* フェーズ2を実行 */
		float T = Tbase;
		float q = Q;
		int it = 0;
		int t_update_count = 0;
		for (; it < 50; it++){
			printf("T=%f Processing... %d/50\n", T, it);

			/* viをコピー */
			d_vi = h_vi;

			/* dikのテーブルを作成してからuikを更新 */
			device_update_dik_parallel << <1, N >> >(
				thrust::raw_pointer_cast(d_dik.data()),
				thrust::raw_pointer_cast(d_vi.data()),
				thrust::raw_pointer_cast(d_xk.data()),
				CLUSTER_NUM, DATA_NUM, P);
			cudaDeviceSynchronize();

			device_update_uik_with_T_parallel << <1, N >> >(
				thrust::raw_pointer_cast(d_uik.data()),
				thrust::raw_pointer_cast(d_dik.data()),
				CLUSTER_NUM, DATA_NUM, q, T);
			cudaDeviceSynchronize();
			h_uik = d_uik;

			//	viのバックアップを取る
			deepcopy(_vi, _vi_bak, CLUSTER_NUM*P);

			//	vi(centroids)を更新する
			update_vi(_uik, _xk, _vi, CLUSTER_NUM, DATA_NUM, P, q);

			//	同一温度での収束を判定
			//	収束していなければそのままの温度で繰り返す
			float err = 0.0;
			calc_convergence(_vi, _vi_bak, CLUSTER_NUM, P, &err);
			if (EPSIRON < err){
				//	温度を下げずにクラスタリングを継続
				continue;
			}

			//	前の温度との収束を判定
			//	収束していたら終了
			calc_convergence(_vi, _Vi_bak, CLUSTER_NUM, P, &err);
			//err = 0; // 終了
			if (err < EPSIRON){
				//	この時点でクラスタリングを終了する
				break;
			}

			//	温度を下げる前のviを保存
			deepcopy(_vi, _Vi_bak, CLUSTER_NUM*P);

			// 収束していなければ温度を下げて繰り返す
			t_update_count++;
			VFA(&T, Tbase, t_update_count, P);
		}

		//	繰り返し回数を記録
		eval_it.add(it);

		//	フェーズ1での更新回数を加味して計算しなおす
		phase1_max_iteration += it;
		eval_phase1_it.add(phase1_max_iteration);

		//	誤分類数を記録
		do_labeling(_uik, _result, CLUSTER_NUM, DATA_NUM);
		int err = compare(_label, _result, DATA_NUM, CLUSTER_NUM);
		eval_err.add(err);

		//	実行時間を記録
		int ms;
		timer.stop(&ms);
		eval_time.add(ms);
		printf("DONE: Err=%d, Itaration=%d Time=%d[ms]\n", err, it, ms);

	}


	/* 結果の表示 */
	int min, max;
	float ave, cov;
	float fmin, fmax;

	printf("Mathod: Hybrid\n");
	eval_it.statics(&min, &max, &ave, &cov);
	printf("Update Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_phase1_it.statics(&min, &max, &ave, &cov);
	printf("Update Count(Total): {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_err.statics(&min, &max, &ave, &cov);
	printf("Error Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_time.statics(&fmin, &fmax, &ave, &cov);
	printf("Execution Time: {min: %.3f, max: %.3f, ave: %.3f, cov: %.3f}\n", fmin, fmax, ave, cov);
	return 0;
}
#endif

#ifdef DHYBRID
int main(){

	/* クラスタリング用の箱を用意する */
	float _Vi_bak[CLUSTER_NUM*P];
	float _vi_bak[CLUSTER_NUM*P];
	int _label[DATA_NUM];
	int _result[DATA_NUM];
	thrust::device_vector<DataFormat> d_ds(N);
	thrust::host_vector<DataFormat> h_ds(N);
	MyTimer timer;

	thrust::device_vector<float> d_dik(DATA_NUM*CLUSTER_NUM);
	thrust::device_vector<float> d_uik(DATA_NUM*CLUSTER_NUM);
	thrust::device_vector<float> d_xk(DATA_NUM*P);
	thrust::device_vector<float> d_vi(CLUSTER_NUM*P);

	thrust::host_vector<float> h_uik(DATA_NUM*CLUSTER_NUM);
	thrust::host_vector<float> h_xk(DATA_NUM*P);
	thrust::host_vector<float> h_vi(CLUSTER_NUM*P);

	float *_vi = thrust::raw_pointer_cast(h_vi.data());
	float *_xk = thrust::raw_pointer_cast(h_xk.data());
	float *_uik = thrust::raw_pointer_cast(h_uik.data());

#ifdef RANDOM2D
	/* データロードをやめてランダム生成 */
	make_2d_random(_xk, _label, INIT_RAND_MIN, INIT_RAND_MAX, DATA_NUM);
#else
	/* データロード */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}
	/* 正解ラベル生成 */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}
#endif

	d_xk = h_xk;

	/* ハイブリッド法を繰り返し行い，誤分類数(err)，帰属度更新回数(it)，実行時間(time)の最小，最大，平均値を求める */
	for (int g_it = 0; g_it < G_IT; g_it++){
		printf("Prosessing %d/%d\n", g_it, G_IT);
		timer.start();

		/* T_baseを決定 */
		float Tbase = 0.0;
		for (int i = 0; i < 1000; i++){
			make_random(_vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			float L1k_bar = calc_L1k(_xk, _vi, CLUSTER_NUM, DATA_NUM, P);
			Tbase += CLUSTER_NUM / L1k_bar;
		}
		Tbase = 1000 / Tbase;
		printf("Tbase=%f\n", Tbase);

		/* 箱の初期化 */
		/* TODO: Tjの決定方法 */
		for (int i = 0; i < N; i++){
			h_ds[i].Thigh = pow(Tbase, (i + 1.0f - N / 2.0f) / (N / 2.0f)) + Tbase;
			h_ds[i].iterations = 0;
			make_random(h_ds[i].vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			deepcopy(_xk, h_ds[i].xk, DATA_NUM*P);
			h_ds[i].finished = FALSE;
		}

		/* 各スレッドでフェーズ1を実行 */
		for (int i = 0; i < 40; i++){
			d_ds = h_ds;
			device_pre_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
			cudaDeviceSynchronize();
			h_ds = d_ds;
		}

		/* vjiを出力する */
		/*
		FILE *fp_vji = fopen("vji.txt", "w");
		for (int n = 0; n < N; n++){
			sort_label(_xk, h_ds[n].vi, _label);
			print_vi(fp_vji, h_ds[n].vi);
		}
		fclose(fp_vji);
		*/

		for (int n = 0; n < N; n++){
			sort_label(_xk, h_ds[n].vi, _label);
		}

		/* 平均値を利用して中心を決定 */
		/* h_ds[0].viを基準値として，最も距離が近いクラスタの平均値を利用することとするけどそれでいいのか？ */
		zero_clear(_vi, CLUSTER_NUM*P);
		for (int n = 0; n < N; n++){
			for (int i = 0; i < CLUSTER_NUM; i++){
				for (int p = 0; p < P; p++){
					_vi[i*P + p] += h_ds[n].vi[i*P + p];
				}
			}
		}
		for (int i = 0; i < CLUSTER_NUM; i++){
			for (int p = 0; p < P; p++){
				_vi[i*P + p] /= N;
			}
		}
		
		/* 平均値を出力する */
		char buf[32];
		sprintf(buf, "vi/%d.txt", g_it);
		FILE *fp_vi = fopen(buf, "w");
		print_vi(fp_vi, _vi);
		fclose(fp_vi);
	}

	return 0;
}
#endif


//	=====================================================================
//	 Thigh自動決定法
//	TODO: 実装
//	====================================================================
#ifdef AUTO_THIGH
int main(){

	/* 評価用変数 */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;
	EvalFormat<float> eval_time;  // 時間だけはfloatで作成

	/* クラスタリング用の箱を用意する */
	float _Vi_bak[CLUSTER_NUM*P];
	float _vi_bak[CLUSTER_NUM*P];
	float _vi[CLUSTER_NUM*P];
	float _uik[DATA_NUM*CLUSTER_NUM];
	float _xk[DATA_NUM*P];
	float _dik[DATA_NUM*CLUSTER_NUM];
	int _label[DATA_NUM];
	int _result[DATA_NUM];
	MyTimer timer;

	/* データロード */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}

	/* 正解ラベル生成 */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}

	/* Thigh見積もりを繰り返し行い，誤分類数(err)，帰属度更新回数(it)，実行時間(time)の最小，最大，平均値を求める */
	for (int g_it = 0; g_it < G_IT; g_it++){
		printf("Prosessing %d/%d\n", g_it, G_IT);
		timer.start();

		/* T_baseを決定 */
		float Tbase = 0.0;
		for (int i = 0; i < 1000; i++){
			make_random(_vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			float L1k_bar = calc_L1k(_xk, _vi, CLUSTER_NUM, DATA_NUM, P);
			Tbase += CLUSTER_NUM / L1k_bar;
		}
		Tbase = 1000 / Tbase;
		printf("Tbase=%f\n", Tbase);

		make_random(_vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);

		/* フェーズ2を実行 */
		float T = Tbase;
		float q = Q;
		int it = 0;
		int t_update_count = 0;
		for (; it < 50; it++){
			printf("T=%f Processing... %d/50\n", T, it);

			/* 帰属度uik計算を並列化して更新する */
			update_dik(_dik, _vi, _xk, CLUSTER_NUM, DATA_NUM, P);
			update_uik_with_T(_uik, _dik, CLUSTER_NUM, DATA_NUM, q, T);

			//	viのバックアップを取る
			deepcopy(_vi, _vi_bak, CLUSTER_NUM*P);

			//	vi(centroids)を更新する
			update_vi(_uik, _xk, _vi, CLUSTER_NUM, DATA_NUM, P, q);

			//	同一温度での収束を判定
			//	収束していなければそのままの温度で繰り返す
			float err = 0.0;
			calc_convergence(_vi, _vi_bak, CLUSTER_NUM, P, &err);
			if (EPSIRON < err){
				//	温度を下げずにクラスタリングを継続
				continue;
			}

			//	前の温度との収束を判定
			//	収束していたら終了
			calc_convergence(_vi, _Vi_bak, CLUSTER_NUM, P, &err);
			//err = 0; // 終了
			if (err < EPSIRON){
				//	この時点でクラスタリングを終了する
				break;
			}

			//	温度を下げる前のviを保存
			deepcopy(_vi, _Vi_bak, CLUSTER_NUM*P);

			// 収束していなければ温度を下げて繰り返す
			t_update_count++;
			VFA(&T, Tbase, t_update_count, P);
		}

		//	繰り返し回数を記録
		eval_it.add(it);

		//	誤分類数を記録
		do_labeling(_uik, _result, CLUSTER_NUM, DATA_NUM);
		int err = compare(_label, _result, DATA_NUM);
		eval_err.add(err);

		//	実行時間を記録
		int ms;
		timer.stop(&ms);
		eval_time.add(ms);

		printf("DONE: Err=%d, Itaration=%d Time=%d[ms]\n", err, it, ms);
	}


	/* 結果の表示 */
	int min, max;
	float ave, cov;
	float fmin, fmax;

	printf("Mathod: Hybrid\n");
	eval_it.statics(&min, &max, &ave, &cov);
	printf("Update Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_err.statics(&min, &max, &ave, &cov);
	printf("Error Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_time.statics(&fmin, &fmax, &ave, &cov);
	printf("Execution Time: {min: %.3f, max: %.3f, ave: %.3f, cov: %.3f}\n", fmin, fmax, ave, cov);

	return 0;
}
#endif

