/*
並列グラフ探索を利用したバージョン
中央集権型 - プロセッサ間でノードのopen listを共有メモリ上に共有する
http://d.hatena.ne.jp/hanecci/20110205/1296924411
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
#include <vector>
#include <string>
#include <sstream>
#include <list>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Timer.h"
#include "CpuGpuData.cuh"
#include <time.h>

#include "FCM.h"
#include "PFCM.h"
#include "Logger.h"


/*
############################ Warning #####################
GPUプログラミングでは可変長配列を使いたくないため定数値を利用しています。
適宜値を変えること
########################################################
*/

//	IRISのデータを使う場合は#defineすること
#define IRIS

#define MAX3(a,b,c) ((a<b)? ((b<c)? c: b):  ((a<c)? c: a))
#define CASE break; case

#ifdef IRIS
	#define CLUSTER_NUM 3 /*クラスタ数*/
	#define DATA_NUM 150 /*データ数*/
	#define P 4 /* 次元数 */
#else
	#define CLUSTER_NUM 2 /*クラスタ数*/
	#define DATA_NUM 150 /*データ数*/
	#define P 2 /* 次元数 */
#endif

#define TEMP_SCENARIO_NUM 20 /*温度遷移シナリオの数*/
#define ERROR_SCENARIO_NUM 20 /*誤差遷移シナリオの数*/
#define MAX_CLUSTERING_NUM 20 /* 最大繰り返し回数 -> 将来的にシナリオの数にしたい */

#define EPSIRON 0.001 /* 許容エラー*/
#define N 128  /* スレッド数*/



typedef unsigned  int uint;
using namespace std;

/*
デバイスに渡すため/受け取るのデータセット
device_vectorに突っ込む構造体の中はどうやら通常の配列で良いらしい。
その為、可変長配列は使用できない可能性が高い。
FCMではdik, uik...
*/
typedef struct{
	float dik[DATA_NUM*CLUSTER_NUM];
	float uik[DATA_NUM*CLUSTER_NUM];
	float xk[DATA_NUM*P];
	float vi[CLUSTER_NUM*P];
	float vi_bak[CLUSTER_NUM*P];			//同一温度での前のvi
	float Vi_bak[CLUSTER_NUM*P];			//異なる温度での前のvi
	int error[ERROR_SCENARIO_NUM];	//	エラーシナリオ
	float obj_func[MAX_CLUSTERING_NUM]; // 目的関数のシナリオ
	float T[TEMP_SCENARIO_NUM]; //	温度遷移のシナリオ
	int results[DATA_NUM];	//	実行結果
	float q;		//	q値
	int t_pos;		//	温度シナリオ参照位置
	int t_change_num;	//	温度変更回数
	int clustering_num;	//	クラスタリング回数
	BOOL is_finished; //クラスタリング終了条件を満たしたかどうか
}DataSet;

__global__ void device_FCM(DataSet *ds);
__device__ void __device_calc_convergence(float *vi, float *vi_bak, int iSize, int pSize, float *err);
__device__ void __device_VFA(float *, float, int, float, float);
__device__ void __device_update_vi(float *uik, float *xk, float *vi, float iSize, int kSize, int pSize, float m);
__device__ void __device_update_uik(float *, float *, int, int, float);
__device__ void __device_update_uik_with_T(float *uik, float *dik, int iSize, int kSize, float q, float T);
__device__ void __device_distance(float *, float *, float *, int);
__device__ void __device_update_dik(float *dik, float *vi, float *xk, int iSize, int kSize, int pSize);
__device__ void __device_jfcm(float *uik, float *dik, float *jfcm, float m, int iSize, int kSize);
__device__ void __device_jtsallis(float *uik, float *dik, float *jfcm, float q, float T, int iSize, int kSize);
__device__ void __device_eval(float *uik, int *results, int iSize, int kSize);
__device__ void __device_iris_error(float *uik, int *error, int iSize, int kSize);

int main(){
	srand((unsigned)time(NULL));

	/*
		ホストとデバイスのデータ領域を確保する
		DataSetIn, DataSetOutがFCMに用いるデータの集合、構造体なので、子ノード数分確保すればよい
		確保数1にすると並列化を行わず、通常VFA+FCMで行う
	*/
	thrust::device_vector<DataSet> d_ds(N);
	thrust::host_vector<DataSet> h_ds(N);

	/*
		正確な分類
	*/
	int targets[150];
	make_iris_150_targes(targets);
	char buf[32];


	/*
		vectorの初期化
	*/
	for(int i=0; i<N; i++){
		h_ds[i].t_pos = 0;
		h_ds[i].q = 2.0;
		h_ds[i].clustering_num = 0;
		//h_ds[i].T[0] = pow(20.0f, (i + 1.0f - N / 2.0f) / (N / 2.0f)); 
		h_ds[i].T[0] = 2.0;	//	2.0固定
		h_ds[i].is_finished = FALSE;

#ifdef IRIS
		if (make_iris_datasets(h_ds[i].xk, DATA_NUM, P) != 0){
			fprintf(stderr, "データセット数と次元数の設定が間違っています\n");
			exit(1);
		}
		make_first_centroids(h_ds[i].vi, P*CLUSTER_NUM, 0.0, 5.0);
#else
		make_datasets(h_ds[i].xk, P*DATA_NUM, 0.0, 1.0);
		make_first_centroids(h_ds[i].vi, P*CLUSTER_NUM, 0.0, 1.0);
#endif
	
	}

	/*
		クラスタリングを繰り返し行う
		BFSバージョン
	*/
	for(int it=0; it<20; it++){


		if (0){
			//	各クラスタごとに
			printf("[%d] ", it);
			for (int k = 0; k < CLUSTER_NUM; k++){
				//	各次元ごとに
				for (int p = 0; p < P; p++){
					float total = 0.0;
					for (int n = 0; n < N; n++){
						total += h_ds[n].vi[k*P + p];
					}
					//	平均値で置き換えてみる
					total /= N;
					for (int n = 0; n < N; n++){
						h_ds[n].vi[k*P + p] = total;
					}
				}
			}
		}


		d_ds = h_ds;

		device_FCM << <1, N>> >(thrust::raw_pointer_cast(d_ds.data()));
		cudaDeviceSynchronize();

		h_ds = d_ds;

		//	エラー計算
		for (int n = 0; n < N; n++){
			h_ds[n].error[h_ds[n].clustering_num-1] = compare(targets, h_ds[n].results, DATA_NUM);
		}

	}
	
	printf("Clustering done.\n");
	printf("Starting writing.\n");


	/*
		結果をファイルにダンプする
	*/
	const char HEAD[6][10] = { "uik", "results", "xk", "err", "objfunc", "soukan"};
	for (int i = 0; i < 6; i++){
		for (int n = 0; n < N; n++){
			sprintf(buf, "out/%s%d.txt", HEAD[i], n);
			FILE *fp = fopen(buf, "w");
			switch (i){
			case 0: fprintf_uik(fp, h_ds[n].uik, CLUSTER_NUM, DATA_NUM);
			CASE 1: fprintf_results(fp, h_ds[n].results, DATA_NUM);
			CASE 2: fprintf_xk(fp, h_ds[n].xk, DATA_NUM, P);
			CASE 3: fprintf_error(fp, h_ds[n].error, h_ds[n].clustering_num);
			//CASE 4: fprintf_objfunc(fp, h_ds[n].obj_func, h_ds[n].clustering_num);
			CASE 5: fprintf_pair_df(fp, h_ds[n].error, h_ds[n].obj_func, h_ds[n].clustering_num, ' ');
			}
			fclose(fp);
		}
	}
	


	/*
		クラスタリング結果を表示する
	*/
	printf("--------------------The Clustering Result----------------------\n");
	for (int j = 0; j < N; j++){
		printf("[%d] T=", j);
		for (int i = 0; i < TEMP_SCENARIO_NUM && h_ds[j].T[i]!=0.0; i++) printf("%1.2f ", h_ds[j].T[i]);
		int error = compare(targets, h_ds[j].results, DATA_NUM);
		printf(" e=%d\n", error);
	}
	


	return 0;
}






/*
FCM収束判定関数
GPUでViの収束判定を行う
*/
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

/*
FCM冷却関数
VFAで温度を下げる
*/
__device__ void __device_VFA(float *T, float Thigh, int k, float D, float Cd = 2.0){
	*T = Thigh * exp(-Cd*pow((float)k - 1, 1.0f / D));
}

/*
FCM
クラスタ中心を更新
*/
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

/*
FCM
uikを更新する
*/
__device__ void __device_update_uik(float *uik, float *dik, int iSize, int kSize, float m){
	for (int i = 0; i < iSize; i++){
		for (int k = 0; k < kSize; k++){
			float sum = 0;
			for (int j = 0; j < iSize; j++){
				sum += pow((float)(dik[i*kSize + k] / dik[j*kSize + k]), float(1.0 / (m- 1.0)));
			}
			uik[i*kSize + k] = 1.0 / sum;
		}
	}
}

/*
アニーリングでuikを更新する
*/
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

/*
eval
*/
__device__ void __device_eval(float *uik, int *results, int iSize, int kSize){
	for (int k = 0; k < kSize; k++){
		results[k] = 0;
		float maxValue = uik[0*kSize + k];
		for (int i = 1; i < iSize; i++){
			if (maxValue < uik[i*kSize + k]){
				maxValue = uik[i*kSize + k];
				results[k] = i;
			}
		}
	}

}

/*
	正しいIRISのデータと比較していくつ間違っているか取得する
	50x3とする
	暫定的な処置です
*/
__device__ void __device_iris_error(float *uik, int *error, int iSize, int kSize){
	int sum[] = { 0, 0, 0 };
	int err = 0;
	
	for (int k = 0; k < kSize; k++){
		float maxValue = uik[0*kSize +k];
		int maxIndex = 0;
		for (int i = 1; i < iSize; i++){
			//	最も大きいindexを取得
			float value = uik[i*kSize + k];
			if (maxValue < value){
				value = maxValue;
				maxIndex = i;
			}
		}
		//	大きいindexに合計値を足す
		sum[maxIndex] ++;
		
		//	50個になったらエラーを計算する
		if (k == 49 || k == 99 || k == 149){
			err += 50 - MAX3(sum[0], sum[1], sum[2]);
			for (int m = 0; m <  3; m++) sum[m] = 0;
		}	
		
	}
	*error = err;

}


/*
目的関数JFCMを定義しておく
最適解の判断に利用する
*/
__device__ void __device_jfcm(float *uik, float *dik, float *jfcm, float m, int iSize, int kSize){
	float total = 0.0;
	for (int i = 0; i < iSize; i++){
		for (int k = 0; k < kSize; k++){
			total += pow(uik[i*kSize + k], 1.0f) * dik[i*kSize + k];
		}
	}
	*jfcm = total;
}

__device__ void __device_jtsallis(float *uik, float *dik, float *j, float q, float T, int iSize, int kSize){
	float total = 0.0;
	for (int i = 0; i < iSize; i++){
		for (int k = 0; k < kSize; k++){
			float ln_q = (pow(uik[i*kSize + k], 1.0f - q) - 1.0f) / (1.0f - q);
			total += pow(uik[i*kSize + k], q) * dik[i*kSize + k] + T * pow(uik[i*kSize + k], q) *ln_q;
		}
	}
	*j = total;
}

/*
FCM
dikを更新する
*/
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


/*
FCM
距離を測る
*/
__device__ void __device_distance(float* d, float *v1, float *v2, int pSize){
	int p;
	double total = 0.0;
	for (p = 0; p < pSize; p++){
		/* v1[p] * v2[p] */
		/* 1次元配列で確保されている場合に備えてあえてこうしています */
		total += pow(*(v1 + p) - *(v2 + p), 2);
	}
	*d = total;
}

/*
arrayをコピーする
*/
__device__ void __device_copy_float(float *src, float *dst, int size){
	for (int i = 0; i < size; i++){
		dst[i] = src[i];
	}
}

/*
	FCM
*/
__global__ void device_FCM(DataSet *ds){
	int i = threadIdx.x;
	float err;
	float t;
	float jfcm;

	//	クラスタリングしない
	if (ds[i].is_finished){
		return;
	}

	//	uikを更新する
	__device_update_dik(ds[i].dik, ds[i].vi, ds[i].xk, CLUSTER_NUM, DATA_NUM, P);
	__device_update_uik_with_T(ds[i].uik, ds[i].dik, CLUSTER_NUM, DATA_NUM, ds[i].q, ds[i].T[ds[i].t_pos]);

	//	分類結果を更新する
	__device_eval(ds[i].uik, ds[i].results, CLUSTER_NUM, DATA_NUM);

	//	viのバックアップを取る
	__device_copy_float(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM*P);

	//	vi(centroids)を更新する
	__device_update_vi(ds[i].uik, ds[i].xk, ds[i].vi, CLUSTER_NUM, DATA_NUM, P, ds[i].q);

	//	クラスタリング回数を増やしておく
	ds[i].clustering_num++;

	//	同一温度での収束を判定
	//	収束していなければそのままの温度で繰り返す
	__device_calc_convergence(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM, P, &err);
	//err= 0; // 温度を下げる
	if (EPSIRON < err){
		//	温度を下げずに関数を終了
		ds[i].t_pos++;
		ds[i].T[ds[i].t_pos] = ds[i].T[ds[i].t_pos - 1];
		return;
	}

	//	前の温度との収束を判定
	//	収束していたら終了
	__device_calc_convergence(ds[i].vi, ds[i].Vi_bak, CLUSTER_NUM, P, &err);
	//err = 0; // 終了
	if (err < EPSIRON){
		//	この時点でクラスタリングを終了する
		ds[i].is_finished = TRUE;
		return;
	}

	//	バックアップ
	//	温度を下げる前のviを保存
	__device_copy_float(ds[i].vi, ds[i].Vi_bak, CLUSTER_NUM*P);

	// 収束していなければ温度を下げて繰り返す
	//	cdをうまいことちょうせいする
	float cd = (4.0-1.01)*i/N + 1.01;
	ds[i].t_pos++; 
	ds[i].t_change_num++;
	__device_VFA(&t, ds[i].T[0], ds[i].t_change_num + 1, P, cd);
	ds[i].T[ds[i].t_pos] = t;

	

}



/*
関数node_expand()
遷移先の温度を決定し、子供を生成する。
ここではFCM法の実行はせず、親から値を引き継ぐのみ。
node_execute()でFCM法を1回だけ実行する。
TODO: 生成する子供の数, 次回温度の決定。
*/
/*
void node_expand(const node_t *node, std::vector<node_t> *children){
if (node->temp_scenario.size() > 2) return;

for (int i = 0; i < 3; i++){
node_t child;
std::copy(node->temp_scenario.begin(), node->temp_scenario.end(), std::back_inserter(child.temp_scenario));
child.temp_scenario.push_back(node->temp_scenario.back() / 2.0f);
children->push_back(child);
}

}
*/



/*
ノードをGPUで展開する
*/
__global__ void gpu_node_execute(int *results){
	int idx = threadIdx.x;
	results[idx] = threadIdx.x;
}


/*
幅優先探索(Breadth First Search)
*/
/*
int BFS(node_t node){
int n = 0; // これまでに探索したノード数
std::list<node_t> open_list;	//	オープンリスト

open_list.push_back(node);
while (!open_list.empty()){
node = open_list.front();
for (int i = 0; i<node.temp_scenario.size(); i++){
printf("%f ", node.temp_scenario[i]);
}
printf("\n");

if (node_is_goal(&node)){
return n;
}

n++;
open_list.pop_front();

//	CPUで子ノードを展開する
std::vector<node_t> children;
node_expand(&node, &children);

//	CPU→GPUデータコピー
//	node_t型のままでは利用できないので変換しておく
thrust::device_vector<int> d_results(8);

for (auto it = children.begin(); it != children.end(); it++){

}

//	並列でFCM実行する
gpu_node_execute << <1, 8 >> >(thrust::raw_pointer_cast(d_results.data()));

//	CPU→GPUデータコピー
// node_t型に変換しておく
//auto it_results = d_results.begin
for (auto it = children.begin(); it != children.end(); it++){
(*it).result.push_back(0);
}

//	open_listに追加して再度探索
int n = children.size();
for (int i = 0; i < n; i++){
open_list.push_back(children[i]);
}

}
return -1;
}
*/

