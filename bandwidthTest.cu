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

#define EPSIRON 0.001 /* 許容エラー*/
#define N 1  /* スレッド数*/

#define DATA_NUM_EACH_PROCESSOR (DATA_NUM/N)

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
	float xk[DATA_NUM_EACH_PROCESSOR*P];
	float vi[CLUSTER_NUM*P];
	float vi_bak[CLUSTER_NUM*P];			//同一温度での前のvi
	float Vi_bak[CLUSTER_NUM*P];			//異なる温度での前のvi
	int results[DATA_NUM];	//	実行結果
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

	float xk_org[DATA_NUM*P];
	float vi_org[CLUSTER_NUM*P];
	float w0[P];
	int targets[150];
	float E1;
	float DK;
	float EK;
	float PBMK;
	float K = CLUSTER_NUM;

	thrust::host_vector<float> h_xk(DATA_NUM*P);
	thrust::host_vector<float> h_uik(CLUSTER_NUM*DATA_NUM);
	thrust::host_vector<float> h_vi(CLUSTER_NUM*P);


	/*
		Step0. データセットの初期化
	*/	
	make_iris_150_targes(targets);	
	make_first_centroids(vi_org, P*CLUSTER_NUM, 0.0, 5.0);
	if (make_iris_datasets(xk_org, DATA_NUM, P) != 0){
		fprintf(stderr, "データセット数と次元数の設定が間違っています\n");
		exit(1);
	}

	/*
		Step1. (Master Processer)
		各スレッドにデータをN/pとなるようにデータセットを分割する．
		ここでは分割を行わず，GPUへのコピー処理のみ行う
		N: データ数
		p: スレッド数
	*/
	for (int i = 0; i < h_xk.size(); i++){
		h_xk[i] = xk_org[i];
	}
	for (int i = 0; i < h_vi.size(); i++){
		h_vi[i] = vi_org[i];
	}

	/*
		Step2. (All Processesors)
		ローカルデータで中央を計算し，全てのプロセッサーで共有
		PBM factor E1をローカルデータで計算し，ルートに送る．
		The function of E1
		このファクターは依存しない，クラスタ数に．
		\sum_{t=1...N}{d(x(t),w_0)}
		w_0: 全体の中心
	*/
	E1 = 0.0f;
	make_first_centroids(w0, P, 0.0, 1.0);	//	w0はランダムにしてみる
	for (int j = 0; j < N; j++){
		for (int t = 0; t < DATA_NUM_EACH_PROCESSOR; t++){
			E1 += distance(&h_xk[t*P + DATA_NUM_EACH_PROCESSOR*P*j], w0, P);
		}
	}
	printf("E1=%f\n", E1);


	/*
		Step3. (Master Processor)
		初期クラスタ中心を設定し，放流する．
		ループの最初で全てのクラスタは同じクラスタ中心を持つ．
	*/
	

	/*
		Step4. (All Processors)
		収束が達成されるまで，距離を計算する．
		uikとviを計算する．
	*/
	float m = 2.0;
	for (int t = 0; t < DATA_NUM; t++){
		for (int i = 0; i < CLUSTER_NUM; i++){
			float sum = 0.0;
			for (int j = 0; j < CLUSTER_NUM; j++){
				sum += pow(distance(&h_xk[t*CLUSTER_NUM], &h_vi[i*P], P) / distance(&h_xk[t*CLUSTER_NUM], &h_vi[j*P], P), 2.0f / (m - 1.0f));
			}
			h_uik[CLUSTER_NUM*t+i] = 1.0f / sum;
		}
	}


	/*
		Step5. (Alll Processors)
		PBM factor EKをローカルデータで計算し，ルートに送る．
	*/
	EK = 0.0f;
	for (int i = 0; i < DATA_NUM; i++){
		for (int k = 0; k < CLUSTER_NUM; k++){
			EK += h_uik[CLUSTER_NUM*i+k] * pow( distance(&h_xk[i*P], &h_vi[k*P], P), 2);
		}
	}
	printf("EK=%f\n", EK);



	/*
		Step6. (Master Processor)
		PBM indexを統合し，保存する．
		クラスタ数の範囲がカバーされたら終了，そうでなければStep 3.からやり直す．
	*/
	DK = 0.0f;
	for (int i = 0; i < K; i++){
		for (int j = 0; j < K; j++){
			DK = MAX(DK, distance(&h_vi[i*P], &h_vi[j*P], P));
		}
	}
	PBMK = pow((1.0f / K) * (E1 / EK) * DK, 2);

	printf("DK=%f\n", DK);
	printf("PBMK=%f\n", PBMK);

	/*
		Step7. 結果出力
	*/
	for (int t = 0; t < DATA_NUM; t++){
		for (int i = 0; i < CLUSTER_NUM; i++){
			printf("%f ", h_uik[t*CLUSTER_NUM+i]);
		}
		printf("\n");
	}


	while (1);
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
	/*
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

	*/

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

