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

/*
############################ Warning #####################
GPUプログラミングでは可変長配列を使いたくないため定数値を利用しています。
適宜値を変えること
########################################################
*/
#define CLUSTER_NUM 2 /*クラスタ数*/
#define DATA_NUM 40 /*データ数*/
#define TEMP_SCENARIO_NUM 20 /*温度遷移シナリオの数*/
#define P 2 /* 次元数 */
#define EPSIRON 0.001 /* 許容エラー*/
#define N 128 /* データセット数 */

typedef unsigned  int uint;

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
	int results[DATA_NUM];	//	実行結果
	float T[TEMP_SCENARIO_NUM]; //	温度遷移のシナリオ
	float q;		//	q値
	int t_pos;		//	温度シナリオ参照位置
	int t_change_num;	//	温度変更回数
	float jfcm;
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

float my_random(float min, float max){
	return min + (float)(rand() * (max - min) / RAND_MAX);
}

void listT_to_str(std::stringstream ss, float *T, int size){
	for (int i = 0; i < size; i++){
		ss << T[i];
	}
}

void init_datasets(DataSet ds[]){
	float tmp_xk[DATA_NUM*P];
	for (int k = 0; k < DATA_NUM / 2; k++){
		tmp_xk[k * P + 0] = my_random(0.0, 0.5);
		tmp_xk[k * P + 1] = my_random(0.0, 0.5);
	}
	for (int k = DATA_NUM / 2; k < DATA_NUM; k++){
		tmp_xk[k * P + 0] = my_random(0.5, 1.0);
		tmp_xk[k * P + 1] = my_random(0.5, 1.0);
	}

	for (int j = 0; j < N; j++){
		ds[j].t_pos = 0;
		ds[j].q = 2.0;		//	とりあえずqは2.0固定
		ds[j].T[0] = pow(20.0f, (j + 1.0f - 64.0f) / 64.0f);  // Thighで初期温度を決定
		ds[j].is_finished = FALSE;
		for (int i = 0; i < CLUSTER_NUM; i++){
			//	ランダム初期化
			ds[j].vi[i * P + 0] = (double)rand() / RAND_MAX;
			ds[j].vi[i * P + 1] = (double)rand() / RAND_MAX;
		}
		for (int k = 0; k < DATA_NUM; k++){
			ds[j].xk[k * P + 0] = tmp_xk[k * P + 0];
			ds[j].xk[k * P + 1] = tmp_xk[k * P + 1];
		}

	}


	/*
	for (int k = 0; k < DATA_NUM / 2; k++){
	ds->xk[k * P + 0] = my_random(0.0, 0.5);
	ds->xk[k * P + 1] = my_random(0.0, 0.5);
	//h_ds[0].xk[k * P + 0] = my_random(0.0, 5.0);
	//h_ds[0].xk[k * P + 1] = my_random(0.0, 0.5);
	}
	for (int k = DATA_NUM / 2; k < DATA_NUM; k++){
	//[0].xk[k * P + 0] = my_random(0.75, 1.0);
	//h_ds[0].xk[k * P + 1] = my_random(0.75, 1.0);
	ds->xk[k * P + 0] = my_random(0.5, 1.0);
	ds->xk[k * P + 1] = my_random(0.5, 1.0);
	}
	*/
}

void print_result(const DataSet *ds){
	printf("T=");
	for (int i = 0; i < TEMP_SCENARIO_NUM; i++) printf("%1.2f ", ds->T[i]);
	printf("\n");
	printf("q=%f", ds->q);
	printf("\n");
	printf("vi=");
	for (int i = 0; i < CLUSTER_NUM; i++){
		for (int p = 0; p < P; p++){
			printf("%1.2f ", ds->vi[i*P + p]);
		}
	}
	printf("\n");
	printf("jfcm = %f\n", ds->jfcm);

	/*
	printf("vi_bak=");
	for (int i = 0; i < CLUSTER_NUM; i++){
	for (int p = 0; p < P; p++){
	printf("%1.2f ", ds->vi_bak[i*P + p]);
	}
	}
	printf("\n");
	*/
}



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
	初期状態を作成する
	TODO:ランダムパターン, 既存のデータセットパターンを用意する必要あり
	*/
	//const float listT[N] = { 50.0, 20.0, 10.0, 5.0, 2.0, 1.0};
	init_datasets(&h_ds[0]);

	printf("vi=\n");
	for (int j = 0; j < CLUSTER_NUM; j++) printf("%1.2f %1.2f\n", h_ds[0].vi[j*P + 0], h_ds[0].vi[j*P + 1]);
	printf("-----------------------\n");

	/*
	ここからBFSで展開する
	*/



	for (int i = 0; i < CLUSTER_NUM; i++){

		/*
		HOSTメモリからGPUメモリへコピー
		*/
		d_ds = h_ds;

		/*
		DataSetInに対しFCM法を適用することにより、DataSetOutを取得する
		*/
		device_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));

		/*
		GPUメモリからHOSTメモリへコピー
		イコールで代入できるらしい
		*/
		h_ds = d_ds;

	}

	/*
	uikをファイルに書き込む
	*/
	FILE *fp2 = fopen("data/uik.txt", "w");
	for (int k = 0; k <DATA_NUM; k++){
		for (int i = 0; i < CLUSTER_NUM; i++){
			fprintf(fp2, "%f ", h_ds[0].uik[i*DATA_NUM + k]);
		}
		fprintf(fp2, "\n");
	}
	fclose(fp2);

	/*
	Datasetをファイルに書き込む
	*/
	FILE *fp3 = fopen("data/ds.txt", "w");
	fprintf(fp3, "Number,q,T,vi.x,vi.y,objFunc\n");
	for (int i = 0; i < N; i++){
		for (int j = 0; j < CLUSTER_NUM; j++){
			std::stringstream ss;
			for (int k = 0; k < TEMP_SCENARIO_NUM; k++) ss << h_ds[i].T[k] << "->";
			fprintf(fp3, "%d,%f,%s,%f,%f,%f\n", i, h_ds[i].q, ss.str().c_str(), h_ds[i].vi[j*P + 0], h_ds[i].vi[j*P + 1], h_ds[i].jfcm);
		}
	}
	fclose(fp3);


	/*
	結果を表示する
	*/
	printf("--------------------The Clustering Result----------------------\n");
	for (int i = 0; i < N; i++){
		printf("[%d] ", i);
		print_result(&h_ds[i]);
	}
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
				sum += pow((float)(dik[i*kSize + k] / dik[j*kSize + k]), float(1.0 / (m - 1.0)));
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
	if (ds->is_finished){
		return;
	}

	//	uikを更新する
	__device_update_dik(ds[i].dik, ds[i].vi, ds[i].xk, CLUSTER_NUM, DATA_NUM, P);
	__device_update_uik_with_T(ds[i].uik, ds[i].dik, CLUSTER_NUM, DATA_NUM, ds[i].q, ds[i].T[ds[i].t_pos]);

	//	ついでにjfcmも求めておく
	__device_jtsallis(ds[i].uik, ds[i].dik, &jfcm, ds[i].q, ds[i].T[ds[i].t_pos], CLUSTER_NUM, DATA_NUM);
	ds[i].jfcm = jfcm;

	//	viのバックアップを取る
	__device_copy_float(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM*P);

	//	vi(centroids)を更新する
	__device_update_vi(ds[i].uik, ds[i].xk, ds[i].vi, CLUSTER_NUM, DATA_NUM, P, ds[i].q);

	//	同一温度での収束を判定
	//	収束していなければそのままの温度で繰り返す
	__device_calc_convergence(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM, P, &err);
	err= 0; // 温度を下げる
	if (EPSIRON < err){
		//	温度を下げずに関数を終了
		ds[i].t_pos++;
		ds[i].T[ds[i].t_pos] = ds[i].T[ds[i].t_pos - 1];
		return;
	}

	//	前の温度との収束を判定
	//	収束していたら終了
	__device_calc_convergence(ds[i].vi, ds[i].Vi_bak, CLUSTER_NUM, P, &err);
	err = 0; // 終了
	if (err < EPSIRON){
		//	この時点でクラスタリングを終了する
		ds[i].is_finished = TRUE;
		return;
	}

	//	バックアップ
	//	温度を下げる前のviを保存
	__device_copy_float(ds[i].vi, ds[i].Vi_bak, CLUSTER_NUM*P);

	// 収束していなければ温度を下げて繰り返す
	ds[i].t_pos++;
	ds[i].t_change_num++;
	__device_VFA(&t, ds[i].T[0], ds[i].t_change_num + 1, P);
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

