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
#include <math.h>
#include <vector>
#include <string>
#include <sstream>
#include <list>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "Timer.h"
#include "CpuGpuData.cuh"
#include <time.h>
#include <direct.h>

#include "FCM.h"
#include "PFCM.h"
#include "Logger.h"


/*
############################ Warning #####################
GPUプログラミングでは可変長配列を使いたくないため定数値を利用しています。適宜値を変えること
########################################################
*/

//	IRISのデータを使う場合は#defineすること
//#define IRIS 1
#define USE_FILE 1

#define MAX3(a,b,c) ((a<b)? ((b<c)? c: b):  ((a<c)? c: a))
#define CASE break; case

#ifdef IRIS
	#define CLUSTER_NUM 3 /*クラスタ数*/
	#define DATA_NUM 150 /*データ数*/
	#define P 4 /* 次元数 */
#elif USE_FILE 
	#define CLUSTER_NUM 5 /*クラスタ数*/
	#define DATA_NUM 200 /*データ数*/
	#define DS_FILENAME "data/c5k200p2.txt" /* データセットファイルをしようする場合のファイル名 */
	#define P 2 /* 次元数 */
#else
	#define CLUSTER_NUM 3 /*クラスタ数*/
	#define DATA_NUM 150 /*データ数*/
	#define P 2 /* 次元数 */
#endif

//#define TEMP_SCENARIO_NUM 80 /*温度遷移シナリオの数*/
//#define ERROR_SCENARIO_NUM 20 /*誤差遷移シナリオの数*/
#define MAX_CLUSTERING_NUM 50 /* 最大繰り返し回数 -> 将来的にシナリオの数にしたい */

#define EPSIRON 0.001 /* 許容エラー*/
#define N 256  /* スレッド数*/

#define CD 2.0
#define Q 2.0


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
	int error[MAX_CLUSTERING_NUM];	//	エラーシナリオ
	float obj_func[MAX_CLUSTERING_NUM]; // 目的関数のシナリオ
	float vi_moving[MAX_CLUSTERING_NUM]; // viの移動量
	float T[MAX_CLUSTERING_NUM]; //	温度遷移のシナリオ
	float entropy[MAX_CLUSTERING_NUM];
	int results[DATA_NUM];	//	実行結果
	float q;		//	q値
	int t_pos;		//	温度シナリオ参照位置
	int t_change_num;	//	温度変更回数
	int clustering_num;	//	クラスタリング回数
	int exchanged;	//	交換済みか
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

void print_to_file(thrust::host_vector<DataSet>&);
void print_entropy(thrust::host_vector<DataSet>&);

void calc_current_entropy(DataSet*);
void calc_current_jfcm(DataSet*); // gpu側で実装済みなんだけど，おかしいので，cpu側で実装
void calc_current_vi_moving();

void print_results(thrust::host_vector<DataSet>&);


/*
	連番フォルダを作る関数
	arg0: 接頭
	return: 成功した場合，連番，失敗した場合-1
*/
int make_seq_dir(char head[], int max=10000){
	char textbuf[256];
	for (int i = 0; i < max; i++){
		sprintf(textbuf, "%s%d", head, i);
		if (_mkdir(textbuf) == 0){
			//	フォルダ作成成功
			return i;
		}
	}
	return -1;
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
		正確な分類
	*/
	int targets[150];
	make_iris_150_targes(targets);
	char buf[32];

	/*
		下準備
	*/
	char textbuf[32];
	int seq = make_seq_dir("vi");

	//	==================================================================
	//	 クラスタリング用データ配列を初期化する
	//	==================================================================

	for (int i = 0; i < N; i++){
		h_ds[i].t_pos = h_ds[i].clustering_num = 0;
		h_ds[i].q = Q;
		//h_ds[i].T[0] = pow(10000, (i + 1.0f - N / 2.0f) / (N / 2.0f));
		h_ds[i].T[0] = pow(25, (i + 1.0f - N / 2.0f) / (N / 2.0f));
		h_ds[i].is_finished = FALSE;
		h_ds[i].exchanged = FALSE;
	}

#ifdef IRIS
	if (make_iris_datasets(h_ds[i].xk, DATA_NUM, P) != 0){
		fprintf(stderr, "データセット数と次元数の設定が間違っています\n");
		exit(1);
	}

	if (i == 0){
		make_first_centroids(h_ds[i].vi, P*CLUSTER_NUM, 0.0, 5.0);
	}
	else{
		deepcopy(h_ds[0].vi, h_ds[i].vi, P*CLUSTER_NUM);
	}
#elif USE_FILE
	if (load_dataset(DS_FILENAME, h_ds[0].xk, P, DATA_NUM) == -1){
		fprintf(stderr, "NO SUCH FILE\n");
		exit(-1);
	}
	for (int i = 1; i < N; i++){
		deepcopy(h_ds[0].xk, h_ds[i].xk, P*DATA_NUM);
	}
	for (int i = 0; i < N; i++){
		make_first_centroids(h_ds[i].vi, P*CLUSTER_NUM, 0.0, 10.0);
	}
#else
	//	通常のファイルデータセット作成モードで生成
	float MU =1.0;
	make_datasets(h_ds[0].xk, P*DATA_NUM / 3, 0.0, 4.0*MU);
	make_datasets(&h_ds[0].xk[P*DATA_NUM / 3], P*DATA_NUM / 3, 3.0*MU, 7.0*MU);
	make_datasets(&h_ds[0].xk[P*DATA_NUM * 2 / 3], P*DATA_NUM / 3, 6.0 *MU, 10.0*MU);
	for (int i = 1; i < N; i++){
		deepcopy(h_ds[0].xk, h_ds[i].xk, P*DATA_NUM);
	}
	for (int i = 0; i < N; i++){
		make_first_centroids(h_ds[i].vi, P*CLUSTER_NUM, 0.0, 10.0*MU);
	}

#endif



	
	//	==================================================================
	//	クラスタリングを行う
	//
	//	==================================================================
	for (int it = 0; it < MAX_CLUSTERING_NUM; it++){
		printf("iterations=%d/%d\n", it, MAX_CLUSTERING_NUM);

		for (int n = 0; n < N; n++){
			//	エラー計算
			h_ds[n].error[h_ds[n].clustering_num - 1] = compare(targets, h_ds[n].results, DATA_NUM);
			//	エントロピー計算
			calc_current_entropy(&h_ds[n]);
			calc_current_jfcm(&h_ds[n]);

			//	viを出力する
			{
				char textbuf[32];
				sprintf(textbuf, "vi%d/Thigh=%.5f.txt", seq, h_ds[n].T[0]);
				FILE *fp = fopen(textbuf, "a");
				for (int k = 0; k < CLUSTER_NUM; k++){
					for (int p = 0; p < P; p++){
						fprintf(fp, "%.6f  ", h_ds[n].vi[k*P + p]);
					}
				}
				fprintf(fp, "\n");
				fclose(fp);
			}
		}


		//	クラスタリング
		d_ds = h_ds;
		device_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
		cudaDeviceSynchronize();
		h_ds = d_ds;

		//	変更前と変更後のviの移動量を計算

	}

	/*
		結果を出力する
	*/
	fprintf(stdout, "Clustering done.\n");
	print_to_file(h_ds);
	print_results(h_ds);

	return 0;
}

/*
	結果をファイルに書き出す関数
	-----------------------------------------
	初期温度, 繰り返し回数，誤り分類数, 目的関数の最大変化量, 目的関数の減少量 エントロピーの最大量 目的関数の最終値 エントロピーの最大値, 目的関数が最大となった回数n
	0.54 12 24
	0.88 12 21

	初期温度 0.54 0.88
	1回目entropy
	2回目
	------------------------------------------
*/
#define DIV 1
void print_to_file(thrust::host_vector<DataSet> &ds){
	FILE *fp = fopen("__dump.txt", "w");
	for (int i = 0; i < N; i++){
		int num = ds[i].clustering_num;
		float max = 0.0;
		float hmax = 0.0;
		float total = 0.0;
		int max_num = 0; // diffが最大化した回数 n回目
		int max_change_num = 0; // diffが最大化した回数 温度更新回数
		float max_temp = 0.0; // diffが最大化した温度
		for (int j = 2; j < num; j++){
			float diff = abs(ds[i].obj_func[j] - ds[i].obj_func[j - 1]);
			float hdiff = abs(ds[i].entropy[j] - ds[i].entropy[j-1]);
			if (diff > max){
				max = diff;
				max_num = j-1;
				max_temp = ds[i].T[j-1];
			}
			//max = MAX(diff, max);
			hmax = MAX(hdiff, hmax);
			total += diff;
		}
		// 最初の値と最後の値の差分(減少量)を出力
		float sub = ds[i].obj_func[1] - ds[i].obj_func[num - 1];
		
		//	何回目の同一温度をクラスタリングした回数かを出力する
		float t_tmp = ds[i].T[0];
		int clustering_num_same_tmp = 0;
		for (int i = 1; i < num; i++){
			if (t_tmp != ds[i].T[i]){
				t_tmp = ds[i].T[i];
				clustering_num_same_tmp = 0;
			}
			clustering_num_same_tmp++;
		}
		
		fprintf(fp, "%.4f %d %d %.4f %.4f %.4f %.4f %.4f %d %.4f %d\n", 
			ds[i].T[0], num, ds[i].error[num-1], max, sub, hmax, ds[i].obj_func[num-1], ds[i].entropy[num-1], max_num, max_temp, clustering_num_same_tmp);
	}
	fclose(fp);
	
	fp = fopen("__entropy.txt", "w");
	for (int i = 0; i < N; i++){
		if (i % DIV== 0){
			fprintf(fp, "T=%.4f", ds[i].T[0]);
			fprintf(fp, ",");
		}
	}
	fprintf(fp, "\n");

	for (int j = 0; j < MAX_CLUSTERING_NUM; j++){
		for (int i = 0; i < N; i++){
			if (i % DIV == 0){
				if (ds[i].entropy[j] != 0)
					fprintf(fp, "%.4f", ds[i].entropy[j]);
				fprintf(fp, ",");
			}
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	fp = fopen("__jfcm.txt", "w");
	for (int i = 0; i < N; i++){
		if (i % DIV == 0){
			fprintf(fp, "T=%.4f", ds[i].T[0]);
			fprintf(fp, ",");
		}
	}
	fprintf(fp, "\n");

	for (int j = 0; j < MAX_CLUSTERING_NUM; j++){
		for (int i = 0; i < N; i++){
			if (i % DIV == 0){
				if (ds[i].obj_func[j] != 0)
					fprintf(fp, "%.4f", ds[i].obj_func[j]);
				fprintf(fp, ",");
			}
		}
		fprintf(fp, "\n");
	}
	fclose(fp);

	//	xkを出力する
	fp = fopen("__xk.txt", "w");
	fprintf_xk(fp, ds[0].xk, DATA_NUM, P);
	fclose(fp);

	//	diffの変化を出力する
	for (int i = 0; i < N; i++){
		char buf[16];
		sprintf(buf, "diff/%d.txt", i);
		fp = fopen(buf, "w");
		int num = ds[i].clustering_num;
		float max = 0.0;
		float total = 0.0;
		for (int j = 1; j < num; j++){
			//	jfcmが増えた量だけ出力
			//float diff = abs(ds[i].obj_func[j - 1] - ds[i].obj_func[j]);
			//max = MAX(diff, max);
			float diff = abs(ds[i].obj_func[j] - ds[i].obj_func[j - 1]);
			float t = ds[i].T[j - 1];
			//fprintf(fp, "%d %.4f %.4f\n", i+1, t, diff);
			fprintf(fp, "%.4f\n",  diff);
		}
		fclose(fp);
	}
}

/*
	結果をプロンプトに吐き出す関数
*/
void print_results(thrust::host_vector<DataSet> &ds){
	for (int i = 0; i < N; i++){
		//	帰属先を出力する
		for (int k = 0; k < DATA_NUM; k++){
			float max = 0.0;
			int index = 0;
			for (int j = 0; j < CLUSTER_NUM; j++){
				if (ds[i].uik[j*DATA_NUM + k] > max){
					max = ds[i].uik[j*DATA_NUM + k];
					index = j;
				}
			}
			printf("%d ", index);
		}
		printf("\n");
	}


}

/*
	エントロピーの計算
*/
void calc_current_entropy(DataSet *ds){
		float ent = 0.0;
		for (int k = 0; k < DATA_NUM; k++){
			for (int i = 0; i < CLUSTER_NUM; i++){
				//fprintf(stdout, "%.3f  ", h_ds[0].uik[i*DATA_NUM+k]);
				float uik = ds->uik[i*DATA_NUM + k];
				ent += pow(uik, Q) * (pow(uik, 1.0 - Q) - 1.0) / (1.0 - Q);
			}
		}
		/*
		float org = 0.0;
		for (int k = 0; k < DATA_NUM; k++){
			for (int i = 0; i < CLUSTER_NUM; i++){
				//fprintf(stdout, "%.3f  ", h_ds[0].uik[i*DATA_NUM+k]);
				float uik = ds->uik[i*DATA_NUM + k];
				//org += pow(uik, Q) * h_ds[n].dik[i*CLUSTER_NUM+k];
				org += pow(uik, Q) * ds->dik[i*DATA_NUM + k];
			}
		}
		float T = ds->T[ds->t_pos];
		ds->entropy[ds->clustering_num - 1] = org + T*ent;
		*/
		ds->entropy[ds->clustering_num - 1] = ent;
}

void calc_current_jfcm(DataSet *ds){
	float org = 0.0;
	for (int k = 0; k < DATA_NUM; k++){
		for (int i = 0; i < CLUSTER_NUM; i++){
			//fprintf(stdout, "%.3f  ", h_ds[0].uik[i*DATA_NUM+k]);
			float uik = ds->uik[i*DATA_NUM + k];
			//org += pow(uik, Q) * h_ds[n].dik[i*CLUSTER_NUM+k];
			org += pow(uik, Q) * ds->dik[i*DATA_NUM + k];
		}
	}
	//float T = ds->T[ds->t_pos];
	ds->obj_func[ds->clustering_num - 1] = org;
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
	float t = ds[i].T[ds[i].t_pos];
	float q = ds[i].q;
//	float jfcm;

	//	クラスタリングしない
	if (ds[i].is_finished){
		return;
	}

	//	uikを更新する
	__device_update_dik(ds[i].dik, ds[i].vi, ds[i].xk, CLUSTER_NUM, DATA_NUM, P);
	__device_update_uik_with_T(ds[i].uik, ds[i].dik, CLUSTER_NUM, DATA_NUM, q, t);

	//	分類結果を更新する
	__device_eval(ds[i].uik, ds[i].results, CLUSTER_NUM, DATA_NUM);

	//	viのバックアップを取る
	__device_copy_float(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM*P);

	//	vi(centroids)を更新する
	__device_update_vi(ds[i].uik, ds[i].xk, ds[i].vi, CLUSTER_NUM, DATA_NUM, P, ds[i].q);

	//	クラスタリング回数を増やしておく
	ds[i].clustering_num++;

	//	目的関数を計算
	//__device_jtsallis(ds[i].uik, ds[i].dik, &ds[i].obj_func[ds[i].clustering_num - 1], q, t, CLUSTER_NUM, DATA_NUM);

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
	//float cd = (4.0-1.01)*i/N + 1.01;
	ds[i].t_pos++; 
	ds[i].t_change_num++;
	__device_VFA(&t, ds[i].T[0], ds[i].t_change_num + 1, P, CD);
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

