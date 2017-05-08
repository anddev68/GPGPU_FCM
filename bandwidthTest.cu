/*
����O���t�T���𗘗p�����o�[�W����
�����W���^ - �v���Z�b�T�ԂŃm�[�h��open list�����L��������ɋ��L����
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
GPU�v���O���~���O�ł͉ϒ��z����g�������Ȃ����ߒ萔�l�𗘗p���Ă��܂��B
�K�X�l��ς��邱��
########################################################
*/

//	IRIS�̃f�[�^���g���ꍇ��#define���邱��
#define IRIS

#define MAX3(a,b,c) ((a<b)? ((b<c)? c: b):  ((a<c)? c: a))
#define CASE break; case

#ifdef IRIS
	#define CLUSTER_NUM 3 /*�N���X�^��*/
	#define DATA_NUM 150 /*�f�[�^��*/
	#define P 4 /* ������ */
#else
	#define CLUSTER_NUM 2 /*�N���X�^��*/
	#define DATA_NUM 150 /*�f�[�^��*/
	#define P 2 /* ������ */
#endif

#define TEMP_SCENARIO_NUM 20 /*���x�J�ڃV�i���I�̐�*/
#define ERROR_SCENARIO_NUM 20 /*�덷�J�ڃV�i���I�̐�*/
#define MAX_CLUSTERING_NUM 20 /* �ő�J��Ԃ��� -> �����I�ɃV�i���I�̐��ɂ����� */

#define EPSIRON 0.001 /* ���e�G���[*/
#define N 128  /* �X���b�h��*/



typedef unsigned  int uint;
using namespace std;

/*
�f�o�C�X�ɓn������/�󂯎��̃f�[�^�Z�b�g
device_vector�ɓ˂����ލ\���̂̒��͂ǂ����ʏ�̔z��ŗǂ��炵���B
���ׁ̈A�ϒ��z��͎g�p�ł��Ȃ��\���������B
FCM�ł�dik, uik...
*/
typedef struct{
	float dik[DATA_NUM*CLUSTER_NUM];
	float uik[DATA_NUM*CLUSTER_NUM];
	float xk[DATA_NUM*P];
	float vi[CLUSTER_NUM*P];
	float vi_bak[CLUSTER_NUM*P];			//���ꉷ�x�ł̑O��vi
	float Vi_bak[CLUSTER_NUM*P];			//�قȂ鉷�x�ł̑O��vi
	int error[ERROR_SCENARIO_NUM];	//	�G���[�V�i���I
	float obj_func[MAX_CLUSTERING_NUM]; // �ړI�֐��̃V�i���I
	float T[TEMP_SCENARIO_NUM]; //	���x�J�ڂ̃V�i���I
	int results[DATA_NUM];	//	���s����
	float q;		//	q�l
	int t_pos;		//	���x�V�i���I�Q�ƈʒu
	int t_change_num;	//	���x�ύX��
	int clustering_num;	//	�N���X�^�����O��
	BOOL is_finished; //�N���X�^�����O�I�������𖞂��������ǂ���
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
		�z�X�g�ƃf�o�C�X�̃f�[�^�̈���m�ۂ���
		DataSetIn, DataSetOut��FCM�ɗp����f�[�^�̏W���A�\���̂Ȃ̂ŁA�q�m�[�h�����m�ۂ���΂悢
		�m�ې�1�ɂ���ƕ��񉻂��s�킸�A�ʏ�VFA+FCM�ōs��
	*/
	thrust::device_vector<DataSet> d_ds(N);
	thrust::host_vector<DataSet> h_ds(N);

	/*
		���m�ȕ���
	*/
	int targets[150];
	make_iris_150_targes(targets);
	char buf[32];


	/*
		vector�̏�����
	*/
	for(int i=0; i<N; i++){
		h_ds[i].t_pos = 0;
		h_ds[i].q = 2.0;
		h_ds[i].clustering_num = 0;
		//h_ds[i].T[0] = pow(20.0f, (i + 1.0f - N / 2.0f) / (N / 2.0f)); 
		h_ds[i].T[0] = 2.0;	//	2.0�Œ�
		h_ds[i].is_finished = FALSE;

#ifdef IRIS
		if (make_iris_datasets(h_ds[i].xk, DATA_NUM, P) != 0){
			fprintf(stderr, "�f�[�^�Z�b�g���Ǝ������̐ݒ肪�Ԉ���Ă��܂�\n");
			exit(1);
		}
		make_first_centroids(h_ds[i].vi, P*CLUSTER_NUM, 0.0, 5.0);
#else
		make_datasets(h_ds[i].xk, P*DATA_NUM, 0.0, 1.0);
		make_first_centroids(h_ds[i].vi, P*CLUSTER_NUM, 0.0, 1.0);
#endif
	
	}

	/*
		�N���X�^�����O���J��Ԃ��s��
		BFS�o�[�W����
	*/
	for(int it=0; it<20; it++){


		if (0){
			//	�e�N���X�^���Ƃ�
			printf("[%d] ", it);
			for (int k = 0; k < CLUSTER_NUM; k++){
				//	�e�������Ƃ�
				for (int p = 0; p < P; p++){
					float total = 0.0;
					for (int n = 0; n < N; n++){
						total += h_ds[n].vi[k*P + p];
					}
					//	���ϒl�Œu�������Ă݂�
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

		//	�G���[�v�Z
		for (int n = 0; n < N; n++){
			h_ds[n].error[h_ds[n].clustering_num-1] = compare(targets, h_ds[n].results, DATA_NUM);
		}

	}
	
	printf("Clustering done.\n");
	printf("Starting writing.\n");


	/*
		���ʂ��t�@�C���Ƀ_���v����
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
		�N���X�^�����O���ʂ�\������
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
FCM��������֐�
GPU��Vi�̎���������s��
*/
__device__ void __device_calc_convergence(float *vi, float *vi_bak, int iSize, int pSize, float *err){
	float max_error = 0;
	for (int i = 0; i < iSize; i++){
		float sum = 0.0;				//	�N���X�^���S�̈ړ��ʂ��v�Z����
		for (int p = 0; p < pSize; p++){
			sum += pow(vi[i*pSize + p] - vi_bak[i*pSize + p], 2.0f);
		}
		max_error = MAX(max_error, sum);	//	�ł��傫���ړ��ʂ𔻒f��ɂ���
	}
	*err = max_error;
}

/*
FCM��p�֐�
VFA�ŉ��x��������
*/
__device__ void __device_VFA(float *T, float Thigh, int k, float D, float Cd = 2.0){
	*T = Thigh * exp(-Cd*pow((float)k - 1, 1.0f / D));
}

/*
FCM
�N���X�^���S���X�V
*/
__device__  void __device_update_vi(float *uik, float *xk, float *vi, int iSize, int kSize, int pSize, float m){
	for (int i = 0; i < iSize; i++){
		//	�Ɨ����Ă��邽�߁A����ɗ��p���鍇�v�l���o���Ă���
		float sum_down = 0;
		for (int k = 0; k < kSize; k++){
			sum_down += pow(uik[i*kSize + k], m);
		}
		//	���q���v�Z����	
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
uik���X�V����
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
�A�j�[�����O��uik���X�V����
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
	������IRIS�̃f�[�^�Ɣ�r���Ă����Ԉ���Ă��邩�擾����
	50x3�Ƃ���
	�b��I�ȏ��u�ł�
*/
__device__ void __device_iris_error(float *uik, int *error, int iSize, int kSize){
	int sum[] = { 0, 0, 0 };
	int err = 0;
	
	for (int k = 0; k < kSize; k++){
		float maxValue = uik[0*kSize +k];
		int maxIndex = 0;
		for (int i = 1; i < iSize; i++){
			//	�ł��傫��index���擾
			float value = uik[i*kSize + k];
			if (maxValue < value){
				value = maxValue;
				maxIndex = i;
			}
		}
		//	�傫��index�ɍ��v�l�𑫂�
		sum[maxIndex] ++;
		
		//	50�ɂȂ�����G���[���v�Z����
		if (k == 49 || k == 99 || k == 149){
			err += 50 - MAX3(sum[0], sum[1], sum[2]);
			for (int m = 0; m <  3; m++) sum[m] = 0;
		}	
		
	}
	*error = err;

}


/*
�ړI�֐�JFCM���`���Ă���
�œK���̔��f�ɗ��p����
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
dik���X�V����
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
�����𑪂�
*/
__device__ void __device_distance(float* d, float *v1, float *v2, int pSize){
	int p;
	double total = 0.0;
	for (p = 0; p < pSize; p++){
		/* v1[p] * v2[p] */
		/* 1�����z��Ŋm�ۂ���Ă���ꍇ�ɔ����Ă����Ă������Ă��܂� */
		total += pow(*(v1 + p) - *(v2 + p), 2);
	}
	*d = total;
}

/*
array���R�s�[����
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

	//	�N���X�^�����O���Ȃ�
	if (ds[i].is_finished){
		return;
	}

	//	uik���X�V����
	__device_update_dik(ds[i].dik, ds[i].vi, ds[i].xk, CLUSTER_NUM, DATA_NUM, P);
	__device_update_uik_with_T(ds[i].uik, ds[i].dik, CLUSTER_NUM, DATA_NUM, ds[i].q, ds[i].T[ds[i].t_pos]);

	//	���ތ��ʂ��X�V����
	__device_eval(ds[i].uik, ds[i].results, CLUSTER_NUM, DATA_NUM);

	//	vi�̃o�b�N�A�b�v�����
	__device_copy_float(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM*P);

	//	vi(centroids)���X�V����
	__device_update_vi(ds[i].uik, ds[i].xk, ds[i].vi, CLUSTER_NUM, DATA_NUM, P, ds[i].q);

	//	�N���X�^�����O�񐔂𑝂₵�Ă���
	ds[i].clustering_num++;

	//	���ꉷ�x�ł̎����𔻒�
	//	�������Ă��Ȃ���΂��̂܂܂̉��x�ŌJ��Ԃ�
	__device_calc_convergence(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM, P, &err);
	//err= 0; // ���x��������
	if (EPSIRON < err){
		//	���x���������Ɋ֐����I��
		ds[i].t_pos++;
		ds[i].T[ds[i].t_pos] = ds[i].T[ds[i].t_pos - 1];
		return;
	}

	//	�O�̉��x�Ƃ̎����𔻒�
	//	�������Ă�����I��
	__device_calc_convergence(ds[i].vi, ds[i].Vi_bak, CLUSTER_NUM, P, &err);
	//err = 0; // �I��
	if (err < EPSIRON){
		//	���̎��_�ŃN���X�^�����O���I������
		ds[i].is_finished = TRUE;
		return;
	}

	//	�o�b�N�A�b�v
	//	���x��������O��vi��ۑ�
	__device_copy_float(ds[i].vi, ds[i].Vi_bak, CLUSTER_NUM*P);

	// �������Ă��Ȃ���Ή��x�������ČJ��Ԃ�
	//	cd�����܂����Ƃ��傤��������
	float cd = (4.0-1.01)*i/N + 1.01;
	ds[i].t_pos++; 
	ds[i].t_change_num++;
	__device_VFA(&t, ds[i].T[0], ds[i].t_change_num + 1, P, cd);
	ds[i].T[ds[i].t_pos] = t;

	

}



/*
�֐�node_expand()
�J�ڐ�̉��x�����肵�A�q���𐶐�����B
�����ł�FCM�@�̎��s�͂����A�e����l�������p���̂݁B
node_execute()��FCM�@��1�񂾂����s����B
TODO: ��������q���̐�, ���񉷓x�̌���B
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
�m�[�h��GPU�œW�J����
*/
__global__ void gpu_node_execute(int *results){
	int idx = threadIdx.x;
	results[idx] = threadIdx.x;
}


/*
���D��T��(Breadth First Search)
*/
/*
int BFS(node_t node){
int n = 0; // ����܂łɒT�������m�[�h��
std::list<node_t> open_list;	//	�I�[�v�����X�g

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

//	CPU�Ŏq�m�[�h��W�J����
std::vector<node_t> children;
node_expand(&node, &children);

//	CPU��GPU�f�[�^�R�s�[
//	node_t�^�̂܂܂ł͗��p�ł��Ȃ��̂ŕϊ����Ă���
thrust::device_vector<int> d_results(8);

for (auto it = children.begin(); it != children.end(); it++){

}

//	�����FCM���s����
gpu_node_execute << <1, 8 >> >(thrust::raw_pointer_cast(d_results.data()));

//	CPU��GPU�f�[�^�R�s�[
// node_t�^�ɕϊ����Ă���
//auto it_results = d_results.begin
for (auto it = children.begin(); it != children.end(); it++){
(*it).result.push_back(0);
}

//	open_list�ɒǉ����čēx�T��
int n = children.size();
for (int i = 0; i < n; i++){
open_list.push_back(children[i]);
}

}
return -1;
}
*/

