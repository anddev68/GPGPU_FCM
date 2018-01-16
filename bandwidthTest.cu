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
GPU�v���O���~���O�ł͉ϒ��z����g�������Ȃ����ߒ萔�l�𗘗p���Ă��܂��B�K�X�l��ς��邱��
########################################################
*/

//	IRIS�̃f�[�^���g���ꍇ��#define���邱��
//#define IRIS 1
#define USE_FILE 1

#define MAX3(a,b,c) ((a<b)? ((b<c)? c: b):  ((a<c)? c: a))
#define CASE break; case

#ifdef IRIS
	#define CLUSTER_NUM 3 /*�N���X�^��*/
	#define DATA_NUM 150 /*�f�[�^��*/
	#define P 4 /* ������ */
#elif USE_FILE 
	#define CLUSTER_NUM 5 /*�N���X�^��*/
	#define DATA_NUM 200 /*�f�[�^��*/
	#define DS_FILENAME "data/c5k200p2.txt" /* �f�[�^�Z�b�g�t�@�C�������悤����ꍇ�̃t�@�C���� */
	#define P 2 /* ������ */
#else
	#define CLUSTER_NUM 3 /*�N���X�^��*/
	#define DATA_NUM 150 /*�f�[�^��*/
	#define P 2 /* ������ */
#endif

//#define TEMP_SCENARIO_NUM 80 /*���x�J�ڃV�i���I�̐�*/
//#define ERROR_SCENARIO_NUM 20 /*�덷�J�ڃV�i���I�̐�*/
#define MAX_CLUSTERING_NUM 50 /* �ő�J��Ԃ��� -> �����I�ɃV�i���I�̐��ɂ����� */

#define EPSIRON 0.001 /* ���e�G���[*/
#define N 256  /* �X���b�h��*/

#define CD 2.0
#define Q 2.0


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
	int error[MAX_CLUSTERING_NUM];	//	�G���[�V�i���I
	float obj_func[MAX_CLUSTERING_NUM]; // �ړI�֐��̃V�i���I
	float vi_moving[MAX_CLUSTERING_NUM]; // vi�̈ړ���
	float T[MAX_CLUSTERING_NUM]; //	���x�J�ڂ̃V�i���I
	float entropy[MAX_CLUSTERING_NUM];
	int results[DATA_NUM];	//	���s����
	float q;		//	q�l
	int t_pos;		//	���x�V�i���I�Q�ƈʒu
	int t_change_num;	//	���x�ύX��
	int clustering_num;	//	�N���X�^�����O��
	int exchanged;	//	�����ς݂�
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

void print_to_file(thrust::host_vector<DataSet>&);
void print_entropy(thrust::host_vector<DataSet>&);

void calc_current_entropy(DataSet*);
void calc_current_jfcm(DataSet*); // gpu���Ŏ����ς݂Ȃ񂾂��ǁC���������̂ŁCcpu���Ŏ���
void calc_current_vi_moving();

void print_results(thrust::host_vector<DataSet>&);


/*
	�A�ԃt�H���_�����֐�
	arg0: �ړ�
	return: ���������ꍇ�C�A�ԁC���s�����ꍇ-1
*/
int make_seq_dir(char head[], int max=10000){
	char textbuf[256];
	for (int i = 0; i < max; i++){
		sprintf(textbuf, "%s%d", head, i);
		if (_mkdir(textbuf) == 0){
			//	�t�H���_�쐬����
			return i;
		}
	}
	return -1;
}


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
		������
	*/
	char textbuf[32];
	int seq = make_seq_dir("vi");

	//	==================================================================
	//	 �N���X�^�����O�p�f�[�^�z�������������
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
		fprintf(stderr, "�f�[�^�Z�b�g���Ǝ������̐ݒ肪�Ԉ���Ă��܂�\n");
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
	//	�ʏ�̃t�@�C���f�[�^�Z�b�g�쐬���[�h�Ő���
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
	//	�N���X�^�����O���s��
	//
	//	==================================================================
	for (int it = 0; it < MAX_CLUSTERING_NUM; it++){
		printf("iterations=%d/%d\n", it, MAX_CLUSTERING_NUM);

		for (int n = 0; n < N; n++){
			//	�G���[�v�Z
			h_ds[n].error[h_ds[n].clustering_num - 1] = compare(targets, h_ds[n].results, DATA_NUM);
			//	�G���g���s�[�v�Z
			calc_current_entropy(&h_ds[n]);
			calc_current_jfcm(&h_ds[n]);

			//	vi���o�͂���
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


		//	�N���X�^�����O
		d_ds = h_ds;
		device_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
		cudaDeviceSynchronize();
		h_ds = d_ds;

		//	�ύX�O�ƕύX���vi�̈ړ��ʂ��v�Z

	}

	/*
		���ʂ��o�͂���
	*/
	fprintf(stdout, "Clustering done.\n");
	print_to_file(h_ds);
	print_results(h_ds);

	return 0;
}

/*
	���ʂ��t�@�C���ɏ����o���֐�
	-----------------------------------------
	�������x, �J��Ԃ��񐔁C��蕪�ސ�, �ړI�֐��̍ő�ω���, �ړI�֐��̌����� �G���g���s�[�̍ő�� �ړI�֐��̍ŏI�l �G���g���s�[�̍ő�l, �ړI�֐����ő�ƂȂ�����n
	0.54 12 24
	0.88 12 21

	�������x 0.54 0.88
	1���entropy
	2���
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
		int max_num = 0; // diff���ő剻������ n���
		int max_change_num = 0; // diff���ő剻������ ���x�X�V��
		float max_temp = 0.0; // diff���ő剻�������x
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
		// �ŏ��̒l�ƍŌ�̒l�̍���(������)���o��
		float sub = ds[i].obj_func[1] - ds[i].obj_func[num - 1];
		
		//	����ڂ̓��ꉷ�x���N���X�^�����O�����񐔂����o�͂���
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

	//	xk���o�͂���
	fp = fopen("__xk.txt", "w");
	fprintf_xk(fp, ds[0].xk, DATA_NUM, P);
	fclose(fp);

	//	diff�̕ω����o�͂���
	for (int i = 0; i < N; i++){
		char buf[16];
		sprintf(buf, "diff/%d.txt", i);
		fp = fopen(buf, "w");
		int num = ds[i].clustering_num;
		float max = 0.0;
		float total = 0.0;
		for (int j = 1; j < num; j++){
			//	jfcm���������ʂ����o��
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
	���ʂ��v�����v�g�ɓf���o���֐�
*/
void print_results(thrust::host_vector<DataSet> &ds){
	for (int i = 0; i < N; i++){
		//	�A������o�͂���
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
	�G���g���s�[�̌v�Z
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
	float t = ds[i].T[ds[i].t_pos];
	float q = ds[i].q;
//	float jfcm;

	//	�N���X�^�����O���Ȃ�
	if (ds[i].is_finished){
		return;
	}

	//	uik���X�V����
	__device_update_dik(ds[i].dik, ds[i].vi, ds[i].xk, CLUSTER_NUM, DATA_NUM, P);
	__device_update_uik_with_T(ds[i].uik, ds[i].dik, CLUSTER_NUM, DATA_NUM, q, t);

	//	���ތ��ʂ��X�V����
	__device_eval(ds[i].uik, ds[i].results, CLUSTER_NUM, DATA_NUM);

	//	vi�̃o�b�N�A�b�v�����
	__device_copy_float(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM*P);

	//	vi(centroids)���X�V����
	__device_update_vi(ds[i].uik, ds[i].xk, ds[i].vi, CLUSTER_NUM, DATA_NUM, P, ds[i].q);

	//	�N���X�^�����O�񐔂𑝂₵�Ă���
	ds[i].clustering_num++;

	//	�ړI�֐����v�Z
	//__device_jtsallis(ds[i].uik, ds[i].dik, &ds[i].obj_func[ds[i].clustering_num - 1], q, t, CLUSTER_NUM, DATA_NUM);

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
	//float cd = (4.0-1.01)*i/N + 1.01;
	ds[i].t_pos++; 
	ds[i].t_change_num++;
	__device_VFA(&t, ds[i].T[0], ds[i].t_change_num + 1, P, CD);
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

