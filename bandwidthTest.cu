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


/*
############################ Warning #####################
GPU�v���O���~���O�ł͉ϒ��z����g�������Ȃ����ߒ萔�l�𗘗p���Ă��܂��B
�K�X�l��ς��邱��
########################################################
*/

#define MAX3(a,b,c) ((a<b)? ((b<c)? c: b):  ((a<c)? c: a))

#define CLUSTER_NUM 3 /*�N���X�^��*/
#define DATA_NUM 150 /*�f�[�^��*/
#define TEMP_SCENARIO_NUM 20 /*���x�J�ڃV�i���I�̐�*/
#define P 4 /* ������ */
#define EPSIRON 0.001 /* ���e�G���[*/
#define N 128 /* �f�[�^�Z�b�g�� */

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
	int error;	//	�G���[��
	float T[TEMP_SCENARIO_NUM]; //	���x�J�ڂ̃V�i���I
	int results[DATA_NUM];	//	���s����
	float q;		//	q�l
	int t_pos;		//	���x�V�i���I�Q�ƈʒu
	int t_change_num;	//	���x�ύX��
	float jfcm;
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

float my_random(float min, float max){
	return min + (float)(rand() * (max - min) / RAND_MAX);
}

void deepcopy_vector(vector<int> *src, vector<int> *dst){
	for (int i = 0; i < src->size(); i++){
		(*dst)[i] = (*src)[i];
	}	
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
		ds[j].q = 5.0;		//	�Ƃ肠����q��2.0�Œ�
		ds[j].T[0] = pow(20.0f, (j + 1.0f - N/2.0f) / (N/2.0f));  // Thigh�ŏ������x������
		ds[j].is_finished = FALSE;
		for (int i = 0; i < CLUSTER_NUM; i++){
			//	�����_��������
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

void iris_datasets(DataSet ds[]){
	FILE *fp = fopen("data/iris.txt", "r");
	float tmp_xk[DATA_NUM*P];
	for (int k = 0; k < DATA_NUM; k++){
		for (int p = 0; p < P; p++){
			float tmp;
			fscanf(fp, "%f", &tmp);
			tmp_xk[k * P + p] = tmp;
		}
	}
	for (int j = 0; j < N; j++){
		ds[j].t_pos = 0;
		ds[j].q = 2.0;		//	�Ƃ肠����q��2.0�Œ�
		ds[j].T[0] = pow(20.0f, (j + 1.0f - N / 2.0f) / (N / 2.0f));  // Thigh�ŏ������x������
		ds[j].is_finished = FALSE;
		for (int i = 0; i < CLUSTER_NUM; i++){
			for (int p = 0; p < P; p++){
				ds[j].vi[i * P + p] = my_random(0.0, 10.0);
			}
		}
		for (int k = 0; k < DATA_NUM; k++){
			for (int p = 0; p < P; p++){
				ds[j].xk[k * P + p] = tmp_xk[k * P + p];
			}
		}
	}
	fclose(fp);
}

void print_result(const DataSet *ds){
	printf("T=");
	for (int i = 0; i < TEMP_SCENARIO_NUM && ds->T[i]!=0.0; i++) printf("%1.2f ", ds->T[i]);
	//printf("\n");
	//printf("q=%f", ds->q);
	//printf("\n");

	/*
	printf("results=\n");
	for (int i = 0; i < DATA_NUM; i++){
			printf("%d ", ds->results[i]);
			if ((i + 1) % 20 == 0) printf("\n");
	}
	printf("\n");
	*/

	printf("error=%d\n", ds->error);
	//printf("jfcm = %f\n", ds->jfcm);

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

int compare(const int *target, int *sample, int size){

	//	[0,1,2]�̑g�ݍ��킹�̍쐬�p�z��Ɛ����p�^�[��
	vector<int> pattern = vector<int>();
	vector<int> good_pattern = vector<int>();
	for (int i = 0; i < 3; i++){
		pattern.push_back(i);
		good_pattern.push_back(0);
	}

	//	�G���[�ŏ��l
	int min_error = INT_MAX;

	//	���ׂĂ̒u���p�^�[���Ń}�b�`���O
	do{
		//	�G���[��
		int error = 0;
		//	���ׂẴf�[�^�ɂ��āA
		for (int j = 0; j < size; j++){
			if (0 <= sample[j] && sample[j] < 3){
				int index = pattern[sample[j]];	//	�u������
				if (target[j] != index) error++;	//	���������
			}
			else{
				error++;
			}
		}
		//	�땪�ސ������Ȃ���Γ���ւ���
		if (error < min_error){
			min_error = error;
			deepcopy_vector(&pattern, &good_pattern);
		}

	} while (next_permutation(pattern.begin(), pattern.end()));

	//	�u���p�^�[���𗘗p���āA�C���f�b�N�X��u������
	for (int i = 0; i < size; i++){
		if (0 <= sample[i] && sample[i] < 3){
			sample[i] = good_pattern[sample[i]];
		}
	}
	return min_error;
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
	������Ԃ��쐬����
	TODO:�����_���p�^�[��, �����̃f�[�^�Z�b�g�p�^�[����p�ӂ���K�v����
	*/
	//const float listT[N] = { 50.0, 20.0, 10.0, 5.0, 2.0, 1.0};
	init_datasets(&h_ds[0]);

	/*
	��������BFS�œW�J����
	*/
	for (int i = 0; i < 20; i++){

		/*
		HOST����������GPU�������փR�s�[
		*/
		d_ds = h_ds;

		/*
		DataSetIn�ɑ΂�FCM�@��K�p���邱�Ƃɂ��ADataSetOut���擾����
		*/
		device_FCM << <1, N>> >(thrust::raw_pointer_cast(d_ds.data()));
		cudaDeviceSynchronize();

		cudaError_t error = cudaGetLastError();
		if (error != cudaSuccess) {
			fprintf(stderr, "%s\n", cudaGetErrorString(error));
		}

		/*
		GPU����������HOST�������փR�s�[
		�C�R�[���ő���ł���炵��
		*/
		h_ds = d_ds;

	}



	/*
		�����쐬
	*/
	int targets[150];
	for (int i = 0; i < 50; i++) targets[i] = 0;
	for (int i = 50; i < 100; i++) targets[i] = 1;
	for (int i = 100; i < 150; i++) targets[i] = 2;
	for (int i = 0; i < N; i++){
		/*
		for (int j = 0; j < DATA_NUM; j++){
			printf("%d ", h_ds[i].results[j]);
		}
		printf("\n");
		*/
		h_ds[i].error = compare(targets, h_ds[i].results, DATA_NUM);
	}

	/*
		uik���t�@�C���ɏ�������
	*/
	for (int n = 0; n < N; n++){
		char buf[256];
		sprintf(buf, "out/uik%d.txt", n);
		FILE *fp2 = fopen(buf, "w");
		for (int k = 0; k < DATA_NUM; k++){
			for (int i = 0; i < CLUSTER_NUM; i++){
				fprintf(fp2, "%f ", h_ds[n].uik[i*DATA_NUM + k]);
			}
			fprintf(fp2, "\n");
		}
		fclose(fp2);
	}

	/* results���������� */
	for (int n = 0; n < N; n++){
		char buf[256];
		sprintf(buf, "out/results%d.txt", n);
		FILE *fp3 = fopen(buf, "w");
		for (int i = 0; i < DATA_NUM; i++){
			fprintf(fp3, "%d ", h_ds[n].results[i]);
		}
		fclose(fp3);
	}


	/*
		���ʂ�\������
	*/
	printf("--------------------The Clustering Result----------------------\n");
	for (int i = 0; i < N; i++){
			printf("[%d] ", i);
			print_result(&h_ds[i]);
	}



	/*
	cudaError_t err = cudaDeviceReset();
	cudaDeviceSynchronize();
	if (err != cudaSuccess) {
		fprintf(stderr, "%s\n", cudaGetErrorString(err));
	}
	*/

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
				sum += pow((float)(dik[i*kSize + k] / dik[j*kSize + k]), float(1.0 / (m - 1.0)));
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

	//	���ł�jfcm�����߂Ă���
	__device_jtsallis(ds[i].uik, ds[i].dik, &jfcm, ds[i].q, ds[i].T[ds[i].t_pos], CLUSTER_NUM, DATA_NUM);
	ds[i].jfcm = jfcm;

	//	vi�̃o�b�N�A�b�v�����
	__device_copy_float(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM*P);

	//	vi(centroids)���X�V����
	__device_update_vi(ds[i].uik, ds[i].xk, ds[i].vi, CLUSTER_NUM, DATA_NUM, P, ds[i].q);

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
		__device_eval(ds[i].uik, ds[i].results, CLUSTER_NUM, DATA_NUM);
		//int cnt;
		//__device_iris_error(ds[i].uik, &cnt, CLUSTER_NUM, DATA_NUM);
		//ds[i].error = cnt;
		return;
	}

	//	�o�b�N�A�b�v
	//	���x��������O��vi��ۑ�
	__device_copy_float(ds[i].vi, ds[i].Vi_bak, CLUSTER_NUM*P);

	// �������Ă��Ȃ���Ή��x�������ČJ��Ԃ�
	ds[i].t_pos++; 
	ds[i].t_change_num++;
	__device_VFA(&t, ds[i].T[0], ds[i].t_change_num + 1, P);
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

