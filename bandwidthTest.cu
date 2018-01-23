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
#include <time.h>
#include <direct.h>


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
	#define DS_FILENAME "data/iris_formatted.txt" /* �f�[�^�Z�b�g�t�@�C�������悤����ꍇ�̃t�@�C���� */
	#define P 4 /* ������ */
#elif USE_FILE 
	#define CLUSTER_NUM 4 /*�N���X�^��*/
	#define DATA_NUM 200 /*�f�[�^��*/
	#define DS_FILENAME "data/c4k200p2.txt" /* �f�[�^�Z�b�g�t�@�C�������悤����ꍇ�̃t�@�C���� */
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
#define N 512  /* �X���b�h��*/

#define CD 2.0
#define Q 2.0


typedef unsigned  int uint;
using namespace std;

/*
	Util�n���\�b�h
*/

void deepcopy(float *src, float *dst, int size){
	for (int i = 0; i < size; i++){
		dst[i] = src[i];
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


/*
########################################################################
WARNING
6�� �n�C�u���b�h�A�j�[�����O�Ŏ������܂�

�ʓ|�Ȃ�őS���O���[�o���ϐ���
##########################################################################
*/

typedef struct{
public:
	float vi[CLUSTER_NUM*P];
	float vi_bak[CLUSTER_NUM*P];
	float uik[DATA_NUM*CLUSTER_NUM];
	float xk[DATA_NUM*P];
	float dik[DATA_NUM*CLUSTER_NUM];
	float Thigh;
	int iterations;
	BOOL finished;
}DataFormat;

void VFA(float *T, float Thigh, int k, float D, float Cd = 2.0){
	*T = Thigh * exp(-Cd*pow((float)k - 1, 1.0f / D));
}

void update_vi(float *uik, float *xk, float *vi, int iSize, int kSize, int pSize, float m){
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
		float sum = 0.0;				//	�N���X�^���S�̈ړ��ʂ��v�Z����
		for (int p = 0; p < pSize; p++){
			sum += pow(vi[i*pSize + p] - vi_bak[i*pSize + p], 2.0f);
		}
		max_error = MAX(max_error, sum);	//	�ł��傫���ړ��ʂ𔻒f��ɂ���
	}
	*err = max_error;
}

void print_results(float *uik){
		//	�A������o�͂���
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
			fprintf(fp, "%.2f  ", vi[i*P+p]);
		}
	}
	fprintf(fp, "\n");
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
	return min + (float)(rand() * (max - min) / RAND_MAX);
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

void sort2(float *src){
	if (src[0] > src[2]){
		float tmp = src[0];
		float tmp2 = src[1];
		src[0] = src[2];
		src[1] = src[3];
		src[2] = tmp;
		src[3] = tmp2;
	}
}

void sort3(float *src){
	if (src[0] > src[3]){
		float tmp = src[0];
		float tmp2 = src[1];
		float tmp3 = src[2];
		src[0] = src[0+3];
		src[1] = src[1+3];
		src[2] = src[2+3];
		src[0+3] = tmp;
		src[1+3] = tmp2;
		src[2 + 3] = tmp3;
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

__device__ void __device_VFA(float *T, float Thigh, int k, float D, float Cd = 2.0){
	*T = Thigh * exp(-Cd*pow((float)k - 1, 1.0f / D));
}

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

__device__ void __device_deepcopy(float *src, float *dst, int size){
	for (int i = 0; i < size; i++){
		dst[i] = src[i];
	}
}

__global__ void device_pre_FCM(DataFormat*dss){
	int i = threadIdx.x;
	DataFormat *ds = &dss[i];

	//	�������Ă���N���X�^�����O���Ȃ�
	if (ds->finished == TRUE)
		return;

	//	vi�̃o�b�N�A�b�v
	__device_deepcopy(ds->vi, ds->vi_bak, CLUSTER_NUM*P);

	float T = ds->Thigh;
	ds->iterations++;
	//__device_VFA(&T, ds->Thigh, ds->iterations, P);
	__device_update_dik(ds->dik, ds->vi, ds->xk, CLUSTER_NUM, DATA_NUM, P);
	__device_update_uik_with_T(ds->uik, ds->dik, CLUSTER_NUM, DATA_NUM, Q, T);
	__device_update_vi(ds->uik, ds->xk, ds->vi, CLUSTER_NUM, DATA_NUM, P, Q);

	//	��������
	float err;
	__device_calc_convergence(ds->vi, ds->vi_bak, CLUSTER_NUM, P, &err);
	if (err < EPSIRON)
		ds->finished = TRUE;
}

int main(){
	srand((unsigned)time(NULL));
	for (int i = 0; i < 100; i++) rand();

	/* ����p�ӂ��� */
	float _Vi_bak[CLUSTER_NUM*P];
	float _vi_bak[CLUSTER_NUM*P];
	float _vi[CLUSTER_NUM*P];
	float _uik[DATA_NUM*CLUSTER_NUM];
	float _xk[DATA_NUM*P];
	float _dik[DATA_NUM*CLUSTER_NUM];
	thrust::device_vector<DataFormat> d_ds(N);
	thrust::host_vector<DataFormat> h_ds(N);

	/* �f�[�^���[�h */
	if (load_dataset(DS_FILENAME, _xk, P,  DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}

	/* Thigh�̊�l���� */
	/* �搶�̕��@�ɂ��Thigh�����߂� */
	float tmp = 0.0;
	for (int i = 0; i < 1000; i++){
		make_random(_vi, CLUSTER_NUM*P, 0.0, 10.0);
		float L1k_bar = calc_L1k(_xk, _vi, CLUSTER_NUM, DATA_NUM, P);
		tmp += CLUSTER_NUM / L1k_bar;
	}
	tmp = 1000 / tmp;
	printf("Thigh=%f\n", tmp);


	/* ���̏����� */
	for (int i = 0; i < N; i++){
		h_ds[i].Thigh = pow(tmp, (i + 1.0f - N / 2.0f) / (N / 2.0f));
		h_ds[i].iterations = 0;
		make_random(h_ds[i].vi, CLUSTER_NUM*P, 0.0, 10.0);
		deepcopy(_xk, h_ds[i].xk, DATA_NUM*P);
		h_ds[i].finished = FALSE;
	}

	/* �v���N���X�^�����O */
	for (int i = 0; i < 10; i++){
		d_ds = h_ds;
		device_pre_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
		cudaDeviceSynchronize();
		h_ds = d_ds;
	}

	/* vi��\�����Ă݂� */
	FILE *fp = fopen("tmp.txt", "w");
	FILE *fp2 = fopen("it.txt", "w");
	for (int i = 0; i < N; i++){
		//sort(h_ds[i].vi, CLUSTER_NUM*P);
		//sort3(h_ds[i].vi);
		print_vi(fp, h_ds[i].vi);
		//print_results(h_ds[i].uik);
		fprintf(fp2, "%d\n", h_ds[i].iterations);
	}
	fclose(fp);
	fclose(fp2);
	exit(1);




	/* GPU��1��N���X�^�����O���Ă���Thigh���v�Z���� */
	/*
	float sum = 0.0;
	for (int i = 0; i < 100; i++){
	make_random(_vi, CLUSTER_NUM*P, 0.0, 10.0);
	update_dik(_dik, _vi, _xk, CLUSTER_NUM, DATA_NUM, P);
	update_uik(_uik, _dik, CLUSTER_NUM, DATA_NUM,  Q);
	update_vi(_uik, _xk, _vi, CLUSTER_NUM, DATA_NUM, P, Q);
	float L1k_bar = calc_L1k(_xk, _vi, CLUSTER_NUM, DATA_NUM, P);
	sum += CLUSTER_NUM / L1k_bar;

	for (int j = 0; j < CLUSTER_NUM; j++){
	for (int p = 0; p < P;  p++){
	printf("%.3f ", _vi[j*P + p]);
	}
	}
	printf("\n");
	}
	*/

	/* Thigh���v�Z���� */
	float sum = 0.0;
	for (int i = 0; i < N; i++){
		float L1k_bar = calc_L1k(_xk, h_ds[i].vi, CLUSTER_NUM, DATA_NUM, P);
		sum += CLUSTER_NUM / L1k_bar;
		sort(h_ds[i].vi, CLUSTER_NUM*P);
		for (int j = 0; j < CLUSTER_NUM; j++){
			for (int p = 0; p < P; p++){
				printf("%.3f ", h_ds[i].vi[j*P + p]);
			}
		}
		printf("\n");
	}
	printf("\n\nThigh=%.3f\n", N/sum);

	/* ���S���� */
	make_random(_vi, CLUSTER_NUM*P, 0.0, 10.0);
	
	/* Thigh���v�Z���� */
	float Thigh = N / sum;
	float T = Thigh;
	float q = 2.0;

	/* ���C���N���X�^�����O�X�e�b�v */
	for (int it = 0; it < 50; it++){
		printf("T=%f Processing... %d/50\n", T, it);

		/* �A���xuik�v�Z����񉻂��čX�V���� */
		update_dik(_dik, _vi, _xk, CLUSTER_NUM, DATA_NUM, P);
		update_uik_with_T(_uik, _dik, CLUSTER_NUM, DATA_NUM, q, T);

		//	vi�̃o�b�N�A�b�v�����
		deepcopy(_vi, _vi_bak, CLUSTER_NUM*P);

		//	vi(centroids)���X�V����
		update_vi(_uik, _xk, _vi, CLUSTER_NUM, DATA_NUM, P, q);

		//	���ꉷ�x�ł̎����𔻒�
		//	�������Ă��Ȃ���΂��̂܂܂̉��x�ŌJ��Ԃ�
		float err = 0.0;
		calc_convergence(_vi, _vi_bak, CLUSTER_NUM, P, &err);
		if (EPSIRON < err){
			//	���x���������ɃN���X�^�����O���p��
			continue;
		}

		//	�O�̉��x�Ƃ̎����𔻒�
		//	�������Ă�����I��
		calc_convergence(_vi, _Vi_bak, CLUSTER_NUM, P, &err);
		//err = 0; // �I��
		if (err < EPSIRON){
			//	���̎��_�ŃN���X�^�����O���I������
			break;
		}

		//	���x��������O��vi��ۑ�
		deepcopy(_vi, _Vi_bak, CLUSTER_NUM*P);

		// �������Ă��Ȃ���Ή��x�������ČJ��Ԃ�
		VFA(&T, Thigh, it, P);
	}
	

	//print_results(_uik);
	return 0;
}
