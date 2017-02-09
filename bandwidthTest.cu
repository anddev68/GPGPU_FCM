/*
	���x���ƂɃX���b�h�𗧂Ăĕ��񏈗����s��
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

#define N 32	//	��x�Ɏ��s����X���b�h/���x�̐�

using namespace std;



/**
	�N���X�^�����O���ʂ�GPU����󂯎�邽�߂̍\����
*/
typedef struct{
	int iterations;
	float q;
	float T;
}Result;


/**
	src��dst�ɃR�s�[����
*/
__device__ void gpu_array_copy(DATA_TYPE *src, DATA_TYPE *dst, int len){
	for (int i = 0; i < len; i++){
		dst[i] = src[i];
	}

}


/**
q-FCM��uik-matrix���쐬����
CPU�Ɠ��l�ɒP�X���b�h�ō쐬����
@param q q�l
@param T ���x
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
q-FCM��vi���쐬����
cpu�Ɠ��l�ɒP�X���b�h�ō쐬����
*/
__device__ void gpu_update_vi(float q, DATA_TYPE *uik, DATA_TYPE *xk, DATA_TYPE *vi, int iLen, int kLen, int pLen){
	for (int i = 0; i < iLen; i++){
		//	�Ɨ����Ă��邽�߁A����ɗ��p���鍇�v�l���o���Ă���
		DATA_TYPE sum_down = 0;
		for (int k = 0; k < kLen; k++){
			sum_down += pow(uik[i*kLen + k], q);
		}
		//	���q���v�Z����	
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
	dik���쐬����
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
	��������
	@return 0 ���� ����ȊO �������Ă��Ȃ�
*/
__device__ void gpu_judgement_convergence(DATA_TYPE *vi, DATA_TYPE *vi_bak, int iLen, int pLen, int *result, DATA_TYPE epsiron = 0.001){
	DATA_TYPE max_error = 0;
	for (int i = 0; i < iLen; i++){
		DATA_TYPE sum = 0.0;				//	�N���X�^���S�̈ړ��ʂ��v�Z����
		for (int p = 0; p < pLen; p++){
			sum += pow(double(vi[p*iLen + i] - vi_bak[p*iLen + i]), 2.0);
		}
		max_error = MAX(max_error, sum);	//	�ł��傫���ړ��ʂ𔻒f��ɂ���
	}

	if (max_error < epsiron) *result = OK;	//	�N���X�^���S�̈ړ����Ȃ��Ȃ�����I��
	//else *result = 0;
}



/*
	�N���X�^�����O���s��
*/
__global__ void gpu_clustering(DATA_TYPE *g_dik, DATA_TYPE *g_uik, DATA_TYPE *g_vi, DATA_TYPE *g_vi_bak, DATA_TYPE *g_xk, Result *g_results, float *g_T, int iLen, int kLen, int pLen){

	//	�X���b�h�ԍ����擾
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	//	q�l�ݒ�
	const float q = 2.0;

	//	xk�͂ǂ�ł��ꏏ�Ȃ̂ŕϊ������͕K�v�Ȃ�
	//	�����A�N�Z�X�ł��Ȃ��ꍇ�͍l����
	DATA_TYPE *xk = &g_xk[0];

	//	�z���3�����Ŋm�ۂ��Ă���̂�
	//	�A�h���X�̕ϊ��������s��
	DATA_TYPE *dik = &g_dik[iLen*kLen*index];
	DATA_TYPE *uik = &g_uik[iLen*kLen*index];
	DATA_TYPE *vi = &g_vi[iLen*pLen*index];
	DATA_TYPE *vi_bak = &g_vi_bak[iLen*pLen*index];
	Result *result_t = &g_results[index];


	//	���x��1�����Ŋm�ۂ��Ă���̂�
	//	�X���b�h�ԍ�����ϊ��������s��
	float T = g_T[index];

	//	����uik���쐬����
	gpu_make_dik(dik, vi, xk, iLen, kLen, pLen);
	gpu_update_uik(q, T, uik, dik, iLen, kLen);


	//	��������܂ŌJ��Ԃ��s��
	//int *repeat_num = &tmp2[0];
	int iterations;
	for (iterations = 1; iterations < 8; iterations++){

		//	vi�̃o�b�N�A�b�v�����
		gpu_array_copy(vi, vi_bak, iLen*pLen);

		//	vi���X�V����
		gpu_update_vi(q, uik, xk, vi, iLen, kLen, pLen);

		//	dik���쐬����
		gpu_make_dik(dik, vi, xk, iLen, kLen, pLen);

		//	uik���쐬����
		gpu_update_uik(q, T, uik, dik, iLen, kLen);

		//	��������
		int result = NG;
		gpu_judgement_convergence(vi, vi_bak, iLen, pLen, &result);
		if (result == OK) break;
	}
	
	result_t->iterations = iterations;
	result_t->q = q;
	result_t->T = T;

}




/*
	�z�񑀍�֐�
*/
void init_random(){
	srand(time(NULL));
	for (int i = 0; i < 100; i++){
		rand();
	}
}
void fill_random(DATA_TYPE *data, int len, float min, float max){
	//	min-max�Ń����_���ɒl�𖄂߂�
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
	iris�̃f�[�^��ǂݍ���
*/
const int IRIS_P = 4;	//	4����
const int IRIS_K = 150;	//	150�̃f�[�^
const int IRIS_I = 3;	//	3�̃N���X�^
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
		fscanf(fp, "%s", &buf);	//	���O
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
	�����ރf�[�^���^�[�Q�b�g�ƃN���X�^�����O���ʁ��T���v�����r��
	����������v�Z����
	�N���X�^�ԍ��͏��s���Ƃ���
*/
int compare(int *target, int *sample, int size){
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
		for (int i = 0; i < size; i++){
			int index = pattern[sample[i]];	//	�u������
			if (target[i] != index) error++;	//	���������
		}
		//	�땪�ސ������Ȃ���Γ���ւ���
		if (error < min_error){
			min_error = error;
			deepcopy_vector(&pattern, &good_pattern);
		}
	} while (next_permutation(pattern.begin(), pattern.end()));

	//	�u���p�^�[���𗘗p���āA�C���f�b�N�X��u������
	for (int i = 0; i < size; i++){
		sample[i] = good_pattern[sample[i]];
	}

	return min_error;
}


/**
	uik����A�����Ă���N���X�^�[�̔ԍ����擾����
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
		//	�����Ƃ������A���x�����N���X�^�ɕύX����
		sample[k] = maxIndex;
	}

}


/**
	���x��ݒ肷��
*/
float VFA(float Thigh, int k, float D, float Cd=2.0){
	return Thigh * exp(-Cd * pow(k-1, 1.0/D));
}



//	====================================================================================================================
//	���C���֐�
//
//	�萔�l
//		iLen	�N���X�^��
//		kLen	�f�[�^��
//		pLen	�N���X�^�ƃf�[�^�̎���
//		N		�����Ɏ��s����X���b�h�̐�
//
//
//	CPU��GPU�ŋ��L����z��
//	�z���(�X���b�h��x�f�[�^��x����)��3�����I�Ɋm�ۂ��AGPU���ŃA�N�Z�X����A�h���X��ݒ肷��B
//	xk�͕ω����邱�Ƃ͂Ȃ��̂ŁA���̂܂܍쐬����
//		dik		|| vi-xk ||^2		
//		uik		�A���x�֐�
//		vi		�N���X�^���S
//		vi_bak	��������p�ɗ��p����ꎞ�I�Ȕz��
//		xk		�f�[�^�Z�b�g
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

	//	���ʕۑ��p
	int target[kLen] = { 0 };
	int sample[kLen] = { 0 };
	CpuGpuData<Result> results(N);	//	���ʕۑ��p

	//	���x�z���CPU�Ŋm�ۂ���
	CpuGpuData<float> T_array(N);	//	���x�z��
	float Thigh_array[N] = { 0 };				//	�������x

	//	xk�̓A�C���X�Őݒ�
	load_iris(xk.m_data, target);

	//	vi�̓����_���l�Őݒ�
	init_random();
	fill_random(vi.m_data, vi.m_size, get_min(xk.m_data, xk.m_size), get_max(xk.m_data, xk.m_size));

	//	�������xThigh��ݒ肷��
	//	�e�X���b�h���Ƃɉ��x��ݒ肷��
	for (int i = 0; i < N; i++){
		T_array.m_data[i] = 2.0 + 0.01 * i + 0.01;
		Thigh_array[i] = 2.0 + 0.01 * i + 0.01;
	}


	//	��������܂ŌJ��Ԃ�
	for (int i = 0; i < 10; i++){

		//	gpu��(���̉��x�Łj�N���X�^�����O���s��
		gpu_clustering << <1, N >> >(dik.m_data, uik.m_data, vi.m_data, vi_bak.m_data, xk.m_data, results.m_data, T_array.m_data, iLen, kLen, pLen);
		cudaDeviceSynchronize();

		cudaError_t cudaErr = cudaGetLastError();
		if (cudaErr != cudaSuccess){
			printf("%s\n", cudaGetErrorString(cudaErr));
		}

		//	���ʂ̏o��
		printf("---------------result--------------\n");
		printf("index\tq\tT\titerations\terror\n");
		for (int j = 0; j < N; j++){
			belongs(&uik.m_data[j*kLen*iLen], sample, kLen, iLen);
			int error = compare(target, sample, kLen);
			printf("%d\t%3.2f\t%3.4f\t%d\t%d\n", j, results.m_data[j].q, results.m_data[j].T, results.m_data[j].iterations, error);
		}

		//	���x��������
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