/*
	WELCOME TO �悤���� GPGPU�̈ł�
	�ɗ́C�֐��ɂ̓}�N����\���̂Ɉˑ����Ȃ��悤�Ȍ`�ɂ��Ă��܂����C
	�ꕔ�蔲���H���ł��̂܂܎g���Ă镔��������܂��D���̃\�[�X�R�[�h�ɓ]�p����ꍇ�͒��ӂ��Ă��������D
	@author Hideki.Kano
	@updated 2018/01/24

	# �����K���ɂ���
	## device/__device
	�ړ�����__device�Ƃ����̂Ɋւ��ẮC�J�[�l���֐�����̂݌Ăяo���\�ł��D
	device�̓J�[�l���֐��ł��DCPU������A�N�Z�X�ł��܂��D

	# �}�N����`�ɂ���
	�ЂƂ̃\�[�X�R�[�h�ŕ����̎�@����舵���Ă�s����C�}�N����`�ɂ��؂�ւ��Ă��܂��D
	�}�N����`��L���ɗ��p���Ă��������D
	
	# �z��̊m�ە��@�ɂ���
	GPU�ɓn���Ȃ����߁C�z��͑S�Ĉꎟ���z��Ŋm�ۂ��Ă��܂��D
	�{���̓�������CPU��GPU�ŕʁX�Ɋm�ۂ��C�R�s�[���鏈�����K�v�ł����C
	thrust���C�u�������g���Ɗm�ۂ�R�s�[�����Ɋy�ł��D
	
	## vi
	�����Ɏ���p�����C�c���ɃN���X�^���S�ԍ�i��ݒ肵�Ă��܂��D
	i�Ԗڂ̃N���X�^��p�����ڂɃA�N�Z�X����ꍇ��vi[i*P+p]�ł��D
	```
	v0.x, v0.y ..., 
	v1.x, v1.y ...
	```

	## xk
	�����Ɏ���p�����C�c���Ƀf�[�^�ԍ�k��ݒ肵�Ă��܂��D
	k�Ԗڂ̃N���X�^��p�����ڂɃA�N�Z�X���邽�߂ɂ�xk[k*P+p]�ł��D
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
//		���s����N���X�^�����O��@��#define�Ő؂�ւ���
//	====================================================================
//#define PFCM 1 // ��Ď�@1 �A���x�֐�����(���o�I�ɂ͈�ԗD�������)
//#define TPFCM 1 // ��Ď�@2 ���x����
//#define TPFCM2 1
#define HYBRID 1  // ��Ď�@3 �n�C�u���b�h�@
//#define DHYBRID 1 
 //#define AUTO_THIGH 1  // �]����@ Thigh�������I�Ɍ��肷����@
//#define DA_FCM 1 // �]����@ DA_FCM�@

#define CD 2.0
#define Q 2.0 // q�l�͑S�Ă̕��@��2.0���g�p
#define THIGH 7.0 // Thigh�ɌŒ�l�𗘗p����ꍇ�Ɏg�p
#define TMAX 20.0 // TPFCM�ŗp����ő�l
#define G_IT 1000  /* �J��Ԃ��������s���� */
#define IT 100 /* �ő�A���x�X�V��(�ł��؂��) */
#define EPSIRON 0.001 /* ��������ɗp���鋖�e�덷 */
#define N 512  /* �X���b�h��*/
#define INIT_RAND_MIN 0.0 /* �����N���X�^���S�̈ʒu */
#define INIT_RAND_MAX 7.0

//#define IRIS 1
//#define RANDOM2D 1

#ifdef IRIS
#define CLUSTER_NUM 3 /*�N���X�^��*/
#define DATA_NUM 150 /*�f�[�^��*/
#define DS_FILENAME "data/iris_formatted.txt" /* �f�[�^�Z�b�g�t�@�C�������悤����ꍇ�̃t�@�C���� */
#define P 4 /* ������ */
#elif RANDOM2D
#define CLUSTER_NUM 2
#define DATA_NUM 200 // =2^16=66536
#define P 2
#else
#define CLUSTER_NUM 4 /*�N���X�^��*/
#define DATA_NUM 4000 /*�f�[�^��*/
//#define DS_FILENAME "data/c4k200p2.txt" /* �f�[�^�Z�b�g�t�@�C�������悤����ꍇ�̃t�@�C���� */
#define DS_FILENAME "data/c4k4000.csv" /* �f�[�^�Z�b�g�t�@�C�������悤����ꍇ�̃t�@�C���� */
#define P 2 /* ������ */
#endif


/*
	=================================================
	�ėpUtil�n���\�b�h�ƃN���X
	�g���܂킵�������悤�ɕ���
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
	�A�ԃt�H���_�����֐�
	path_to_dir/[head]1,
	path_to_dir/[head]2,
	path_to_dir/[head]3,
	�Ƃ����悤�ɍ��܂��D
	arg0: �ړ�
	return: ���������ꍇ�C�A�ԁC���s�����ꍇ-1
*/
int make_seq_dir(char path[], char head[], int max = 10000){
	char textbuf[256];
	for (int i = 0; i < max; i++){
		sprintf(textbuf, "%s%s%d", head, path, i);
		if (_mkdir(textbuf) == 0){
			//	�t�H���_�쐬����
			return i;
		}
	}
	return -1;
}

/*
=================================================
	��������̓\�[�X�R�[�h�ˑ����������֐�
	iSize,kSize���}�N����`����Ă�̂ŕ����ł��܂���
	�ꎞ�I�Ɋ֐��������Ǝv���Ă�������
=================================================
*/

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

//	�����̃G�C���A�X�Ȃ�ł����ǈꉞ�������Ă���
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
	��������FCM�@��p�֐�
	���炭���̂܂ܕ����ł���Ƃ͎v�����C
	TODO: FCM�@�Ŏg���܂킹��悤�ɕ���������
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
	int iterations; /* �A���x�X�V��(�g�[�^��) */
	BOOL finished;

	/* ���x���񉻂ł̂ݎg�p */
	int t_update_cnt; /* ���x�X�V�� */
	float Vi_bak[CLUSTER_NUM*P]; /* �O���x�ł̂�vi */
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
	//	[0,1,2]�̑g�ݍ��킹�̍쐬�p�z��Ɛ����p�^�[��
	vector<int> pattern = vector<int>();
	vector<int> good_pattern = vector<int>();
	for (int i = 0; i < iSize; i++){
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
		for (int i = 0; i < kSize; i++){
			//if (2 < sample[i]) return -2;
			int index = pattern[sample[i]];	//	�u������
			if (target[i] != index) error++;	//	���������
		}
		//	�땪�ސ������Ȃ���Γ���ւ���
		if (error < min_error){
			min_error = error;
			deepcopy(&pattern, &good_pattern);
		}

	} while (next_permutation(pattern.begin(), pattern.end()));

	//	�u���p�^�[���𗘗p���āA�C���f�b�N�X��u������
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
	//	��U�x�N�g���ɂ��܂�
	vector<vector<float>> tmp;
	tmp.resize(CLUSTER_NUM);
	for (int i = 0; i < CLUSTER_NUM; i++){
		tmp[i].resize(P);
		for (int p = 0; p < P; p++){
			tmp[i][p] = src[i*P + p];
		}
	}
	//		x,y,z...���̏��ł��ꂼ��\�[�g����
	//		�K��l�̂��͓̂������̂Ƃ��ăJ�E���g����
	/*
	sort�̒���for���J��Ԃ��Ƃ��܂������Ȃ�
	for (int p = 0; p < 1; p++)
	for (int p = 1; p < 2; p++)���Ƃ��܂������D
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

	//	�\�[�g���ʂ�������
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

	/* vi���ǂ�xk�ƍł��߂������ׂāC���������x���Ƃ��� */
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
		/* �ł��߂����x��(�C���f�b�N�X)���擾���� */
		int index = label[min_k];
		/* ���x���̈ʒu��vi���R�s�[���� */
		for (int p = 0; p < P; p++){
			vi[index*P + p] = _vi[i*P + p];
		}
	}
}

/*
	2D�f�[�^�Z�b�g���쐬����֐�
	@param xk
	@param label �������x��
	@param minValue ��肦��ŏ��l
	@param maxValue ��肦��ő�l
	@param xkSize �f�[�^��
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
	*T = Thigh * exp(-Cd*pow((float)k, 1.0f / D));
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

	//	�N���X�^�����O���Ȃ�
	if (ds[i].finished){
		return;
	}

	float err;
	float Thigh = ds[i].Thigh;
	float T;
	__device_VFA(&T, Thigh, ds[i].t_update_cnt, P, CD);

	//	uik���X�V����
	__device_update_dik(ds[i].dik, ds[i].vi, ds[i].xk, CLUSTER_NUM, DATA_NUM, P);
	__device_update_uik_with_T(ds[i].uik, ds[i].dik, CLUSTER_NUM, DATA_NUM, Q, T);


	//	vi�̃o�b�N�A�b�v�����
	__device_deepcopy(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM*P);

	//	vi(centroids)���X�V����
	__device_update_vi(ds[i].uik, ds[i].xk, ds[i].vi, CLUSTER_NUM, DATA_NUM, P, Q);

	//	�N���X�^�����O�񐔂𑝂₵�Ă���
	ds[i].iterations++;

	//	���ꉷ�x�ł̎����𔻒�
	//	�������Ă��Ȃ���΂��̂܂܂̉��x�ŌJ��Ԃ�
	__device_calc_convergence(ds[i].vi, ds[i].vi_bak, CLUSTER_NUM, P, &err);
	//err= 0; // ���x��������
	if (EPSIRON < err){
		//	���x���������Ɋ֐����I��
		return;
	}

	//	�O�̉��x�Ƃ̎����𔻒�
	//	�������Ă�����I��
	__device_calc_convergence(ds[i].vi, ds[i].Vi_bak, CLUSTER_NUM, P, &err);
	//err = 0; // �I��
	if (err < EPSIRON){
		//	���̎��_�ŃN���X�^�����O���I������
		ds[i].finished = TRUE;
		return;
	}

	//	�o�b�N�A�b�v
	//	���x��������O��vi��ۑ�
	__device_deepcopy(ds[i].vi, ds[i].Vi_bak, CLUSTER_NUM*P);

	//	���x�X�V
	ds[i].t_update_cnt++;

}


//	=====================================================================
//	 �A���x�֐�����
//	====================================================================
#ifdef  PFCM
int main(){

	/* �]���p�ϐ� */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;
	EvalFormat<float> eval_time;  // ���Ԃ�����float�ō쐬
	MyTimer timer;

	/* �N���X�^�����O�p�̔���p�ӂ��� */
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

	/* �f�[�^���[�h */
	if (load_dataset(DS_FILENAME, _xk,  P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}

	/* �������x������ */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}

	/* GPU�Ƀf�[�^��n�� */
	d_xk = h_xk;

	/* FCM�@���J��Ԃ��s���C�땪�ސ�(err)�C�A���x�X�V��(it)�C���s����(time)�̍ŏ��C�ő�C���ϒl�����߂� */
	for (int g_it = 0; g_it < G_IT; g_it++){
		printf("Prosessing %d/%d\n", g_it, G_IT);
		timer.start();

		/* �����N���X�^���S������ */
		make_random(_vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);

		/* ��������܂ŌJ��Ԃ� */
		float T = THIGH;
		float q = Q;
		int it = 0;
		int t_update_count = 0;
		for (; it < 50; it++){
			printf("T=%f Processing... %d/50\n", T, it);

			/* vi���R�s�[ */
			d_vi = h_vi;

			/* dik�̃e�[�u�����쐬���Ă���uik���X�V */
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
			t_update_count++;
			VFA(&T, THIGH, t_update_count, P);
		}

		//	�J��Ԃ��񐔂��L�^
		eval_it.add(it);

		//	�땪�ސ����L�^
		do_labeling(_uik, _result, CLUSTER_NUM, DATA_NUM);
		int err = compare(_label, _result, DATA_NUM, CLUSTER_NUM);
		eval_err.add(err);

		//	���s���Ԃ��L�^
		int ms;
		timer.stop(&ms);
		eval_time.add(ms);

		printf("DONE: Err=%d, Itaration=%d Time=%d[ms]\n", err, it, ms);
	}

	/* ���ʂ̕\�� */
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
//	 ���x����
//	====================================================================
#ifdef TPFCM
int main(){

	/* �]���p�ϐ� */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;

	/* �N���X�^�����O�p�̔���p�ӂ��� */
	int _label[DATA_NUM];
	int _result[DATA_NUM];
	float _xk[DATA_NUM*P];

	thrust::device_vector<DataFormat> d_ds(N);
	thrust::host_vector<DataFormat> h_ds(N);

#ifdef RANDOM2D
	/* �f�[�^���[�h����߂ă����_������ */
	make_2d_random(_xk, _label, INIT_RAND_MIN, INIT_RAND_MAX, DATA_NUM);
#else
	/* �f�[�^���[�h */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}
	/* �������x������ */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}
#endif

	/* ���̏����� */
	/* TODO: Tj�̌�����@ */
	for (int i = 0; i < N; i++){
		h_ds[i].Thigh = pow(TMAX, (i + 1.0f - N / 2.0f) / (N / 2.0f));
		h_ds[i].iterations = 0;
		make_random(h_ds[i].vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
		deepcopy(_xk, h_ds[i].xk, DATA_NUM*P);
		h_ds[i].finished = FALSE;
		h_ds[i].t_update_cnt = 0;
	}

	/* �e�X���b�h��FCM�����s */
	for (int it=0; it< IT; it++){
		// device_pre_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
		d_ds = h_ds;
		device_TPFCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
		cudaDeviceSynchronize();
		h_ds = d_ds;

		/* �����ŃG���g���s�[�ƖړI�֐������߂� */
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

	/* ���ʂ��L�^ */
	int best_it = INT_MAX;
	int best_error = INT_MAX;
	FILE *fp_result = fopen("T_it_err.txt", "w");

	for (int n = 0; n < N; n++){
		int it = h_ds[n].iterations;
		do_labeling(h_ds[n].uik, _result, CLUSTER_NUM, DATA_NUM);
		int err = compare(_label, _result, DATA_NUM, CLUSTER_NUM);
		
		//	�ŗǉ��̌땪�ސ����擾
		if (it<best_it){
			best_it = it;
			best_error = err;
		}

		//	�J��Ԃ��񐔂��L�^
		eval_it.add(it);

		//	�땪�ސ����L�^
		eval_err.add(err);

		//	�G���g���s�[�ƖړI�֐��̕ω����o��
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

		//	�������x�C�J��Ԃ���, �땪�ސ��̃y�A���o��
		fprintf(fp_result, "%.4f,%d,%d\n", h_ds[n].Thigh, it, err);

	}
	fclose(fp_result);


	/* ���ʂ̕\�� */
	int min, max;
	float ave, cov;
	float fmin, fmax;
	printf("Mathod: ���x���񉻃A�j�[�����O\n");
	eval_it.statics(&min, &max, &ave, &cov);
	printf("Update Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_err.statics(&min, &max, &ave, &cov);
	printf("Error Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	printf("Best It:%d, Best Error:%d\n", best_it, best_error);
	return 0;
}
#endif

//	=====================================================================
//	 ���x����
// �J��Ԃ��o�[�W����
//	====================================================================
#ifdef TPFCM2
int main(){

	/* �]���p�ϐ� */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;

	/* �N���X�^�����O�p�̔���p�ӂ��� */
	int _label[DATA_NUM];
	int _result[DATA_NUM];
	float _xk[DATA_NUM*P];

	thrust::device_vector<DataFormat> d_ds(N);
	thrust::host_vector<DataFormat> h_ds(N);

#ifdef RANDOM2D
	/* �f�[�^���[�h����߂ă����_������ */
	make_2d_random(_xk, _label, INIT_RAND_MIN, INIT_RAND_MAX, DATA_NUM);
#else
	/* �f�[�^���[�h */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}
	/* �������x������ */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}
#endif

	for(int g_it=0; g_it < G_IT; g_it++){
		printf("g_it=%d\n", g_it);
		
		/* ���̏����� */
		/* TODO: Tj�̌�����@ */
		for (int i = 0; i < N; i++){
			h_ds[i].Thigh = pow(TMAX, (i + 1.0f - N / 2.0f) / (N / 2.0f));
			h_ds[i].iterations = 0;
			make_random(h_ds[i].vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			deepcopy(_xk, h_ds[i].xk, DATA_NUM*P);
			h_ds[i].finished = FALSE;
			h_ds[i].t_update_cnt = 0;
		}

		/* �e�X���b�h��FCM�����s */
		d_ds = h_ds;
		for (int it = 0; it< 100; it++){
			// device_pre_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
			device_TPFCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
			cudaDeviceSynchronize();
		}
		h_ds = d_ds;

		/* ���ʂ��L�^ */
		int best_it = INT_MAX;
		int best_error = INT_MAX;
		for (int n = 0; n < N; n++){
			int it = h_ds[n].iterations;
			do_labeling(h_ds[n].uik, _result, CLUSTER_NUM, DATA_NUM);
			int err = compare(_label, _result, DATA_NUM, CLUSTER_NUM);

			//	�ŗǉ��̌땪�ސ����擾
			if (it<best_it){
				best_it = it;
				best_error = err;
			}
		}

		//	�J��Ԃ��񐔂��L�^
		eval_it.add(best_it);

		//	�땪�ސ����L�^
		eval_err.add(best_error);


	}

	/* ���ʂ̕\�� */
	int min, max;
	float ave, cov;
	float fmin, fmax;
	printf("Mathod: ���x���񉻃A�j�[�����O(1000�񕽋�)\n");
	eval_it.statics(&min, &max, &ave, &cov);
	printf("Update Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	eval_err.statics(&min, &max, &ave, &cov);
	printf("Error Count: {min: %d, max: %d, ave: %.3f, cov: %.3f}\n", min, max, ave, cov);
	return 0;
}
#endif

//	=====================================================================
//	 �]����@DA-FCM
//	====================================================================
#ifdef  DA_FCM
int main(){

	/* �]���p�ϐ� */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;
	EvalFormat<float> eval_time;  // ���Ԃ�����float�ō쐬

	/* �N���X�^�����O�p�̔���p�ӂ��� */
	float _Vi_bak[CLUSTER_NUM*P];
	float _vi_bak[CLUSTER_NUM*P];
	float _vi[CLUSTER_NUM*P];
	float _uik[DATA_NUM*CLUSTER_NUM];
	float _xk[DATA_NUM*P];
	float _dik[DATA_NUM*CLUSTER_NUM];
	int _label[DATA_NUM];
	int _result[DATA_NUM];
	MyTimer timer;

	/* �f�[�^���[�h */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}

	/* �������x������ */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}

	/* �n�C�u���b�h�@���J��Ԃ��s���C�땪�ސ�(err)�C�A���x�X�V��(it)�C���s����(time)�̍ŏ��C�ő�C���ϒl�����߂� */
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
			t_update_count++;
			VFA(&T, THIGH, t_update_count, P, CD);
		}

		//	�J��Ԃ��񐔂��L�^
		eval_it.add(it);

		//	�땪�ސ����L�^
		do_labeling(_uik, _result, CLUSTER_NUM, DATA_NUM);
		int err = compare(_label, _result, DATA_NUM);
		eval_err.add(err);

		//	���s���Ԃ��L�^
		int ms;
		timer.stop(&ms);
		eval_time.add(ms);

	}


	/* ���ʂ̕\�� */
	
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
//	 �n�C�u���b�h�@
//	====================================================================
#ifdef HYBRID
int main(){

	/* �]���p�ϐ� */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;
	EvalFormat<int> eval_phase1_it;
	EvalFormat<float> eval_time;  // ���Ԃ�����float�ō쐬

	/* �N���X�^�����O�p�̔���p�ӂ��� */
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
	/* �f�[�^���[�h����߂ă����_������ */
	make_2d_random(_xk, _label, INIT_RAND_MIN, INIT_RAND_MAX, DATA_NUM);
#else
	/* �f�[�^���[�h */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}
	/* �������x������ */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}
#endif

	d_xk = h_xk;

	/* �n�C�u���b�h�@���J��Ԃ��s���C�땪�ސ�(err)�C�A���x�X�V��(it)�C���s����(time)�̍ŏ��C�ő�C���ϒl�����߂� */
	for (int g_it = 0; g_it < G_IT; g_it++){
		printf("Prosessing %d/%d\n", g_it, G_IT);
		timer.start();

		/* T_base������ */
		float Tbase = 0.0;
		for (int i = 0; i < 1000; i++){
			make_random(_vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			float L1k_bar = calc_L1k(_xk, _vi, CLUSTER_NUM, DATA_NUM, P);
			Tbase += CLUSTER_NUM / L1k_bar;
		}
		Tbase = 1000 / Tbase;
		printf("Tbase=%f\n", Tbase);
		

		/* ���̏����� */
		/* TODO: Tj�̌�����@ */
		for (int i = 0; i < N; i++){
			h_ds[i].Thigh = pow(Tbase, (i + 1.0f - N / 2.0f) / (N / 2.0f)) + Tbase;
			h_ds[i].iterations = 0;
			make_random(h_ds[i].vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			deepcopy(_xk, h_ds[i].xk, DATA_NUM*P);
			h_ds[i].finished = FALSE;
		}

		/* �e�X���b�h�Ńt�F�[�Y1�����s */
		for (int i = 0; i <40; i++){
			d_ds = h_ds;
			device_pre_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
			cudaDeviceSynchronize();
			h_ds = d_ds;
		}

		/* ���ϒl�𗘗p���Ē��S������ */
		/* h_ds[0].vi����l�Ƃ��āC�ł��������߂��N���X�^�̕��ϒl�𗘗p���邱�ƂƂ��邯�ǂ���ł����̂��H */
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

		/* �R�R�܂łɂ��������J��Ԃ���(�ő�l)���v�Z���� */
		int phase1_max_iteration = 0;
		for (int n = 0; n < N; n++){
			printf("%d ", h_ds[n].iterations);
			phase1_max_iteration = MAX(phase1_max_iteration, h_ds[n].iterations);
		}
		printf("\n");
		

		/* �t�F�[�Y2�����s */
		float T = Tbase;
		float q = Q;
		int it = 0;
		int t_update_count = 0;
		for (; it < 50; it++){
			printf("T=%f Processing... %d/50\n", T, it);

			/* vi���R�s�[ */
			d_vi = h_vi;

			/* dik�̃e�[�u�����쐬���Ă���uik���X�V */
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
			t_update_count++;
			VFA(&T, Tbase, t_update_count, P);
		}

		//	�J��Ԃ��񐔂��L�^
		eval_it.add(it);

		//	�t�F�[�Y1�ł̍X�V�񐔂��������Čv�Z���Ȃ���
		phase1_max_iteration += it;
		eval_phase1_it.add(phase1_max_iteration);

		//	�땪�ސ����L�^
		do_labeling(_uik, _result, CLUSTER_NUM, DATA_NUM);
		int err = compare(_label, _result, DATA_NUM, CLUSTER_NUM);
		eval_err.add(err);

		//	���s���Ԃ��L�^
		int ms;
		timer.stop(&ms);
		eval_time.add(ms);
		printf("DONE: Err=%d, Itaration=%d Time=%d[ms]\n", err, it, ms);

	}


	/* ���ʂ̕\�� */
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

	/* �N���X�^�����O�p�̔���p�ӂ��� */
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
	/* �f�[�^���[�h����߂ă����_������ */
	make_2d_random(_xk, _label, INIT_RAND_MIN, INIT_RAND_MAX, DATA_NUM);
#else
	/* �f�[�^���[�h */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}
	/* �������x������ */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}
#endif

	d_xk = h_xk;

	/* �n�C�u���b�h�@���J��Ԃ��s���C�땪�ސ�(err)�C�A���x�X�V��(it)�C���s����(time)�̍ŏ��C�ő�C���ϒl�����߂� */
	for (int g_it = 0; g_it < G_IT; g_it++){
		printf("Prosessing %d/%d\n", g_it, G_IT);
		timer.start();

		/* T_base������ */
		float Tbase = 0.0;
		for (int i = 0; i < 1000; i++){
			make_random(_vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			float L1k_bar = calc_L1k(_xk, _vi, CLUSTER_NUM, DATA_NUM, P);
			Tbase += CLUSTER_NUM / L1k_bar;
		}
		Tbase = 1000 / Tbase;
		printf("Tbase=%f\n", Tbase);

		/* ���̏����� */
		/* TODO: Tj�̌�����@ */
		for (int i = 0; i < N; i++){
			h_ds[i].Thigh = pow(Tbase, (i + 1.0f - N / 2.0f) / (N / 2.0f)) + Tbase;
			h_ds[i].iterations = 0;
			make_random(h_ds[i].vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			deepcopy(_xk, h_ds[i].xk, DATA_NUM*P);
			h_ds[i].finished = FALSE;
		}

		/* �e�X���b�h�Ńt�F�[�Y1�����s */
		for (int i = 0; i < 40; i++){
			d_ds = h_ds;
			device_pre_FCM << <1, N >> >(thrust::raw_pointer_cast(d_ds.data()));
			cudaDeviceSynchronize();
			h_ds = d_ds;
		}

		/* vji���o�͂��� */
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

		/* ���ϒl�𗘗p���Ē��S������ */
		/* h_ds[0].vi����l�Ƃ��āC�ł��������߂��N���X�^�̕��ϒl�𗘗p���邱�ƂƂ��邯�ǂ���ł����̂��H */
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
		
		/* ���ϒl���o�͂��� */
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
//	 Thigh��������@
//	TODO: ����
//	====================================================================
#ifdef AUTO_THIGH
int main(){

	/* �]���p�ϐ� */
	EvalFormat<int> eval_err;
	EvalFormat<int> eval_it;
	EvalFormat<float> eval_time;  // ���Ԃ�����float�ō쐬

	/* �N���X�^�����O�p�̔���p�ӂ��� */
	float _Vi_bak[CLUSTER_NUM*P];
	float _vi_bak[CLUSTER_NUM*P];
	float _vi[CLUSTER_NUM*P];
	float _uik[DATA_NUM*CLUSTER_NUM];
	float _xk[DATA_NUM*P];
	float _dik[DATA_NUM*CLUSTER_NUM];
	int _label[DATA_NUM];
	int _result[DATA_NUM];
	MyTimer timer;

	/* �f�[�^���[�h */
	if (load_dataset(DS_FILENAME, _xk, P, DATA_NUM) != 0){
		fprintf(stderr, "LOAD FAILED.");
		exit(1);
	}

	/* �������x������ */
	for (int i = 0; i < DATA_NUM; i++){
		_label[i] = i  * CLUSTER_NUM / DATA_NUM;
	}

	/* Thigh���ς�����J��Ԃ��s���C�땪�ސ�(err)�C�A���x�X�V��(it)�C���s����(time)�̍ŏ��C�ő�C���ϒl�����߂� */
	for (int g_it = 0; g_it < G_IT; g_it++){
		printf("Prosessing %d/%d\n", g_it, G_IT);
		timer.start();

		/* T_base������ */
		float Tbase = 0.0;
		for (int i = 0; i < 1000; i++){
			make_random(_vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);
			float L1k_bar = calc_L1k(_xk, _vi, CLUSTER_NUM, DATA_NUM, P);
			Tbase += CLUSTER_NUM / L1k_bar;
		}
		Tbase = 1000 / Tbase;
		printf("Tbase=%f\n", Tbase);

		make_random(_vi, CLUSTER_NUM*P, INIT_RAND_MIN, INIT_RAND_MAX);

		/* �t�F�[�Y2�����s */
		float T = Tbase;
		float q = Q;
		int it = 0;
		int t_update_count = 0;
		for (; it < 50; it++){
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
			t_update_count++;
			VFA(&T, Tbase, t_update_count, P);
		}

		//	�J��Ԃ��񐔂��L�^
		eval_it.add(it);

		//	�땪�ސ����L�^
		do_labeling(_uik, _result, CLUSTER_NUM, DATA_NUM);
		int err = compare(_label, _result, DATA_NUM);
		eval_err.add(err);

		//	���s���Ԃ��L�^
		int ms;
		timer.stop(&ms);
		eval_time.add(ms);

		printf("DONE: Err=%d, Itaration=%d Time=%d[ms]\n", err, it, ms);
	}


	/* ���ʂ̕\�� */
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

