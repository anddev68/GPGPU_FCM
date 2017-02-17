/*
    fcm.h
    Functuions for fcm
*/
#ifndef __FCM_H__
#define __FCM_H__

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

/*
    make_datasets()
    min~maxの範囲でランダムにデータセットを作成する
    size = データの数 * 次元で作成すること
*/
void make_datasets(float *xk, int size, float min=0.0, float max=1.0);

/*
    make_sample_sets()
    1dの実験用データを作成する
*/
void make_sample_sets(float *xk, int size, float min=-1.0, float max=1.0);

/*
	make_iris_datasets()
	irisのデータセットで作成する
*/
int make_iris_datasets(float *xk, int kSize, int pSize);

/*
    make_first_centroids()
    ランダムにクラスタ中心を設定する
*/
void make_first_centroids(float *xk, int size, float min=0.0, float max = 1.0);

/*
	sampleとtargetを比較し、誤って分類した数を出力する
*/
int compare(int *target, int *sample, int size);

/*
    make_iris_150_targets()
    iris50x3用の正解分類ファイルを作成する
*/
void make_iris_150_targes(int *targets);


/*
    ファイルにuikを書き出す
*/
void fprintf_uik(FILE* fp, float *uik, int iSize, int kSize);

/*
	xkを書き出す
*/
void fprintf_xk(FILE* fp, float *xk, int kSize, int pSize);

/*
	ファイルにresultsを書き出す
*/
void fprintf_results(FILE *fp, int *results, int size);






#endif