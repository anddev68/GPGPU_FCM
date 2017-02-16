/*
    fcm.h
    Functuions for fcm
*/
#ifndef __FCM_H__
#define __FCM_H__

#include <stdio.h>


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
    make_first_centroids()
    ランダムにクラスタ中心を設定する
*/
void make_first_centroids(float *xk, int size, float min=0.0, float max = 1.0);


/*
    make_iris_150_targets()
    iris50x3用の正解分類ファイルを作成する
*/
void make_iris_150_targes(int *targets);


/*
    ファイルにuikを書き出す
*/
void fprintf_uik(FILE* fp, float *uik, int iSize, int kSize);







#endif