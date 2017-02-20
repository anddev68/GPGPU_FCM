#ifndef __LOGGER_H__
#define __LOGGER_H__

/*
	Logger.h
	汎用出力関数
*/

#include <stdio.h>

/*
	AとBを組にして出力する
*/
void fprintf_pair_f(FILE *fp, float *A, float *B, int size, char div=',');
void fprintf_pair_d(FILE *fp, int *A, int *B, int size, char div=',');
void fprintf_pair_df(FILE *fp, int *A, float *B, int size, char div = ',');

#endif