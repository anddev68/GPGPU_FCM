#ifndef __LOGGER_H__
#define __LOGGER_H__

/*
	Logger.h
	�ėp�o�͊֐�
*/

#include <stdio.h>

/*
	A��B��g�ɂ��ďo�͂���
*/
void fprintf_pair_f(FILE *fp, float *A, float *B, int size, char div=',');
void fprintf_pair_d(FILE *fp, int *A, int *B, int size, char div=',');
void fprintf_pair_df(FILE *fp, int *A, float *B, int size, char div = ',');

#endif