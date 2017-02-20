#ifndef __PFCM_H__
#define __PFCM_H__

/*

	PFCM.h
	Functions for parallel FCM

*/

#include <stdio.h>


/*
	エラー遷移配列を出力する
*/
void fprintf_error(FILE *fp, int *v, int size);

/*
	温度遷移配列を出力する
*/
void fprintf_T(FILE *fp, float *v, int size);

/*
	目的関数の遷移配列を出力する
*/
void fprintf_objfunc(FILE *fp, float *v, int size);






#endif