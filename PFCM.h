#ifndef __PFCM_H__
#define __PFCM_H__

/*

	PFCM.h
	Functions for parallel FCM

*/

#include <stdio.h>


/*
	�G���[�J�ڔz����o�͂���
*/
void fprintf_error(FILE *fp, int *v, int size);

/*
	���x�J�ڔz����o�͂���
*/
void fprintf_T(FILE *fp, float *v, int size);

/*
	�ړI�֐��̑J�ڔz����o�͂���
*/
void fprintf_objfunc(FILE *fp, float *v, int size);






#endif