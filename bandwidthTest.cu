/*
並列グラフ探索を利用したバージョン
中央集権型 - プロセッサ間でノードのopen listを共有メモリ上に共有する
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

#define N 32
#define M 20

typedef struct{
	int a[20];
}DataStruct;

__global__ void test_code(DataStruct *arg){
	int i = threadIdx.x;
	for (int j = 0; j < M; j++){
		arg[i].a[j] = j;
	}

}


int main(void){
	
	thrust::host_vector<DataStruct> host(N);
	thrust::device_vector<DataStruct> device(N);
	device = host;

	test_code<<<1, N>>>(thrust::raw_pointer_cast(device.data()));
	cudaDeviceSynchronize();

	host = device;
	
	for (int i = 0; i < N; i++){
		for (int j = 0; j < M; j++){
			printf("%d ", host[i].a[j]);
		}
		printf("\n");
	}

	return 0;
}