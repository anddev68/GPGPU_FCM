#pragma once

template <typename T>
class CpuGpuData {
public:
	
	CpuGpuData(const int iSize){
		cudaMallocManaged(&m_data, sizeof(T)*iSize);
		m_size = iSize;
	}


	~CpuGpuData(){
		cudaFree(m_data);
	}

	void zero_clear(){
		for (int i = 0; i < m_size; i++){
			m_data[i] = 0;
		}
	}

	void dump(){
		for (int i = 0; i < m_size; i++){
			printf("%.2f ", m_data[i]);
		}
	}


	T* m_data;
	int m_size;
};


template <typename T>
void array_copy(CpuGpuData<T> *src, CpuGpuData<T> *dst){
	for (int i = 0; i < src->m_size; i++){
		dst->m_data[i] = src->m_data[i];
	}
}

