#ifndef __DATASET_LOADER_H__
#define __DATASET_LOADER_H__

class DatasetLoader{
public:
	DatasetLoader(){

	}

	virtual void load(std::vector<float> &xk, std::vector<float> &vi){}

protected:
	
	float __random(float min, float max){
		return min + (float)(rand() * (max - min) / RAND_MAX);
	}



};




#endif