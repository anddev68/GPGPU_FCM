#ifndef __RANDOM_LOADER_H__
#define __RANDOM_LOADER_H__

#include "DatasetLoader.h"
#include <math.h>

class RandomLoader : public DatasetLoader{
public:
	RandomLoader(int kSize, int cSize, int pSize, float min, float max){
		this->kSize = kSize;
		this->cSize = cSize;
		this->pSize = pSize;
		this->max = max;
		this->min = min;
	}

	void load(std::vector<float> &xk, std::vector<float> &vi) override{
		//	xk‚ğì¬
		for (int k = 0; k < kSize; k++){
			for (int p = 0; p < pSize; p++){
				float tmp = __random(min, max);
				xk.push_back(tmp);
			}
		}

		for (int c = 0; c < cSize; c++){
			for (int p = 0; p < pSize; p++){
				float r = __random(min, max);
				vi.push_back(r);
			}
		}

	}

private:
	float max;
	float min;
	int kSize, cSize, pSize;

};


#endif