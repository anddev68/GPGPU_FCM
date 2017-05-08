#ifndef __IRIS_LOADER_H__
#define __IRIS_LOADER_H__

#include <vector>
#include <stdio.h>
#include <string>
#include "DatasetLoader.h"

class IrisLoader : public DatasetLoader{
public:

	IrisLoader(std::string fileName){
		srand((unsigned)time(NULL));
		this->fileName = fileName;
	}

	void load(std::vector<float> &xk, std::vector<float> &vi) override{
		//	xk‚ðì¬
		FILE *fp = fopen(this->fileName.c_str(), "r");
		int kSize = 150;
		int pSize = 4;
		int cSize = 3;
		float min = 0.0;
		float max = 1.0;
		for (int k = 0; k < kSize; k++){
			for (int p = 0; p < pSize; p++){
				float tmp;
				fscanf(fp, "%f", &tmp);
				//xk[k * pSize + p] = tmp;
				xk.push_back(tmp);
			}
			char buf[32];
			fscanf(fp, "%s", buf);
		}
		fclose(fp);

		for (int c = 0; c < cSize; c++){
			for (int p = 0; p < pSize; p++){
				float r = __random(min, max);
				vi.push_back(r);
			}
		}

	}



private:




	std::string fileName;

};


#endif