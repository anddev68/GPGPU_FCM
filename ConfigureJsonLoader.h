#ifndef __CONFIGURE_JSON_LOADER_H_
#define __CONFIGURE_JSON_LOADER_H_

#include <string>
#include "picjson.h"

class ConfigureJsonLoader{
public:
	
	static const std::string DATASET_RANDOM;
	static const std::string DATASET_IRIS;

	/*
		設定ファイルを解析してFCMの設定を行う．
	*/
	ConfigureJsonLoader(std::string fileName){
		std::ifstream fs;
		fs.open(fileName, std::ios::binary);
		picojson::value val;
		fs >> val;
		fs.close();

		auto tmp = val.get<picojson::object>();
		this->dataset = tmp["dataset"].get<std::string>();
		this->parallel = tmp["parallel"].get<bool>();

		//	データセットがランダムの場合は細かい設定を読み込む
		if (this->dataset == DATASET_RANDOM){
			this->dataN = (int)tmp["random"].get<picojson::object>()["N"].get<double>();
			this->dataP = (int)tmp["random"].get<picojson::object>()["P"].get<double>();
			this->clusterC = (int)tmp["random"].get<picojson::object>()["C"].get<double>();
			this->min = (float)tmp["random"].get<picojson::object>()["min"].get<double>();
			this->max = (float)tmp["random"].get<picojson::object>()["max"].get<double>();
		}
		//	データセットがIRISの場合はIRISを指定
		else if(this->dataset == DATASET_IRIS){
			this->dataN = 150;
			this->dataP = 4;
			this->clusterC = 3;
		}
		else{
			fprintf(stderr, "パラメータ設定が間違っています\n");
			exit(-1);
		}
		
	}

	bool parallel;
	std::string dataset;

	int dataN;	// データセットサイズ
	int dataP;		//	データ次元数
	int clusterC;	//	クラスタサイズ
	float min;
	float max;

};

const std::string ConfigureJsonLoader::DATASET_RANDOM = "random";
const std::string ConfigureJsonLoader::DATASET_IRIS = "iris";



#endif