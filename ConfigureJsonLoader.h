#ifndef __CONFIGURE_JSON_LOADER_H_
#define __CONFIGURE_JSON_LOADER_H_

#include <string>
#include "picjson.h"

class ConfigureJsonLoader{
public:
	
	static const std::string DATASET_RANDOM;
	static const std::string DATASET_IRIS;

	/*
		�ݒ�t�@�C������͂���FCM�̐ݒ���s���D
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

		//	�f�[�^�Z�b�g�������_���̏ꍇ�ׂ͍����ݒ��ǂݍ���
		if (this->dataset == DATASET_RANDOM){
			this->dataN = (int)tmp["random"].get<picojson::object>()["N"].get<double>();
			this->dataP = (int)tmp["random"].get<picojson::object>()["P"].get<double>();
			this->clusterC = (int)tmp["random"].get<picojson::object>()["C"].get<double>();
			this->min = (float)tmp["random"].get<picojson::object>()["min"].get<double>();
			this->max = (float)tmp["random"].get<picojson::object>()["max"].get<double>();
		}
		//	�f�[�^�Z�b�g��IRIS�̏ꍇ��IRIS���w��
		else if(this->dataset == DATASET_IRIS){
			this->dataN = 150;
			this->dataP = 4;
			this->clusterC = 3;
		}
		else{
			fprintf(stderr, "�p�����[�^�ݒ肪�Ԉ���Ă��܂�\n");
			exit(-1);
		}
		
	}

	bool parallel;
	std::string dataset;

	int dataN;	// �f�[�^�Z�b�g�T�C�Y
	int dataP;		//	�f�[�^������
	int clusterC;	//	�N���X�^�T�C�Y
	float min;
	float max;

};

const std::string ConfigureJsonLoader::DATASET_RANDOM = "random";
const std::string ConfigureJsonLoader::DATASET_IRIS = "iris";



#endif