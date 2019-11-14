#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "faceAnalysis.hpp"
#include "kcftracker.hpp"
#include "utils_config.hpp"
#include "dataBase.hpp"
#include "kdtree.hpp"
#include<ctime>

using namespace cv;
using namespace RESIDEO;

int main(int argc, char* argv[]){
	faceAnalysis faceInfernece;

	dataBase baseface(faceDir, facefeaturefile);
#if 0
	baseface.generateBaseFeature(faceInfernece);
#else
	FaceBase dataColletcion = baseface.getStoredDataBaseFeature(facefeaturefile);
	FaceBase dataSubset;

	std::map<int, KDtype >trainData;
	Prediction goal;
	encodeFeature detFeature;
	int gender = 0;
	int goal_gender = 0;
	FaceBase::iterator iter;
	for(iter = dataColletcion.begin(); iter != dataColletcion.end(); iter++){
		gender = iter->first;
		vector_feature feature = iter->second;
		for(int j = 0; j < feature.size(); j++){
			if(feature[j].first!="1156174130"){
				if(trainData.find(gender) == trainData.end()){
					KDtype new_feature;
					new_feature.push_back(std::make_pair(feature[j].second.featureFace, feature[j].first));
					trainData.insert(std::make_pair(gender, new_feature));
            	}else{
					KDtype feature_list = trainData.find(gender)->second;
					feature_list.push_back(std::make_pair(feature[j].second.featureFace, feature[j].first));
					trainData[gender] = feature_list;
            	}
				/*******/
				if(dataSubset.find(gender) == dataSubset.end()){
					vector_feature new_feature;
					new_feature.push_back(std::make_pair(feature[j].first, feature[j].second));
					dataSubset.insert(std::make_pair(gender, new_feature));
            	}else{
					vector_feature feature_list = dataSubset.find(gender)->second;
					feature_list.push_back(std::make_pair(feature[j].first, feature[j].second));
					dataSubset[gender] = feature_list;
            	}
				/*******/
			}else{
				goal = feature[j].second.featureFace;
				gender = iter->first;
				goal_gender = iter->first;
				detFeature = feature[j].second;
			}
			
		}
	}

	
/****************测试二叉树检索方式**********************************************/
	KDtreeNode *male_kdtree = new KDtreeNode;
	KDtreeNode *female_kdtree = new KDtreeNode;
	buildKdtree(male_kdtree, trainData.find(0)->second, 0);
	buildKdtree(female_kdtree, trainData.find(1)->second, 0);
	std::cout<<"build tree end"<<std::endl;
	clock_t startTime,endTime;
	//printKdTree(kdtree, 0);
 	startTime = clock();//计时开始
	std::pair<float, std::string > nearestNeighbor;
	if(goal_gender == 0){
		nearestNeighbor = searchNearestNeighbor(goal, male_kdtree);
	}else{
		nearestNeighbor = searchNearestNeighbor(goal, female_kdtree);
	}
	endTime = clock();//计时结束
	std::cout << "kd tree run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    std::cout<<"the kd method result: "<<nearestNeighbor.second<<std::endl;


/****************测试循环检索方式**********************************************/
	startTime = clock();//计时开始
	std::pair<float, std::string>nearestNeighbor_loop= serachCollectDataNameByloop(dataSubset,
             															detFeature, goal_gender);
	std::string person = nearestNeighbor_loop.second;
	endTime = clock();//计时结束
	std::cout << "loop run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	std::cout<<"the loop method result: "<<person<<std::endl;


/****************测试map方式*************************************************/
    mapFaceCollectDataSet dataTestSet;
	
    for(iter = dataSubset.begin(); iter != dataSubset.end(); iter++){
        vector_feature feature = iter->second;
        mapFeature subfeature;
        for(int j = 0; j < feature.size(); j++){
            subfeature.insert(std::make_pair(feature[j].second, feature[j].first));
        }
        gender = iter->first;
        dataTestSet[gender] = subfeature;
    }
    int num = 0;
    mapFaceCollectDataSet::iterator itermap;
    for(itermap = dataTestSet.begin(); itermap != dataTestSet.end(); itermap++){
        mapFeature subfeature = itermap->second;
        num += subfeature.size();
    }
    std::cout<<"num: "<<num<<std::endl;
    startTime = clock();//计时开始
    person = serachCollectDataNameBymapSet(dataTestSet,
             detFeature, goal_gender);
    endTime = clock();//计时结束
	std::cout << "map method run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    std::cout<<"the third method result: "<<person<<std::endl;
/**************************************************************************/
#endif
}