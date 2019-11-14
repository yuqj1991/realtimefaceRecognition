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
#include "ms_kdtree.hpp"
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
#endif
/****************测试******************kd二叉树获取方式中时间节省方式********************/
	std::vector<std::pair<Prediction, std::string > > trainData;
	Prediction goal;
	encodeFeature detFeature;
	int gender = 0;
	for(int i = 0; i < dataColletcion.size(); i++){
		vector_feature feature = dataColletcion[i];
		for(int j = 0; j < feature.size(); j++){
			if(feature[j].first!="1156174130")
				trainData.push_back(std::make_pair(feature[j].second.featureFace, feature[j].first));
			else{
				goal = feature[j].second.featureFace;
				gender = i;
				detFeature = feature[j].second;
			}
			
		}
	}
	KDtreeNode *kdtree = new KDtreeNode;
	buildKdtree(kdtree, trainData, 0);
	clock_t startTime,endTime;
	printKdTree(kdtree, 0);
 	startTime = clock();//计时开始
	std::pair<float, std::string > nearestNeighbor = searchNearestNeighbor(goal, kdtree);
	endTime = clock();//计时结束
	std::cout << "kd tree run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    std::cout<<"the kd method result: "<<nearestNeighbor.second<<std::endl;
/****************测试循环获取方式中时间节省方式*****************************************/
	startTime = clock();//计时开始
	std::pair<float, std::string>nearestNeighbor_loop= serachCollectDataNameByloop(dataColletcion,
             															detFeature, gender);
	std::string person = nearestNeighbor_loop.second;
	std::cout<<"the loop method result: "<<person<<std::endl;
	endTime = clock();//计时结束
	std::cout << "for recusive run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
/****************测试***************************map时间节省方式**********************/
    mapFaceCollectDataSet dataTestSet;
	FaceBase::iterator it;
    for(it = dataColletcion.begin(); it != dataColletcion.end(); it++){
        vector_feature feature = it->second;
        mapFeature subfeature;
        for(int j = 0; j < feature.size(); j++){
            subfeature.insert(std::make_pair(feature[j].second, feature[j].first));
        }
        int gender = it->first;
        dataTestSet[gender] = subfeature;
    }
    int num = 0;
    mapFaceCollectDataSet::iterator iter;
    for(iter = dataTestSet.begin(); iter != dataTestSet.end(); iter++){
        mapFeature subfeature = iter->second;
        num += subfeature.size();
    }
    std::cout<<"num: "<<num<<std::endl;
    startTime = clock();//计时开始
    person = serachCollectDataNameBymapSet(dataTestSet,
             detFeature, gender);
    endTime = clock();//计时结束
	std::cout << "map method run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    std::cout<<"the third method result: "<<person<<std::endl;
/**************************************************************************/

}