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

mapFaceCollectDataSet getmapDatafaceBase(FaceBase &dataColletcion){
	mapFaceCollectDataSet dataTestSet;
	FaceBase::iterator it;
	std::cout<<"************************"<<std::endl;
	for(it = dataColletcion.begin(); it != dataColletcion.end(); it++){
		int gender = it->first;
		vector_feature feature = it->second;
		for(int i = 0; i < feature.size(); i++){

		}
		mapFeature subfeature;
		
		if(dataTestSet.find(gender) == dataTestSet.end()){
			for(int j = 0; j < feature.size(); j++){
				std::cout<<"gender: "<<gender<<" j: "<<j<<std::endl;
				subfeature.insert(std::make_pair(feature[j].second, feature[j].first));
			}
			std::cout<<"feature size: "<<subfeature.size()<<std::endl;
			dataTestSet.insert(std::make_pair(gender, subfeature));
		}
	}
	int num = 0;
	mapFaceCollectDataSet::iterator iter;
	for(iter = dataTestSet.begin(); iter != dataTestSet.end(); iter++){
		mapFeature subfeature = iter->second;
		num += subfeature.size();
	}
	std::cout<<"map num: "<<num<<std::endl;
	return dataTestSet;
}
/************************以上测试map*********************************/
int main(int argc, char* argv[]){
	faceAnalysis faceInfernece;

	dataBase baseface(faceDir, facefeaturefile);
#if 0
	baseface.generateBaseFeature(faceInfernece);
#else
	FaceBase dataColletcion = baseface.getStoredDataBaseFeature(facefeaturefile);
	std::map<int, KDtype >trainData;
	FaceBase::iterator iter;
	int gender = 0;
	for(iter = dataColletcion.begin(); iter != dataColletcion.end(); iter++){
		gender = iter->first;
		vector_feature feature = iter->second;
		for(int j = 0; j < feature.size(); j++){
			if(trainData.find(gender) == trainData.end()){
				KDtype new_feature;
				new_feature.push_back(std::make_pair(feature[j].second.featureFace, feature[j].first));
				trainData.insert(std::make_pair(gender, new_feature));
			}else{
				KDtype feature_list = trainData.find(gender)->second;
				feature_list.push_back(std::make_pair(feature[j].second.featureFace, feature[j].first));
				trainData[gender] = feature_list;
			}			
		}
	}
	KDtreeNode *male_kdtree = new KDtreeNode;
	KDtreeNode *female_kdtree = new KDtreeNode;
	buildKdtree(male_kdtree, trainData.find(0)->second, 0);
	buildKdtree(female_kdtree, trainData.find(1)->second, 0);

	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
    /**********************初始化跟踪******************/
	Mat frame;
	Rect result;
	int FrameIdx = 0;
	int nFrames = 0;
	VideoCapture cap(0);  
    if(!cap.isOpened())  
    {  
        return -1;  
    }
    bool stop = false;
	RecognResultTrack resutTrack;
    while(!stop)  
    {  
        cap>>frame;
		int width = frame.cols;
		int height = frame.rows;
		int nDataBaseSize = 0;
		if(FrameIdx % 1 == 0){
			resutTrack.clear();
			std::vector<faceAnalysisResult> result= faceInfernece.faceInference(frame, detMargin, 20.0f);
			string person = "unknown man";
			
			for(int ii = 0; ii < result.size(); ii++){
				if(result[ii].haveFeature){
					/*
					detBoxInfo trackBoxInfo;
					trackBoxInfo.detBox.xmin = xmin;
					trackBoxInfo.detBox.xmax = xmax;
					trackBoxInfo.detBox.ymin = ymin;
					trackBoxInfo.detBox.ymax = ymax;
					resutTrack.push_back(trackBoxInfo);//获取跟踪信息
					*/ 
					encodeFeature detFeature = result[ii].faceFeature;
					#if 0 //loop search
					std::pair<float, std::string>nearestNeighbor= serachCollectDataNameByloop(dataColletcion,
             															detFeature, result[ii].faceAttri.gender);
					person = nearestNeighbor.second;
					#else //kdtree search
					std::pair<float, std::string > nearestNeighbor;
					if(result[ii].faceAttri.gender==0)
						nearestNeighbor = searchNearestNeighbor(detFeature.featureFace, male_kdtree);
					else
					{
						nearestNeighbor = searchNearestNeighbor(detFeature.featureFace, female_kdtree);	
					}
					person = nearestNeighbor.second;
					#endif
					if(nearestNeighbor.first < cosValueThresold){
						person = "unknown man";
					}
				}
				box detBox = result[ii].faceBox;
                cv::rectangle( frame, cv::Point( detBox.xmin, detBox.ymin ), 
											cv::Point( detBox.xmax, detBox.ymax), 
															cv::Scalar( 0, 255, 255 ), 1, 8 );
				cv::putText(frame, person.c_str(), cv::Point( detBox.xmin, detBox.ymin ), 
					cv::FONT_HERSHEY_COMPLEX, 2, Scalar(0, 255, 255), 2, 8, 0);
			}
		}else{
			for(int ii = 0; ii <resutTrack.size(); ii++){//跟踪
				int xMin = resutTrack[ii].detBox.xmin;
				int yMin = resutTrack[ii].detBox.ymin;
				int width_ = resutTrack[ii].detBox.xmax - resutTrack[ii].detBox.xmin;
				int height_ = resutTrack[ii].detBox.ymax - resutTrack[ii].detBox.ymin;
				if (nFrames == 0) {
					tracker.init( Rect(xMin, yMin, width_, height_), frame );
					rectangle( frame, Point( xMin, yMin ), Point( xMin+width_, yMin+height_), 
													Scalar( 0, 255, 255 ), 1, 8 );
				}else{// 更新位置信息
					result = tracker.update(frame);
					rectangle( frame, Point( result.x, result.y ), 
											Point( result.x+result.width, result.y+result.height), 
															Scalar( 0, 255, 255 ), 1, 8 );
				}
			}
			nFrames++;
		}
		FrameIdx++;
		if(FrameIdx == 60){
			FrameIdx = 0;
			nFrames = 0;
		}
		
        imshow("faceRecognition",frame);
        if(waitKey(1) > 0)  
            stop = true;
    }
	cap.release();
#endif
}
