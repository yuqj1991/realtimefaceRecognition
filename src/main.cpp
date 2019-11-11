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
	std::vector<std::pair<Prediction, std::string > > trainData;
	for(int i = 0; i < dataColletcion.size(); i++){
		vector_feature feature = dataColletcion[i];
		for(int j = 0; j < feature.size(); j++){
			trainData.push_back(std::make_pair(feature[j].second.featureFace, feature[j].first));		
		}
	}
	KDtreeNode *kdtree = new KDtreeNode;
	buildKdtree(kdtree, trainData);

    /**********************初始化跟踪******************/
	KCFTracker tracker(HOG, FIXEDWINDOW, MULTISCALE, LAB);
	Mat frame;
	/**********************跟踪结果********************/
	Rect result;
	int FrameIdx = 0;
	int nFrames = 0;
	/*********************打开摄像头 ******************/
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
			std::vector<faceAnalysisResult> result= faceInfernece.faceInference(frame, 32, 20.0f);
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
<<<<<<< HEAD
					#if 0
					#if 0 //loop
=======
					#if 1 //loop
>>>>>>> 2e1a88f1da7baa0a53002258c905a1df976910c9
					
					std::pair<float, std::string>nearestNeighbor= serachCollectDataNameByloop(dataColletcion,
             															detFeature, result[ii].faceAttri.gender);
					person = nearestNeighbor.second;
					if(nearestNeighbor.first < cosValueThresold){
						person = "unknown man";
					}
					#else //kdtree
					std::pair<float, std::string > nearestNeighbor = searchNearestNeighbor(detFeature.featureFace, kdtree);
					person = nearestNeighbor.second;
					#endif
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
