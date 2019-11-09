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
	std::vector<std::pair<std::vector<float>, std::string > > trainData;
	std::vector<float>goal;
	for(int i = 0; i < dataColletcion.size(); i++){
		vector_feature feature = dataColletcion[i];
		for(int j = 0; j < feature.size(); j++){
			trainData.push_back(std::make_pair(feature[j].second.featureFace, feature[j].first));		
		}
	}
	KDtreeNode *kdtree = new KDtreeNode;
	buildKdtree(kdtree, trainData);
#endif
#if 1
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
					#if 1 //loop
					encodeFeature detFeature = result[ii].faceFeature;
					std::pair<float, std::string>nearestNeighbor= serachCollectDataNameByloop(dataColletcion,
             															detFeature, result[ii].faceAttri.gender);
					person = nearestNeighbor.second;
					#else
					goal = result[ii].faceFeature.featureFace;
					std::pair<std::vector<float>, std::string > nearestNeighbor = searchNearestNeighbor(goal, kdtree);
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
