#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "faceAnalysis/faceAnalysis.hpp"
#include "utils_config.hpp"
#include "dataBase.hpp"
#include "kdtree.hpp"

#ifdef USE_KCF_TRACKING
#include "kcf/kcftracker.hpp"
#else
#include "sort/tracker.h"
#include <opencv2/opencv.hpp>
#endif

#include <lshbox.h>
#include<ctime>

using namespace cv;
using namespace RESIDEO;

#ifdef LSH_SEARCH
std::vector<lshbox::dataUnit> getlshDataset(FaceBase dataColletcion){
		FaceBase::iterator iter;
		std::vector<lshbox::dataUnit> dataSet;
		for(iter = dataColletcion.begin(); iter != dataColletcion.end(); iter++){
			vector_feature feature = iter->second;
			for(int j = 0; j < feature.size(); j++){
				dataSet.push_back(std::make_pair(feature[j].second.featureFace, feature[j].first));
			}
		}
		return dataSet;
}
#endif
#ifndef USE_KCF_TRACKING
bool getRectsFeature(const cv::Mat& img, DETECTIONS& d){
	std::vector<cv::Mat> mats;
	for(DETECTION_ROW& dbox : d) {
		cv::Rect rc = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
				int(dbox.tlwh(2)), int(dbox.tlwh(3)));
		rc.x -= (rc.height * 0.5 - rc.width) * 0.5;
		rc.width = rc.height * 0.5;
		rc.x = (rc.x >= 0 ? rc.x : 0);
		rc.y = (rc.y >= 0 ? rc.y : 0);
		rc.width = (rc.x + rc.width <= img.cols? rc.width: (img.cols-rc.x));
		rc.height = (rc.y + rc.height <= img.rows? rc.height:(img.rows - rc.y));

		cv::Mat mattmp = img(rc).clone();
		cv::resize(mattmp, mattmp, cv::Size(64, 128));
		Mat dst_gray;
		cvtColor(img, dst_gray, CV_BGR2GRAY);
	
		HOGDescriptor detector(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
		vector<float> descriptor;
		vector<Point> location;
		detector.compute(dst_gray, descriptor, Size(0, 0), Size(0, 0),location);
		for(int j = 0; j < feature_dim; j++){
			dbox.feature[j] = descriptor[j];
		}
	}
	return true;
}

void get_detections(DETECTBOX box,float confidence,DETECTIONS& d){
	DETECTION_ROW tmpRow;
	tmpRow.tlwh = box;//DETECTBOX(x, y, w, h);

	tmpRow.confidence = confidence;
	d.push_back(tmpRow);
}

void postprocess(std::vector<output>& outs,DETECTIONS& d){
	std::vector<int> classIds;
	std::vector<float> confidences;
	std::vector<cv::Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i){
		output box = outs[i];
		int width = box.second.xmax - box.second.xmin;
		int height = box.second.ymax - box.second.ymin;
		get_detections(DETECTBOX(box.second.xmin, box.second.ymin, width, height),box.first,d);
	}
}
#endif

int trackingGap = 60;
util configParam;
/************************main*********************************/
int main(int argc, char* argv[]){
	faceAnalysis faceInfernece;
	dataBase baseface(configParam.faceDir, configParam.facefeaturefile);

#if 0
	#ifdef USE_KCF_TRACKING
	baseface.generateBaseFeature(faceInfernece);
	#else
	baseface.generateBaseHOGFeature(faceInfernece);
	#endif
#else
	#ifdef USE_KCF_TRACKING
	FaceBase dataColletcion = baseface.getStoredDataBaseFeature(configParam.facefeaturefile, 512);
	#else
	FaceBase dataColletcion = baseface.getStoredDataBaseFeature(configParam.HOGfacefeaturefile, 3780);
	#endif
	#ifdef KDTREE_SEARCH
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
	#endif
	#ifdef LSH_SEARCH
	lshbox::featureUnit lshgoal;
	std::vector<lshbox::dataUnit> lshDataSet = getlshDataset(dataColletcion);
	std::string file = "lsh.binary";
	bool use_index = false;
    lshbox::PSD_VECTOR_LSH<float> mylsh;
    if (use_index)
    {
        mylsh.load(file);
    }else{
        lshbox::PSD_VECTOR_LSH<float>::Parameter param;
        param.M = 521;
        param.L = 5;
        param.D = 512;
        param.T = GAUSSIAN;
        param.W = 0.5;
		mylsh.reset(param);
        mylsh.hash(lshDataSet);
        mylsh.save(file);
    }
    lshbox::Metric<float> metric(512, L2_DIST);
	#endif
	/**********************初始化跟踪******************/
	#ifdef USE_KCF_TRACKING
	std::vector<KCFTracker> trackerVector;
	RecognResultTrack resutTrack;
	#else
	tracker DeepSortTracker(configParam.max_cosine_distance, configParam.nn_budget);
	#endif
    /**********************while********************/
	Mat frame;
	int FrameIdx = 0;
	bool stop = false;
	VideoCapture cap(0);  
    if(!cap.isOpened())  
    {  
        return -1;  
    }
    while(!stop)  
    {  
        cap>>frame;
		#if USE_KCF_TRACKING
		if(FrameIdx % trackingGap == 0){
			resutTrack.clear();
			std::vector<faceAnalysisResult> result= faceInfernece.faceInference(frame, configParam.detMargin, 20.0f);
			string person = "unknown man";	
			for(int ii = 0; ii < result.size(); ii++){
				if(result[ii].haveFeature){
					encodeFeature detFeature = result[ii].faceFeature;
					#ifdef KDTREE_SEARCH
					std::pair<float, std::string > nearestNeighbor;
					if(result[ii].faceAttri.gender==0)
						nearestNeighbor = searchNearestNeighbor(detFeature.featureFace, male_kdtree);
					else{
						nearestNeighbor = searchNearestNeighbor(detFeature.featureFace, female_kdtree);	
					}
					person = nearestNeighbor.second;
					if(nearestNeighbor.first > configParam.euclideanValueThresold){
						person = "unknown man";
					}
					#endif
					#ifdef LOOP_SEARCH
					std::pair<float, std::string>nearestNeighbor= configParam.serachCollectDataNameByloop(dataColletcion,
             															detFeature, result[ii].faceAttri.gender);
					person = nearestNeighbor.second;
					if(nearestNeighbor.first < configParam.cosValueThresold){
						person = "unknown man";
					}
					#endif
					#ifdef LSH_SEARCH
					lshgoal = result[ii].faceFeature.featureFace;
					std::pair<float, std::string>nearestNeighbor = mylsh.query(lshgoal, metric, lshDataSet);
					person = nearestNeighbor.second;
					if(nearestNeighbor.first > configParam.euclideanValueThresold){
						person = "unknown man";
					}
					#endif
					#ifdef USE_KCF_TRACKING
					detBoxInfo trackBoxInfo;
					trackBoxInfo.detBox.xmin = result[ii].faceBox.xmin;
					trackBoxInfo.detBox.xmax = result[ii].faceBox.xmax;
					trackBoxInfo.detBox.ymin = result[ii].faceBox.ymin;
					trackBoxInfo.detBox.ymax = result[ii].faceBox.ymax;
					trackBoxInfo.name = person;
					resutTrack.push_back(trackBoxInfo);
					#endif
				}
				#if DEBUG
				box detBox = result[ii].faceBox;
                cv::rectangle( frame, cv::Point( detBox.xmin, detBox.ymin ), 
											cv::Point( detBox.xmax, detBox.ymax), 
															cv::Scalar( 0, 255, 255 ), 1, 8 );
				cv::putText(frame, person.c_str(), cv::Point( detBox.xmin, detBox.ymin ), 
					cv::FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2, 8, 0);
				cv::Mat roiImage = frame(cv::Rect(detBox.xmin, detBox.ymin, detBox.xmax - detBox.xmin, detBox.ymax - detBox.ymin));
				for(unsigned i = 0; i < 5; i++){
					cv::circle(roiImage, cv::Point(result[ii].faceAttri.landmarks[i].point_x, result[ii].faceAttri.landmarks[i].point_y)
								, 3, cv::Scalar(0, 0, 213), -1);
				}
				std::string title = configParam.labelGender[result[ii].faceAttri.gender] + std::string(", ") + configParam.labelGlass[result[ii].faceAttri.glass];
				cv::putText(frame, title, cv::Point( detBox.xmin + 40, detBox.ymin + 40 ), 
					cv::FONT_ITALIC, 0.6, Scalar(0, 255, 0), 1);
				#endif
			}
		}else{
			Rect result;
			if(FrameIdx == 1){
				trackerVector.clear();
				for(unsigned ii = 0; ii <resutTrack.size(); ii++){//tracking
					int xMin = resutTrack[ii].detBox.xmin;
					int yMin = resutTrack[ii].detBox.ymin;
					int width_ = resutTrack[ii].detBox.xmax - resutTrack[ii].detBox.xmin;
					int height_ = resutTrack[ii].detBox.ymax - resutTrack[ii].detBox.ymin;
					KCFTracker tracker(configParam.HOG, configParam.FIXEDWINDOW, configParam.MULTISCALE, configParam.LAB);
					tracker.init( Rect(xMin, yMin, width_, height_), frame );
					rectangle( frame, Point( xMin, yMin ), Point( xMin+width_, yMin+height_), 
													Scalar( 0, 255, 255 ), 1, 8 );
					cv::putText(frame, resutTrack[ii].name, cv::Point( result.x, result.y), 
							cv::FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2, 8, 0);
					trackerVector.push_back(tracker);
				}
			}else{
				for(unsigned ii = 0; ii <resutTrack.size(); ii++){
					result = trackerVector[ii].update(frame);
					rectangle( frame, Point( result.x, result.y ), 
											Point( result.x+result.width, result.y+result.height), 
															Scalar( 0, 255, 255 ), 1, 8 );
					cv::putText(frame, resutTrack[ii].name, cv::Point( result.x, result.y), 
						cv::FONT_HERSHEY_COMPLEX, 1, Scalar(0, 255, 255), 2, 8, 0);
				}
			}
			
		}
		FrameIdx++;
		if(FrameIdx == trackingGap){
			FrameIdx = 0;
		}
		#else
		std::vector<output> outs = faceInfernece.faceDetector(frame);
		DETECTIONS detections;
		postprocess(outs,detections);
		if(getRectsFeature(frame, detections)){
			DeepSortTracker.predict();
          	DeepSortTracker.update(detections);
			std::vector<RESULT_DATA> result;
          	for(Track& track : DeepSortTracker.tracks) {
				if(!track.is_confirmed() || track.time_since_update > 1)
					continue;
				result.push_back(std::make_pair(track.track_id, track.to_tlwh()));
            }
          	for(unsigned n = 0; n < detections.size(); n++){
				DETECTBOX tmpbox = detections[n].tlwh;
				cv::Rect rect(tmpbox(0), tmpbox(1), tmpbox(2), tmpbox(3));
				cv::rectangle(frame, rect, cv::Scalar(0,0,255), 4);

				for(unsigned k = 0; k < result.size(); k++){
					DETECTBOX tmp = result[k].second;
					cv::Rect rect = cv::Rect(tmp(0), tmp(1), tmp(2), tmp(3));
					rectangle(frame, rect, cv::Scalar(255, 255, 0), 2);
					std::string label = cv::format("%d", result[k].first);
					cv::putText(frame, label, cv::Point(rect.x, rect.y), cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 0), 2);
				}
            }
		}
		#endif
        imshow("faceRecognition",frame);
        if(waitKey(1) > 0)  
            stop = true;
    }
	cap.release();
#endif
}
