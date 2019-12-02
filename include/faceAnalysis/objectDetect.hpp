#ifndef RESIDEO_OBJECTDETECT_
#define RESIDEO_OBJECTDETECT_
#include <stdlib.h>
#include <iostream>
#include <fstream>
#define USE_OPENCV 1
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include "objectbase.hpp"
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include <boost/shared_ptr.hpp>


namespace RESIDEO{
    class objectDetect:public objectbase
    {
        private:
            cv::Size m_input_geometry_;
            int m_num_channels_;
            float m_confidence_threshold_;
        public:
            explicit objectDetect(modelParameter &param, float & confidence_threshold);
            void WrapInputLayer(std::vector<cv::Mat>* input_channels);
            void Preprocess(cv::Mat& img, std::vector<cv::Mat>* input_channels);
            std::vector<Prediction> Predict(cv::Mat &inputImg);
            std::vector<output> getDetectfaceResultBox(cv::Mat& img);
            std::vector<output> getDetectpersonResultBox(cv::Mat& img);
    };  
}//namespace

#endif //header
