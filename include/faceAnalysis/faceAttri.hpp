#ifndef RESIDEO_FACEATTRI_
#define RESIDEO_FACEATTRI_
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
    class faceAttri:public objectbase
    {
        private:
            cv::Size m_input_geometry_;
            int m_num_channels_;
        public:
            explicit faceAttri(modelParameter &param);
            void WrapInputLayer(std::vector<cv::Mat>* input_channels);
            void Preprocess(cv::Mat& img, std::vector<cv::Mat>* input_channels);
            faceattribute Predict(cv::Mat &inputImg);
            cv::Mat getwarpAffineImg(cv::Mat &src, std::vector<cv::Point2f> landmarks);
    };  
}//namespace

#endif //header