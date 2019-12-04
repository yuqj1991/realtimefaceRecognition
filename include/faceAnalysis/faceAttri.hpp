#ifndef RESIDEO_FACEATTRI_
#define RESIDEO_FACEATTRI_
#include <stdlib.h>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include "objectbase.hpp"
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

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