#ifndef _RESIDEOREIDRECO_
#define _RESIDEOREIDRECO_

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
    class reID:public objectbase
    {
        public:
            cv::Size m_input_geometry_;
            int m_num_channels_;
        public:
            explicit reID(modelParameter &param);
            void WrapInputLayer(std::vector<cv::Mat>* input_channels);
            void Preprocess(cv::Mat& img, std::vector<cv::Mat>* input_channels);
            std::vector<float> Predict(cv::Mat &inputImg);
            inline vector<float> normL2Vector(vector<float> feature){
                assert(feature.size() > 0);
                float sum_vector;
                for(int i = 0; i < feature.size(); i++){
                    sum_vector += std::pow(feature[i], 2);
                }
                float sum_sqrt =std::sqrt(sum_vector) + 0.0000001;
                vector<float> normVector;
                for(int i = 0; i < feature.size(); i++){
                    normVector.push_back( float (feature[i] / sum_sqrt));
                }
                return normVector;
            }
    };

}
#endif
