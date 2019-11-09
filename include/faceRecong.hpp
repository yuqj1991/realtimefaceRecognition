#ifndef _RESIDEOFACENETRECO_
#define _RESIDEOFACENETRECO_

#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "objectbase.hpp"
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>
#define USE_OPENCV 1
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif


namespace RESIDEO{
    
    class Facenet:public objectbase
    {
        public:
            cv::Size m_input_geometry_;
            int m_num_channels_;
        public:
            explicit Facenet(modelParameter &param);
            void WrapInputLayer(std::vector<cv::Mat>* input_channels);
            void Preprocess(cv::Mat& img, std::vector<cv::Mat>* input_channels);
            encodeFeature Predict(cv::Mat &inputImg);
            inline encodeFeature normL2Vector(encodeFeature en_feature){
                assert(en_feature.featureFace.size() == 512);
                float sum_vector;
                for(int i = 0; i < en_feature.featureFace.size(); i++){
                    sum_vector += std::pow(en_feature.featureFace[i], 2);
                }
                float sum_sqrt =std::sqrt(sum_vector) + 0.0000001;
                encodeFeature normVector;
                for(int i = 0; i < en_feature.featureFace.size(); i++){
                    normVector.featureFace.push_back( float (en_feature.featureFace[i] / sum_sqrt));
                }
                return normVector;
            }
    };

}
#endif
