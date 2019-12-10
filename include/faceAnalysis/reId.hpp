#ifndef _RESIDEOREIDRECO_
#define _RESIDEOREIDRECO_

#include "objectbase.hpp"
#include "util/utils_config.hpp"
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif

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
            encodeFeature Predict(cv::Mat &inputImg);
            inline encodeFeature normL2Vector(encodeFeature en_feature){
                float sum_vector = 0.f;
                for(unsigned i = 0; i < en_feature.featureFace.size(); i++){
                    sum_vector += std::pow(en_feature.featureFace[i], 2);
                }
                float sum_sqrt =std::sqrt(sum_vector) + 0.0000001;
                encodeFeature normVector;
                for(unsigned i = 0; i < en_feature.featureFace.size(); i++){
                    normVector.featureFace.push_back( float (en_feature.featureFace[i] / sum_sqrt));
                }
                return normVector;
            }
    };

}
#endif
