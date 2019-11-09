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

const float cosValueThresold = 0.65f;
const float euclideanValueThresold = 0.25f;


namespace RESIDEO{
    struct encodeFeature{
        std::vector<float> featureFace;
    };
    struct featureCmp{
        bool operator()(const encodeFeature &leftValue, const encodeFeature &rightValue) const{
            float top =0.0f, bottomLeft=0.0f, bottomRight=0.0f, EuclideanValue = 0.0f;
            assert(leftValue.featureFace.size()==rightValue.featureFace.size());
            assert(leftValue.featureFace.size() == 512);
            for(int ii = 0; ii < 512; ii++){
                top += leftValue.featureFace[ii]*rightValue.featureFace[ii];
                bottomLeft += leftValue.featureFace[ii]*leftValue.featureFace[ii];
                bottomRight += rightValue.featureFace[ii]*rightValue.featureFace[ii];
                EuclideanValue += std::pow((leftValue.featureFace[ii]-rightValue.featureFace[ii]),2);
            }
            float cosValue = (float) (top/(sqrt(bottomLeft)*sqrt(bottomRight)));
            float Euclidean = std::sqrt(EuclideanValue);
            printf("EuclideanValue: %f, cosValue: %f\n", Euclidean, cosValue);
            if(cosValue > cosValueThresold && Euclidean < euclideanValueThresold){
                return false;
            }else{
                if(bottomLeft != bottomRight){
				    return bottomLeft < bottomRight;
                }else{
                    return (bottomLeft + 0.00025) > bottomRight;
			    }
            }
        }
    };
    typedef std::map<encodeFeature, std::string, featureCmp> mapFeature;
    class reID:public objectbase
    {
        public:
            cv::Size m_input_geometry_;
            int m_num_channels_;
            int mapFeatureIndex;
            std::map<int, mapFeature>featureDataBase;  //<gender, encodefeatue>
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
