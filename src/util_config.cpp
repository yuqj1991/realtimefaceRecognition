#include "utils_config.hpp"

namespace RESIDEO{
    float util::computeDistance(const Prediction leftValue, const Prediction &rightValue, 
                                unsigned int method){
        float top =0.0f, bottomLeft=0.0f, bottomRight=0.0f, euclideanValue = 0.0f;

        assert(leftValue.size()==rightValue.size());
        assert(leftValue.size() == 512);

        for(int ii = 0; ii < 512; ii++){
            top += leftValue[ii]*rightValue[ii];
            bottomLeft += leftValue[ii]*leftValue[ii];
            bottomRight += rightValue[ii]*rightValue[ii];
            euclideanValue += std::pow((leftValue[ii]-rightValue[ii]),2);
        }

        float cosValue = (float) (top/(sqrt(bottomLeft)*sqrt(bottomRight)));
        float Euclidean = std::sqrt(euclideanValue);
        switch (method)
        {
        case 0:
            return Euclidean;
            break;
        case 1:
            return cosValue;
            break;
        default:
            break;
        }
    }

    std::pair<float, std::string> util::serachCollectDataNameByloop(FaceBase dataColletcion,
             encodeFeature feature, int gender){
        std::pair<float, std::string> result;
        float maxDist = 0.f, comDist = 0.f;
        if(dataColletcion.find(gender)!=dataColletcion.end()){
            vector_feature subFaceDataSet = dataColletcion.find(gender)->second;
            for(unsigned nn = 0; nn<subFaceDataSet.size(); nn++){
                comDist = computeDistance(feature.featureFace, subFaceDataSet[nn].second.featureFace, 1);
                //printf("nn: %d, cosDis: %f, dataset name: %s\n", nn, comDist, subFaceDataSet[nn].first.c_str());
                if(maxDist < comDist){
                    result.second = subFaceDataSet[nn].first;
                    result.first = comDist;
                    maxDist = comDist;
                }
            }  
        }else{
            result.second = "unknown one";
            result.first = 0.f;
        }
        return result;
    }

    std::string util::serachCollectDataNameBymapSet(mapFaceCollectDataSet dataTestSet,
             encodeFeature detFeature, int gender){
        std::string result;
        if(dataTestSet.find(gender)!=dataTestSet.end()){
            mapFeature subSet = dataTestSet.find(gender)->second;
            if(subSet.find(detFeature)!=subSet.end()){
                result = subSet.find(detFeature)->second;
            }else{
                result = "unknown person";
            }
        }else{
            result = "unknown person";
        }
        return result;
    }
    /******************初始化网络模型*************************/
    util::util(){
        cosValueThresold = 0.55f;
        euclideanValueThresold = 1.20f;
        faceDir = "../faceBase";
        facefeaturefile = "../savefeature.txt";
        cropfaceDir = "../faceCropBase/";
        detParam ={
            .m_model_weight_ = "../model/face_detector.caffemodel",
            .m_model_prototxt_ = "../model/face_detector.prototxt",
            .m_std_value_ = 0.007845,
            {103.94, 116.78, 123.68}
        };
        attriParam ={
            .m_model_weight_ = "../model/face_attributes_glass.caffemodel",
            .m_model_prototxt_ = "../model/face_attributes_glass.prototxt",
            .m_std_value_ = 0.007845,
            {127.5, 127.5, 127.5}
        };
        facenetParam ={
            .m_model_weight_ = "../model/facenet.caffemodel",
            .m_model_prototxt_ = "../model/facenet.prototxt",
            .m_std_value_ = 0.007845,
            {127.5, 127.5, 127.5}
        };
        /*****************static variables******************/
        detMargin = 32;
        confidencethreold = 0.35;
        labelGender[2] = {"male", "female"};
        labelGlass[2] = {"wearing glasses", "not wearing glasses"};
        /****************初始化跟踪模块***********************/
        HOG = true;
        FIXEDWINDOW = false;
        MULTISCALE = true;
        LAB = false;
        nn_budget = 100;
        max_cosine_distance = 0.2f;
    }
}