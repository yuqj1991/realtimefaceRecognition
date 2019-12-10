#include "util/utils_config.hpp"
#include<opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <fstream>
using namespace std;
namespace RESIDEO{
    static bool readConfigFile(const char * cfgfilepath, const string & key, string & value)
    {
        fstream cfgFile;
        cfgFile.open(cfgfilepath);//打开文件	
        if( ! cfgFile.is_open())
        {
            cout<<"can not open cfg file!"<<endl;
            return false;
        }
        char tmp[1000];
        while(!cfgFile.eof())//循环读取每一行
        {
            cfgFile.getline(tmp,1000);//每行读取前1000个字符，1000个应该足够了
            string line(tmp);
            size_t pos = line.find('=');//找到每行的“=”号位置，之前是key之后是value
            if(pos==string::npos) return false;
            string tmpKey = line.substr(0,pos);//取=号之前
            if(key==tmpKey)
            {
                value = line.substr(pos+1);//取=号之后
                return true;
            }
        }
        return false;
    }
    float util::computeDistance(const Prediction leftValue, const Prediction &rightValue, 
                                unsigned int method, int featureDim){
        float top =0.0f, bottomLeft=0.0f, bottomRight=0.0f, euclideanValue = 0.0f;

        assert(leftValue.size()==rightValue.size());
        assert(leftValue.size() == featureDim);

        for(int ii = 0; ii < featureDim; ii++){
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
                comDist = computeDistance(feature.featureFace, subFaceDataSet[nn].second.featureFace, 1, feature.featureFace.size());
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

    std::pair<float, std::string> util::serachCollectDataNameByloop(vector_feature dataColletcion,
             encodeFeature feature){
        std::pair<float, std::string> result;
        float maxDist = 0.f, comDist = 0.f;
        for(unsigned nn = 0; nn<dataColletcion.size(); nn++){
            comDist = computeDistance(feature.featureFace, dataColletcion[nn].second.featureFace, 1, feature.featureFace.size());
            if(maxDist < comDist){
                result.second = dataColletcion[nn].first;
                result.first = comDist;
                maxDist = comDist;
            }
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

    mapFaceCollectDataSet util::getmapDatafaceBase(FaceBase &dataColletcion){
        mapFaceCollectDataSet dataTestSet;
        FaceBase::iterator it;
        for(it = dataColletcion.begin(); it != dataColletcion.end(); it++){
            int gender = it->first;
            vector_feature feature = it->second;
            for(unsigned i = 0; i < feature.size(); i++){

            }
            mapFeature subfeature;
            
            if(dataTestSet.find(gender) == dataTestSet.end()){
                for(unsigned j = 0; j < feature.size(); j++){
                    subfeature.insert(std::make_pair(feature[j].second, feature[j].first));
                }
                dataTestSet.insert(std::make_pair(gender, subfeature));
            }
        }
        int num = 0;
        mapFaceCollectDataSet::iterator iter;
        for(iter = dataTestSet.begin(); iter != dataTestSet.end(); iter++){
            mapFeature subfeature = iter->second;
            num += subfeature.size();
        }
        return dataTestSet;
    }
    /******************初始化网络模型*************************/
    util::util(){
        cosValueThresold = 0.55f;
        euclideanValueThresold = 1.20f;
        faceDir = "../faceBase";
        facefeaturefile = "../savefeature.txt";
        HOGfacefeaturefile = "../hogSaveFeature.txt";
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
        detMargin = 32;
        confidencethreold = 0.35f;
        HOG = true;
        FIXEDWINDOW = false;
        MULTISCALE = true;
        LAB = false;
        nn_budget = 100;
        facefeatureDim = 128;
        faceHOGfeatureDim = 3780;
    }
}