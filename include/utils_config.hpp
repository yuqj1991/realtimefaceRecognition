#ifndef _RESIDEO_UTILS_HPP_
#define _RESIDEO_UTILS_HPP_
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <stdlib.h>
#include <assert.h>
const float cosValueThresold = 0.65f;
const float euclideanValueThresold = 1.20f;

typedef struct modelParameter_{
    std::string m_model_weight_;
    std::string m_model_prototxt_;
    float m_std_value_;
    float m_mean_value_[3];
}modelParameter;

typedef std::vector<float> Prediction;
struct encodeFeature{
    Prediction featureFace;
};

typedef std::vector<std::pair<std::string, encodeFeature > >vector_feature;
typedef std::map<int,  vector_feature> FaceBase;

typedef struct resideo_point_{
    int point_x;
    int point_y;
}point;

typedef struct Angle_{
    float yaw;
    float pitch;
    float roll;
}angle;

typedef struct faceAttri_{
    int gender;
    std::vector<point> landmarks;
    angle facepose;
}faceattribute;


typedef struct Box_{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
}box;
typedef std::pair<float, box> output;

typedef struct _faceAnalysis_result{
    box faceBox;
    faceattribute faceAttri;
    encodeFeature faceFeature;
    bool haveFeature;
} faceAnalysisResult;

typedef struct detBoxInfo_{
	box detBox;
	std::string name;
}detBoxInfo;

typedef std::vector<detBoxInfo> RecognResultTrack;
/*******************************第三种方式map存储********************************/
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

        if(Euclidean <= euclideanValueThresold){
            return false;
        }else{
            for(int ii = 0; ii < 512; ii++){
                if(std::abs(leftValue.featureFace[ii])!=std::abs(rightValue.featureFace[ii]))
                    return std::abs(leftValue.featureFace[ii])>std::abs(rightValue.featureFace[ii]);
            }
        }
    }
};
typedef std::map<encodeFeature, std::string, featureCmp> mapFeature;
typedef std::map<int, mapFeature> mapFaceCollectDataSet;

/******************静态函数******************************/

static float computeDistance(const Prediction leftValue, const Prediction &rightValue, 
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
    case 0: //euclideanDistance
        return Euclidean;
        break;
    case 1: //cosDistance
        return cosValue;
        break;
    default:
        break;
    }
    
}

static std::pair<float, std::string>serachCollectDataNameByloop(FaceBase dataColletcion,
             encodeFeature feature, int gender){
    std::pair<float, std::string> result;
    float maxDist = 0.f, comDist = 0.f;
    if(dataColletcion.find(gender)!=dataColletcion.end()){
        vector_feature subFaceDataSet = dataColletcion.find(gender)->second;
        for(int nn = 0; nn<subFaceDataSet.size(); nn++){
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


/****************计算维度的标准方差*********************/
static float computeVariance(Prediction Dimfeature){
    float mean = 0.f, variance = 0.f;
    for(int i = 0; i < Dimfeature.size(); i++){
        mean += Dimfeature[i];
        variance += std::pow((Dimfeature[i]), 2.0);
    }
    mean *= float(1 / Dimfeature.size());
    variance *= float(1 / Dimfeature.size());
    variance = variance - std::pow(mean, 2.0);
    return variance;
}
/****************计算每个维度的中度值*******************/
static float computeMedianValue(std::vector<std::pair<Prediction, std::string > > points, int featureIdx){
    Prediction dimfeature;
    int nrof_samples = points.size(); 
    for(int i = 0; i < nrof_samples; i++){
        dimfeature.push_back(points[i].first[featureIdx]);
    }
    std::sort(dimfeature.begin(), dimfeature.end());
    int pos = dimfeature.size() /2 ;
    return dimfeature[pos];
}
/****************计算样本中每个维度的方差值***************/
static int choose_feature(std::vector<std::pair<Prediction, std::string > > points){
    int nrof_samples = points.size(); 
    int N = points[0].first.size();
    Prediction dimfeature;
    float variance_max = 0.f;
    int featureidx = 0;
    for(int i = 0; i < N; i++){
        dimfeature.clear();
        for(int j = 0; j < nrof_samples; j++)
            dimfeature.push_back(points[j].first[i]);
        float variance = computeVariance(dimfeature);
        if(variance_max < variance){
            variance_max = variance;
            featureidx = i;
        }
    }
    return featureidx;
}


static std::string serachCollectDataNameBymapSet(mapFaceCollectDataSet dataTestSet,
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

static std::string faceDir = "../faceBase";
static std::string facefeaturefile = "../savefeature.txt";
static std::string cropfaceDir = "../faceCropBase/";
static modelParameter detParam ={
    .m_model_weight_ = "../model/face_detector.caffemodel",
    .m_model_prototxt_ = "../model/face_detector.prototxt",
    .m_std_value_ = 0.007845,
    {103.94, 116.78, 123.68}
};
static modelParameter attriParam ={
    .m_model_weight_ = "../model/facelandmark.caffemodel",
    .m_model_prototxt_ = "../model/facelandmark.prototxt",
    .m_std_value_ = 0.007845,
    {127.5, 127.5, 127.5}
};
static modelParameter facenetParam ={
    .m_model_weight_ = "../model/facenet.caffemodel",
    .m_model_prototxt_ = "../model/facenet.prototxt",
    .m_std_value_ = 0.007845,
    {127.5, 127.5, 127.5}
};
/*****************static variables******************/
static int detMargin = 32;
static float confidencethreold = 0.35;
static char *labelGender[] = {"male", "female"};
/****************初始化跟踪模块***********************/
static bool HOG = true;
static bool FIXEDWINDOW = false;
static bool MULTISCALE = true;
static bool LAB = false;
#endif