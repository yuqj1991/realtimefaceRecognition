#ifndef _RESIDEO_UTILS_HPP_
#define _RESIDEO_UTILS_HPP_
#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <stdlib.h>
#include <assert.h>
#include <cmath>
namespace RESIDEO{
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
        int glass;
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
    typedef std::pair<int, box> objectBox;

    typedef struct _faceAnalysis_result{
        box faceBox;
        faceattribute faceAttri;
        encodeFeature faceFeature;
        bool haveFeature;
    } faceAnalysisResult;

    typedef struct _reidAnalysis_result{
        box bodyBox;
        encodeFeature reidfeature;
    } reidAnalysisResult;

    typedef struct detBoxInfo_{
        box detBox;
        std::string name;
    }detBoxInfo;
    typedef std::vector<detBoxInfo> RecognResultTrack;

    /*******************************mapSet方式存储********************************/
    struct featureCmp{
        bool operator()(const encodeFeature &leftValue, const encodeFeature &rightValue) const{
            float top =0.0f, bottomLeft=0.0f, bottomRight=0.0f, EuclideanValue = 0.0f;
            int featureDim = 512;
            assert(leftValue.featureFace.size()==rightValue.featureFace.size());
            assert(leftValue.featureFace.size() == featureDim);
            for(int ii = 0; ii < featureDim; ii++){
                top += leftValue.featureFace[ii]*rightValue.featureFace[ii];
                bottomLeft += leftValue.featureFace[ii]*leftValue.featureFace[ii];
                bottomRight += rightValue.featureFace[ii]*rightValue.featureFace[ii];
                EuclideanValue += std::pow((leftValue.featureFace[ii]-rightValue.featureFace[ii]),2);
            }
            
            float cosValue = (float) (top/(sqrt(bottomLeft)*sqrt(bottomRight)));
            float Euclidean = std::sqrt(EuclideanValue);

            if(Euclidean <= 1.2){
                return false;
            }else{
                for(int ii = 0; ii < featureDim; ii++){
                    if(std::abs(leftValue.featureFace[ii])!=std::abs(rightValue.featureFace[ii]))
                        return std::abs(leftValue.featureFace[ii])>std::abs(rightValue.featureFace[ii]);
                }
            }
        }
    };
    typedef std::map<encodeFeature, std::string, featureCmp> mapFeature;
    typedef std::map<int, mapFeature> mapFaceCollectDataSet;

    class util{
        public:
        util();
        float computeDistance(const Prediction leftValue, const Prediction &rightValue, 
                        unsigned int method, int featureDim);
        std::pair<float, std::string>serachCollectDataNameByloop(FaceBase dataColletcion,
             encodeFeature feature, int gender);
        std::pair<float, std::string>serachCollectDataNameByloop(vector_feature dataColletcion,
             encodeFeature feature);
        std::string serachCollectDataNameBymapSet(mapFaceCollectDataSet dataTestSet,
             encodeFeature detFeature, int gender);
        mapFaceCollectDataSet getmapDatafaceBase(FaceBase &dataColletcion);
        
        public:
        float cosValueThresold;
        float euclideanValueThresold;
        std::string faceDir;
        std::string facefeaturefile;
        std::string HOGfacefeaturefile;
        std::string cropfaceDir;
        modelParameter detParam;
        modelParameter attriParam;
        modelParameter facenetParam;
        modelParameter reidParam;
        int detMargin;
        float confidencethreold;
        std::vector < std::string > labelGender{"male", "female"};
        std::vector < std::string > labelGlass{"wearing glasses", "not wearing glasses"};
        bool HOG;
        bool FIXEDWINDOW;
        bool MULTISCALE;
        bool LAB;
        int nn_budget;
        float max_cosine_distance;
        int facefeatureDim;
        int faceHOGfeatureDim;
    };
}
#endif