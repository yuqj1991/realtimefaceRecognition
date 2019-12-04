#ifndef _FACEANALYSIS_RESIDEO_
#define _FACEANALYSIS_RESIDEO_

#include <opencv2/core/core.hpp>

#include "objectDetect.hpp"
#include "faceAttri.hpp"
#include "faceRecong.hpp"
#include "utils_config.hpp"
using namespace std;
using namespace cv;
using namespace RESIDEO;

class faceAnalysis{
    public:
        faceAnalysis();
        ~faceAnalysis();
        std::vector<output> faceDetector(cv::Mat frame);
        std::vector<faceAnalysisResult> faceInference(cv::Mat frame, int detMargin, float angleThreold);
    private:
        util configParam;
        objectDetect m_faceDet;
        faceAttri m_faceAttri;
        Facenet m_facenet;
        inline bool AngleThresholdJudgment(angle an, float yawThreshold, float pitchThreshold, 
                        float rollThreshold){
            if(an.pitch > pitchThreshold || an.roll > rollThreshold || an.yaw > yawThreshold)
                return false;
            else
                return true;
        }
    protected:
};

#endif