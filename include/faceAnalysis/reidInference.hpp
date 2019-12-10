#ifndef _REIDANALYSIS_RESIDEO_
#define _REIDANALYSIS_RESIDEO_

#include <opencv2/core/core.hpp>

#include "objectDetect.hpp"
#include "reId.hpp"
#include "util/utils_config.hpp"
using namespace std;
using namespace cv;
using namespace RESIDEO;

class reidAnalysis{
    public:
        reidAnalysis();
        ~reidAnalysis();
        std::vector<output> bodyDetector(cv::Mat frame);
        std::vector<reidAnalysisResult> reidInference(cv::Mat frame, int detMargin);
    private:
        util configParam;
        objectDetect m_bodyDet;
        reID m_reidRecnet;
    protected:
};

#endif