#include "faceAnalysis/reidInference.hpp"
using namespace RESIDEO;

reidAnalysis::reidAnalysis():
    m_bodyDet(configParam.detParam, configParam.confidencethreold),
    m_reidRecnet(configParam.reidParam, configParam.confidencethreold){

}

std::vector<output> reidAnalysis::bodyDetector(cv::Mat frame){
    return m_bodyDet.getDetectfaceResultBox(frame);
}

std::vector<reidAnalysisResult> reidAnalysis::faceInference(cv::Mat frame, int detMargin){
    int width = frame.cols;
    int height = frame.rows;
    std::vector<reidAnalysisResult>result;
    std::vector<output> Detect= m_bodyDet.getDetectfaceResultBox(frame);
    for(unsigned ii = 0; ii < Detect.size(); ii++){
        reidAnalysisResult tempResult;
        box bodyDetBox = Detect[ii].second;
        int xmin = max(bodyDetBox.xmin - detMargin/2, 0);
        int ymin = max(bodyDetBox.ymin - detMargin/2, 0);
        int xmax = min(bodyDetBox.xmax + detMargin/2, width);
        int ymax = min(bodyDetBox.ymax + detMargin/2, height);
        box tempBox = {
            .xmin = xmin,
            .ymin = ymin,
            .xmax = xmax,
            .ymax = ymax
        };
        tempResult.bodyBox = tempBox;
        int w = (xmax - xmin);
        int h=  (ymax - ymin);
        cv::Mat RoiImg = frame(cv::Rect(xmin, ymin, w, h));
        encodeFeature feature = m_reidRecnet.Predict(RoiImg);
        tempResult.reidfeature = feature;
        result.push_back(tempResult);        
    }
    return result;
}
reidAnalysis::~reidAnalysis(){

}