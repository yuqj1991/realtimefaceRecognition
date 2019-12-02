#include "faceAnalysis/faceAnalysis.hpp"
using namespace RESIDEO;

faceAnalysis::faceAnalysis():
    m_faceDet(detParam, confidencethreold),
    m_faceAttri(attriParam),
    m_facenet(facenetParam){

}

std::vector<faceAnalysisResult> faceAnalysis::faceInference(cv::Mat frame, int detMargin, float angleThreold){
    int width = frame.cols;
    int height = frame.rows;
    std::vector<faceAnalysisResult>result;
    std::vector<output> faceDetect= m_faceDet.getDetectfaceResultBox(frame);
    for(int ii = 0; ii < faceDetect.size(); ii++){
        faceAnalysisResult tempResult;
        box faceDetBox = faceDetect[ii].second;
        int xmin = max(faceDetBox.xmin - detMargin/2, 0);
        int ymin = max(faceDetBox.ymin - detMargin/2, 0);
        int xmax = min(faceDetBox.xmax + detMargin/2, width);
        int ymax = min(faceDetBox.ymax + detMargin/2, height);
        box tempBox = {
            .xmin = xmin,
            .ymin = ymin,
            .xmax = xmax,
            .ymax = ymax
        };
        tempResult.faceBox = tempBox;
        int w = (xmax - xmin);
        int h=  (ymax - ymin);
        cv::Mat RoiImg = frame(cv::Rect(xmin, ymin, w, h));
        faceattribute faceAttriResult = m_faceAttri.Predict(RoiImg);
        tempResult.faceAttri = faceAttriResult;
        tempResult.haveFeature = false;
        if(AngleThresholdJudgment(faceAttriResult.facepose, angleThreold, angleThreold, angleThreold)){
            tempResult.haveFeature = true;		
            /*****************对齐人脸，依据两眼坐标的角度**************/
            std::vector<cv::Point2f> landmarks(5);
            for(int ii =0; ii<5; ii++){
                landmarks[ii].x = faceAttriResult.landmarks[ii].point_x;
                landmarks[ii].y = faceAttriResult.landmarks[ii].point_y;
            }
            cv::Mat alignRoiImg = m_faceAttri.getwarpAffineImg(RoiImg, landmarks);
            /**********************人脸识别**************************/
            encodeFeature faceFeature = m_facenet.Predict(alignRoiImg);
            tempResult.faceFeature = faceFeature;
        }

        result.push_back(tempResult);
                
    }

    return result;
}
faceAnalysis::~faceAnalysis(){

}