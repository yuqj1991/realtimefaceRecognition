/*
*time :2019.8.30
*author:yuqianjin
*/

#pragma once

#include "tracker.h"

#ifndef _OPENCV_KCFTRACKER_HPP_
#define _OPENCV_KCFTRACKER_HPP_
#endif

namespace RESIDEO{
  class KCFTracker : public Tracker
  {
  public:
      KCFTracker(bool hog = true, bool fixed_window = true, bool multiscale = true, bool lab = true);

      virtual void init(const cv::Rect &roi, cv::Mat image);
      
      virtual cv::Rect update(cv::Mat image);//新的一帧更新位置

      float interp_factor; // 线性插值因子
      float sigma; // 高斯核带宽
      float lambda; // 常数
      int cell_size; // HOG cell size
      int cell_sizeQ; // cell size^2, to avoid repeated operations
      float padding; 
      float output_sigma_factor;
      int template_size; 
      float scale_step; 
      float scale_weight;

  protected:
      cv::Point2f detect(cv::Mat z, cv::Mat x, float &peak_value);

      void train(cv::Mat x, float train_interp_factor);

      cv::Mat gaussianCorrelation(cv::Mat x1, cv::Mat x2);

      cv::Mat createGaussianPeak(int sizey, int sizex);

      cv::Mat getFeatures(const cv::Mat & image, bool inithann, float scale_adjust = 1.0f);

      void createHanningMats();

      float subPixelPeak(float left, float center, float right);

      cv::Mat _alphaf;
      cv::Mat _prob;
      cv::Mat _tmpl;
      cv::Mat _num;
      cv::Mat _den;
      cv::Mat _labCentroids;

  private:
      int size_patch[3];
      cv::Mat hann;
      cv::Size _tmpl_sz;
      float _scale;
      int _gaussian_size;
      bool _hogfeatures;
      bool _labfeatures;
  };
}
