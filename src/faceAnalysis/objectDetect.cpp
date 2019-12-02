#include "faceAnalysis/objectDetect.hpp"

using namespace std;
using namespace cv;
using namespace caffe;

namespace RESIDEO{
    objectDetect::objectDetect(modelParameter &param, 
                    float & confidence_threshold):objectbase(param){
        init_net();
        Blob<float>* input_layer = net_->input_blobs()[0];
        m_num_channels_ = input_layer->channels();
        m_input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
        m_confidence_threshold_ = confidence_threshold;
    }
    
    void objectDetect::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
        Blob<float>* input_layer = net_->input_blobs()[0];
        int width = input_layer->width();
        int height = input_layer->height();
        float* input_data = input_layer->mutable_cpu_data();
        for (int i = 0; i < input_layer->channels(); ++i) {
            cv::Mat channel(height, width, CV_32FC1, input_data);
            input_channels->push_back(channel);
            input_data += width * height;
        }
    }

    void objectDetect::Preprocess(cv::Mat& img,
                            std::vector<cv::Mat>* input_channels) {
        cv::Mat sample;
        if (img.channels() == 3 && m_num_channels_ == 1)
            cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
        else if (img.channels() == 1 && m_num_channels_ == 3)
            cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
        else
            sample = img;

        cv::Mat sample_resized;
        if (sample.size() != m_input_geometry_)
            cv::resize(sample, sample_resized, m_input_geometry_);
        else
            sample_resized = sample;
        int height = sample_resized.rows;
        int width = sample_resized.cols;
        cv::Mat sample_float;
        if (m_num_channels_ == 3){
            sample_float=cv::Mat(width,height, CV_32FC3);
            for(int ii =0; ii< height; ii++){
                uchar * rowdata = sample_resized.ptr<uchar>(ii);
                float * fdata = sample_float.ptr<float>(ii);
                for(int jj =0; jj< width; jj++){
                    fdata[jj*3 ] = (rowdata[jj*3] - m_model_parameter.m_mean_value_[0]) *m_model_parameter.m_std_value_;
                    fdata[jj*3 + 1 ] = (rowdata[jj*3 + 1 ] - m_model_parameter.m_mean_value_[1]) *m_model_parameter.m_std_value_;
                    fdata[jj*3 + 2 ] = (rowdata[jj*3 + 2 ] - m_model_parameter.m_mean_value_[2]) *m_model_parameter.m_std_value_;
                }
            }
        }
        else
            sample_resized.convertTo(sample_float, CV_32FC3, 
            m_model_parameter.m_std_value_,
            -m_model_parameter.m_mean_value_[0] * m_model_parameter.m_std_value_);
        cv::split(sample_float, *input_channels);
    }

    std::vector<Prediction> objectDetect::Predict(cv::Mat &inputImg){
        Blob<float>* input_layer = net_->input_blobs()[0];
        input_layer->Reshape(1, m_num_channels_,
                       m_input_geometry_.height, m_input_geometry_.width);
        net_->Reshape();
        std::vector<cv::Mat> input_channels;
        WrapInputLayer(&input_channels);
        Preprocess(inputImg, &input_channels);
        net_->Forward();
        Blob<float>* output_layer = net_->output_blobs()[0];
        const float* result = output_layer->cpu_data();
        const int num_det = output_layer->height();
        std::vector<Prediction> detections;
        #if DEBUG
            printf("num_det: %d\n", num_det);
        #endif
        for(int k = 0; k < num_det; ++k){
            if(result[0] == -1){
                result +=7;
                continue;
            }
            Prediction detection(result, result + 7);
            detections.push_back(detection);
            result += 7;
        }
        return detections;
    }

    std::vector<output> objectDetect::getDetectfaceResultBox(cv::Mat& img){
        std::vector<Prediction> detections = Predict(img);
        std::vector<output> out;
        int width = img.cols;
        int height = img.rows;
        for (int i = 0; i < detections.size(); ++i) {
            const vector<float>& d = detections[i];
            const float score = d[2];
            if (score >= m_confidence_threshold_) {
                box ou = {
                    .xmin = static_cast<int>(d[3]*width),
                    .ymin = static_cast<int>(d[4]*height),
                    .xmax = static_cast<int>(d[5]*width),
                    .ymax = static_cast<int>(d[6]*height)
                };
                printf("score: %f,xmin: %f,ymin: %f, xmax: %f, ymax: %f\n", score, d[3], d[4], d[5], d[6]);
                printf("score: %f,xmin: %d,ymin: %d, xmax: %d, ymax: %d\n", score, int(d[3]*width), ou.ymin, ou.xmax, ou.ymax);
                out.push_back(std::make_pair(score, ou));
                                    
            }
        }
        return out;
    }
    std::vector<output> objectDetect::getDetectpersonResultBox(cv::Mat& img){
        std::vector<Prediction> detections = Predict(img);
        std::vector<output> out;
        int width = img.cols;
        int height = img.rows;
        for (int i = 0; i < detections.size(); ++i) {
            const vector<float>& d = detections[i];
            const float score = d[2];
            if(d[1] != 15)
                continue;
            if (score >= m_confidence_threshold_) {
                box ou = {
                    .xmin = static_cast<int>(d[3]*width),
                    .ymin = static_cast<int>(d[4]*height),
                    .xmax = static_cast<int>(d[5]*width),
                    .ymax = static_cast<int>(d[6]*height)
                };
                printf("score: %f,xmin: %f,ymin: %f, xmax: %f, ymax: %f\n", score, d[3], d[4], d[5], d[6]);
                printf("score: %f,xmin: %d,ymin: %d, xmax: %d, ymax: %d\n", score, int(d[3]*width), ou.ymin, ou.xmax, ou.ymax);
                out.push_back(std::make_pair(score, ou));
                                    
            }
        }
        return out;
    }
}//namespace
