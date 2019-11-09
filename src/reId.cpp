#include "reId.hpp"

using namespace cv;
using namespace std;

namespace RESIDEO{

    reID::reID(modelParameter &param):objectbase(param){
        init_net();
        Blob<float>* input_layer = net_->input_blobs()[0];
        m_num_channels_ = input_layer->channels();
        m_input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
        mapFeatureIndex = 0;
    }

    void reID::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
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

    void reID::Preprocess(cv::Mat& img,
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

        cv::Mat sample_float;
        if (m_num_channels_ == 3)
            sample_resized.convertTo(sample_float, CV_32FC3, 
                        m_model_parameter.m_std_value_, 
                        -m_model_parameter.m_mean_value_[0] * m_model_parameter.m_std_value_);
        else
            sample_resized.convertTo(sample_float, CV_32FC1, 
            m_model_parameter.m_std_value_,
            -m_model_parameter.m_mean_value_[0] * m_model_parameter.m_std_value_);
        cv::split(sample_float, *input_channels);
    }
    std::vector<float> reID::Predict(cv::Mat &inputImg){
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

        std::vector<float> featurerawFace, normface;
        for(int ii=0; ii<512; ii++){
            featurerawFace.push_back(result[ii]);
        }
        normface = normL2Vector(featurerawFace);
        return normface;
    }
}//namespace
