#include "faceAnalysis/faceAttri.hpp"

using namespace std;
using namespace cv;
using namespace caffe;

namespace RESIDEO{
    faceAttri::faceAttri(modelParameter &param):objectbase(param){
        init_net();
        Blob<float>* input_layer = net_->input_blobs()[0];
        m_num_channels_ = input_layer->channels();
        m_input_geometry_ = cv::Size(input_layer->width(), input_layer->height());
    }
    
    void faceAttri::WrapInputLayer(std::vector<cv::Mat>* input_channels) {
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

    void faceAttri::Preprocess(cv::Mat& img,
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

    faceattribute faceAttri::Predict(cv::Mat &inputImg){
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
        const int num_result = output_layer->channels();
        vector<float> face_prediction_attributes;
        for(int jj =5*2; jj< num_result; jj++){
            face_prediction_attributes.push_back(result[jj]);
        }
        std::vector<point> landmarks;
        for(int jj =0; jj<5; jj++){
            point p = {
                .point_x = int(result[jj] * inputImg.cols),
                .point_y = int(result[jj + 5] * inputImg.rows)
            };
            landmarks.push_back(p);
        }
        angle ang = {
            .yaw = face_prediction_attributes[0],
            .pitch = face_prediction_attributes[1],
            .roll = face_prediction_attributes[2]
        };
        int gender_index=0, glass_index = 0; 
        float gender_temp=0.0, glasses_temp=0.0;
        for(int jj=0; jj<2; jj++){
            if(gender_temp<face_prediction_attributes[jj +3]){
                gender_index = jj;
                gender_temp = face_prediction_attributes[jj + 3];
            }
            if(glasses_temp<face_prediction_attributes[jj +5]){
                glass_index = jj;
                glasses_temp = face_prediction_attributes[jj + 5];
            }
        }
        faceattribute detection = {
            .gender = gender_index,
            .glass = glass_index,
            .landmarks = landmarks,
            .facepose = ang
        };
        return detection;
    }
    cv::Mat faceAttri::getwarpAffineImg(cv::Mat &src, std::vector<cv::Point2f> landmarks){
        cv::Mat oral; 
		src.copyTo(oral);
	
		//计算两眼中心点，按照此中心点进行旋转， 第0个为左眼坐标，1为右眼坐标
		cv::Point2f eyesCenter = cv::Point2f((landmarks[0].x + landmarks[1].x) * 0.5f, (landmarks[0].y + landmarks[1].y) * 0.5f);
	
		//计算两个眼睛间的角度
		double dy = (landmarks[1].y - landmarks[0].y);
		double dx = (landmarks[1].x - landmarks[0].x);
		double angle = atan2(dy, dx) * 180.0 / CV_PI; // Convert from radians to degrees.
	
		//由eyesCenter, angle, scale按照公式计算仿射变换矩阵，此时1.0表示不进行缩放
		Mat rot_mat = getRotationMatrix2D(eyesCenter, angle, 1.0);
		Mat rot;
		//进行仿射变换，变换后大小为src的大小
		cv::warpAffine(src, rot, rot_mat, src.size());
		std::vector<cv::Point2f> marks;
	
		//使用仿射变换矩阵，计算变换后各关键点在新图中所对应的位置坐标。
		for (int n = 0; n<5; n++)
		{
			cv::Point2f p = Point2f(0, 0);
			p.x = rot_mat.ptr<double>(0)[0] * landmarks[n].x + rot_mat.ptr<double>(0)[1] * landmarks[n].y + rot_mat.ptr<double>(0)[2];
			p.y = rot_mat.ptr<double>(1)[0] * landmarks[n].x + rot_mat.ptr<double>(1)[1] * landmarks[n].y + rot_mat.ptr<double>(1)[2];
			marks.push_back(p);
			landmarks[n].x = p.x;
			landmarks[n].y = p.y;
		}
		return rot;
    }
}//namespace
