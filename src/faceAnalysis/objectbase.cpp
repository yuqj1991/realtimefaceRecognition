#include  "faceAnalysis/objectbase.hpp"

namespace RESIDEO{
    objectbase::objectbase(modelParameter &param):m_model_parameter(param){
    }
    void objectbase::init_net(){
        #ifdef CPU_ONLY
            Caffe::set_mode(Caffe::CPU);
        #else
            Caffe::set_mode(Caffe::GPU);
        #endif
        net_.reset(new Net<float>(m_model_parameter.m_model_prototxt_, TEST));
        net_->CopyTrainedLayersFrom(m_model_parameter.m_model_weight_);
    }
}//namespace
