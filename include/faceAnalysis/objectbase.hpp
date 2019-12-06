#ifndef DEEPANO_OBJECTDETECT_
#define DEEPANO_OBJECTDETECT_
#include <stdlib.h>
#include <iostream>
#include <string>
#include <caffe/caffe.hpp>
#include "util/utils_config.hpp"
using namespace caffe;


namespace RESIDEO{
    class objectbase
    {
        public:
            modelParameter m_model_parameter;
            std::shared_ptr<Net<float> > net_;
        public:
            objectbase(modelParameter &param);
            //~objectbase();
            void init_net();
        private:
            util configParam;
        };
    
}//namespace

#endif //header
