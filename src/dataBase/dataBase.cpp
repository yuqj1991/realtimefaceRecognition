#include "dataBase/dataBase.hpp"
#ifdef USE_OPENCV
#include<opencv2/opencv.hpp>
#endif
namespace RESIDEO{
    dataBase::dataBase(string faceDir, string saveFeatureFile):m_BaseDir(faceDir), 
                    m_faceFile(saveFeatureFile)
    {
    }
    std::vector<float> getHogFeatureMap(cv::Mat mattmp){
        cv::resize(mattmp, mattmp, cv::Size(64, 128));
		cv::Mat dst_gray;
		cv::cvtColor(mattmp, dst_gray, CV_BGR2GRAY);
	
		HOGDescriptor detector(Size(64, 128), Size(16, 16), Size(8, 8), Size(8, 8), 9);
		vector<float> descriptor;
		vector<Point> location;
		detector.compute(dst_gray, descriptor, cv::Size(0, 0), Size(0, 0),location);
		return descriptor;
    }
    void dataBase::generateBaseFeature(faceAnalysis faceRegister){
        std::vector<string> filenames;
	    GetFileNames(m_BaseDir,filenames);
        std::cout<<"total base: "<<filenames.size()<<endl;
        std::ofstream outfile;
        outfile.open(m_faceFile, ios::out|ios::trunc);
        outfile.clear();
        for(unsigned i = 0; i < filenames.size(); i++){
            std::cout<<"****image name: "<<filenames[i]<<std::endl;
            cv::Mat image = cv::imread(filenames[i]);
            if (!image.data)
            {
                printf("No image data\n");
                return ;
            }
            std::vector<faceAnalysisResult> result= faceRegister.faceInference(image, configParam.detMargin, 20.0f);
            if (result.size() > 1){
                std::cout << "file =" << __FILE__ << ", line =" << __LINE__ << ", faceDetect result more than one: "
                                        <<result.size()<<" image name: "<<filenames[i]<<std::endl;
            }else{
                std::string tempString = filenames[i].substr(filenames[i].find_last_of("/")+1);
                std::string RegisterName = tempString.substr(0, tempString.find_last_of("."));
                #if 1
                box detBox = result[0].faceBox;
                cv::rectangle( image, cv::Point( detBox.xmin, detBox.ymin ), 
											cv::Point( detBox.xmax, detBox.ymax), 
															cv::Scalar( 0, 255, 255 ), 1, 8 );
                cv::imwrite(configParam.cropfaceDir + tempString, image);
                #endif
                
                if(result[0].haveFeature){
                    outfile << RegisterName;
                    outfile << " "<<result[0].faceAttri.gender<<" ";
                    for(unsigned nn = 0; nn< result[0].faceFeature.featureFace.size(); nn++)
                        outfile << " "<<result[0].faceFeature.featureFace[nn];
                }else{
                    std::cout << "file =" << __FILE__ << ", line =" << __LINE__ 
                        << ", face head pose more than threold, imgage file: "<< filenames[i]<<std::endl;
                }
            }
            outfile <<std::endl; 
        }
        outfile.close();
    }

    void dataBase::generatebodyFeature(reidAnalysis reidRegister){
        std::vector<string> filenames;
	    GetFileNames(m_BaseDir,filenames);
        std::cout<<"total base: "<<filenames.size()<<endl;
        std::ofstream outfile;
        outfile.open(m_faceFile, ios::out|ios::trunc);
        outfile.clear();
        for(unsigned i = 0; i < filenames.size(); i++){
            std::cout<<"****image name: "<<filenames[i]<<std::endl;
            cv::Mat image = cv::imread(filenames[i]);
            if (!image.data)
            {
                printf("No image data\n");
                return ;
            }
            std::vector<reidAnalysisResult> result= reidRegister.reidInference(image, configParam.detMargin);
            if (result.size() > 1){
                std::cout << "file =" << __FILE__ << ", line =" << __LINE__ << ", bodyDetect result more than one: "
                                        <<result.size()<<" image name: "<<filenames[i]<<std::endl;
            }else{
                std::string tempString = filenames[i].substr(filenames[i].find_last_of("/")+1);
                std::string RegisterName = tempString.substr(0, tempString.find_last_of("."));
                #if 1
                box detBox = result[0].bodyBox;
                cv::rectangle( image, cv::Point( detBox.xmin, detBox.ymin ), 
											cv::Point( detBox.xmax, detBox.ymax), 
															cv::Scalar( 0, 255, 255 ), 1, 8 );
                cv::imwrite(configParam.cropfaceDir + tempString, image);
                #endif

                outfile << RegisterName;
                for(unsigned nn = 0; nn< result[0].reidfeature.featureFace.size(); nn++)
                    outfile << " "<<result[0].reidfeature.featureFace[nn];
            }
            outfile <<std::endl; 
        }
        outfile.close();
    }

    void dataBase::generateBaseHOGFeature(faceAnalysis faceRegister){
        std::vector<string> filenames;
	    GetFileNames(m_BaseDir,filenames);
        std::cout<<"total base: "<<filenames.size()<<endl;
        std::ofstream outfile;
        outfile.open(configParam.HOGfacefeaturefile, ios::out|ios::trunc);
        outfile.clear();
        for(unsigned i = 0; i < filenames.size(); i++){
            std::cout<<"****image name: "<<filenames[i]<<std::endl;
            cv::Mat image = cv::imread(filenames[i]);
            if (!image.data)
            {
                printf("No image data\n");
                return ;
            }
            std::vector<output> result = faceRegister.faceDetector(image);
            if (result.size() > 1){
                std::cout << "file =" << __FILE__ << ", line =" << __LINE__ << ", faceDetect result more than one: "
                                        <<result.size()<<" image name: "<<filenames[i]<<std::endl;
            }else{
                std::string tempString = filenames[i].substr(filenames[i].find_last_of("/")+1);
                std::string RegisterName = tempString.substr(0, tempString.find_last_of("."));
                box detBox = result[0].second;
                cv::Mat RoiImg = image(cv::Rect(detBox.xmin, detBox.ymin, detBox.xmax - detBox.xmin, 
                                    detBox.ymax- detBox.ymin));
                std::vector<float> HOGfeature= getHogFeatureMap(RoiImg);
                
                outfile << RegisterName;
                int gender = 0;
                outfile << " "<< gender <<" ";
                for(unsigned nn = 0; nn< HOGfeature.size(); nn++)
                    outfile << " "<<HOGfeature[nn];
            }
            outfile <<std::endl; 
        }
        outfile.close();
    }

    FaceBase dataBase::getStoredDataBaseFeature(std::string basefeaturefile, int featureDim){
        std::ifstream infile(basefeaturefile.c_str());
        if (!infile.good()) {
            std::cout << "Cannot open " << basefeaturefile;
            fprintf(stderr, "error");
        }
        std::string lineStr ;
        std::stringstream sstr ;
        std::string name;
        int gender;
        float value;
        FaceBase base;
        encodeFeature current_feature;
        while(std::getline(infile, lineStr )){
            current_feature.featureFace.clear();
            sstr << lineStr;
            sstr >> name >> gender;
            
            for (int i = 0; i<featureDim; i++){
                sstr >> value;
                current_feature.featureFace.push_back(value);
            }
            if(base.find(gender) == base.end()){
                vector_feature new_feature;
                new_feature.push_back(std::make_pair(name, current_feature));
                base.insert(std::make_pair(gender, new_feature));
            }else{
                vector_feature feature_list = base.find(gender)->second;
                feature_list.push_back(std::make_pair(name, current_feature));
                base[gender] = feature_list;
            }
            sstr.clear();
        }
        infile.close();
        std::cout<<"male size: "<<base.find(0)->second.size() << ", female size: " << base.find(1)->second.size();
        return base;
    }

    vector_feature dataBase::getStoredReidFeature(std::string basefeaturefile, int featureDim){
        std::ifstream infile(basefeaturefile.c_str());
        if (!infile.good()) {
            std::cout << "Cannot open " << basefeaturefile;
            fprintf(stderr, "error");
        }
        std::string lineStr ;
        std::stringstream sstr ;
        std::string name;
        float value;
        vector_feature base;
        encodeFeature current_feature;
        while(std::getline(infile, lineStr )){
            current_feature.featureFace.clear();
            sstr << lineStr;
            sstr >> name;
            
            for (int i = 0; i<featureDim; i++){
                sstr >> value;
                current_feature.featureFace.push_back(value);
            }
            base.push_back(std::make_pair(name, current_feature));
            sstr.clear();
        }
        infile.close();
        std::cout<<"database size: "<<base.size() << endl;
        return base;
    }

    void dataBase::generateBasebodyHOGFeature(reidAnalysis reidRegister){
    std::vector<string> filenames;
    GetFileNames(m_BaseDir,filenames);
    std::cout<<"total base: "<<filenames.size()<<endl;
    std::ofstream outfile;
    outfile.open(configParam.HOGfacefeaturefile, ios::out|ios::trunc);
    outfile.clear();
    for(unsigned i = 0; i < filenames.size(); i++){
        std::cout<<"****image name: "<<filenames[i]<<std::endl;
        cv::Mat image = cv::imread(filenames[i]);
        if (!image.data)
        {
            printf("No image data\n");
            return ;
        }
        std::vector<output> result = reidRegister.bodyDetector(image);
        if (result.size() > 1){
            std::cout << "file =" << __FILE__ << ", line =" << __LINE__ << ", faceDetect result more than one: "
                                    <<result.size()<<" image name: "<<filenames[i]<<std::endl;
        }else{
            std::string tempString = filenames[i].substr(filenames[i].find_last_of("/")+1);
            std::string RegisterName = tempString.substr(0, tempString.find_last_of("."));
            box detBox = result[0].second;
            cv::Mat RoiImg = image(cv::Rect(detBox.xmin, detBox.ymin, detBox.xmax - detBox.xmin, 
                                detBox.ymax- detBox.ymin));
            std::vector<float> HOGfeature= getHogFeatureMap(RoiImg);
            
            outfile << RegisterName;
            for(unsigned nn = 0; nn< HOGfeature.size(); nn++)
                outfile << " "<<HOGfeature[nn];
        }
        outfile <<std::endl; 
    }
    outfile.close();
    }

    dataBase::~dataBase()
    {
    }
}