#include "dataBase.hpp"
namespace RESIDEO{
    dataBase::dataBase(string faceDir, string saveFeatureFile):m_BaseDir(configParam.faceDir), 
                    m_faceFile(saveFeatureFile)
    {
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

    FaceBase dataBase::getStoredDataBaseFeature(std::string basefeaturefile){
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
            
            for (int i = 0; i<512; i++){
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
        std::cout<<"male size: "<<base.find(0)->second.size() << ", female size: %d\n" << base.find(1)->second.size();
        return base;
    }

    dataBase::~dataBase()
    {
    }
}