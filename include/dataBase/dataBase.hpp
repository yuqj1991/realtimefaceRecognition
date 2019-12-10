#ifndef _RESIDEO_DATABASE_H_
#define _RESIDEO_DATABASE_H_
#include "faceAnalysis/faceAnalysis.hpp"
#include "faceAnalysis/reidInference.hpp"
#include <string.h>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <algorithm>
#include "util/utils_config.hpp"
using namespace std;
namespace RESIDEO{
    class dataBase
    {
        private:
            util configParam;
        inline void GetFileNames(string path,vector<string>& filenames)
        {
            DIR *pDir;
            struct dirent* ptr;
            if(!(pDir = opendir(path.c_str()))){
                std::cout<<"Folder doesn't Exist!"<<endl;
                return;
            }
            while((ptr = readdir(pDir))!=0) {
                if (strcmp(ptr->d_name, ".") != 0 && strcmp(ptr->d_name, "..") != 0){
                    filenames.push_back(path + "/" + ptr->d_name);
            }
            }
            closedir(pDir);
        }
        
        public:
            explicit dataBase(string faceDir, string saveFeatureFile);
            ~dataBase();
            std::string m_BaseDir;
            std::string m_faceFile;
            void generateBaseFeature(faceAnalysis faceRegister);
            void generateBaseHOGFeature(faceAnalysis faceRegister);
            FaceBase getStoredDataBaseFeature(std::string basefeaturefile, int featureDim);

            void generatebodyFeature(reidAnalysis reidRegister);
            void generateBasebodyHOGFeature(reidAnalysis reidRegister);
            vector_feature getStoredReidFeature(std::string basefeaturefile, int featureDim);
    };
    
}

#endif