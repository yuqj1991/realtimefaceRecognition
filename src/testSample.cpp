#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "faceAnalysis.hpp"
#include "kcftracker.hpp"
#include "utils_config.hpp"
#include "dataBase.hpp"
#include "ms_kdtree.hpp"
#include "kdtree.hpp"
#include<ctime>

using namespace cv;
using namespace RESIDEO;

int main(int argc, char* argv[]){
	faceAnalysis faceInfernece;

	dataBase baseface(faceDir, facefeaturefile);
#if 0
	baseface.generateBaseFeature(faceInfernece);
#else
	FaceBase dataColletcion = baseface.getStoredDataBaseFeature(facefeaturefile);
#endif
#if 1
/****************测试循环获取和第一种kd二叉树获取方式中时间节省方式**********************/
	std::vector<std::pair<Prediction, std::string > > trainData;
	Prediction goal;
	encodeFeature detFeature;
	int gender = 0;
	for(int i = 0; i < dataColletcion.size(); i++){
		vector_feature feature = dataColletcion[i];
		for(int j = 0; j < feature.size(); j++){
			if(feature[j].first!="1156174130")
				trainData.push_back(std::make_pair(feature[j].second.featureFace, feature[j].first));
			else{
				goal = feature[j].second.featureFace;
				gender = i;
				detFeature = feature[j].second;
			}
			
		}
	}
	KDtreeNode *kdtree = new KDtreeNode;
	buildKdtree(kdtree, trainData);
	clock_t startTime,endTime;
 	startTime = clock();//计时开始
	std::pair<float, std::string > nearestNeighbor = searchNearestNeighbor(goal, kdtree);
	endTime = clock();//计时结束
	std::cout << "kd tree run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    std::cout<<"the kd method result: "<<nearestNeighbor.second<<std::endl;
	startTime = clock();//计时开始
	std::pair<float, std::string>nearestNeighbor_loop= serachCollectDataNameByloop(dataColletcion,
             															detFeature, gender);
	std::string person = nearestNeighbor_loop.second;
	std::cout<<"the loop method result: "<<person<<std::endl;
	endTime = clock();//计时结束
	std::cout << "for recusive run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
/****************测试循环获取和第一种kd二叉树获取方式中时间节省方式**********************/
/****************测试循环获取和第三种map红黑二叉树获取方式中时间节省方式******************/
#if 1
	mapFaceCollectDataSet dataTestSet;
	FaceBase::iterator it;
    for(it = dataColletcion.begin(); it != dataColletcion.end(); it++){
        vector_feature feature = it->second;
        mapFeature subfeature;
        for(int j = 0; j < feature.size(); j++){
            subfeature.insert(std::make_pair(feature[j].second, feature[j].first));
        }
        int gender = it->first;
        dataTestSet[gender] = subfeature;
    }
    int num = 0;
    mapFaceCollectDataSet::iterator iter;
    for(iter = dataTestSet.begin(); iter != dataTestSet.end(); iter++){
        mapFeature subfeature = iter->second;
        num += subfeature.size();
    }
    std::cout<<"num: "<<num<<std::endl;
    startTime = clock();//计时开始
    person = serachCollectDataNameBymapSet(dataTestSet,
             detFeature, gender);
    endTime = clock();//计时结束
	std::cout << "map method run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
    std::cout<<"the third method result: "<<person<<std::endl;
#endif
/****************测试循环获取和第三种map红黑二叉树获取方式中时间节省方式******************/
#else
/****************测试循环获取和第二种kd二叉树获取方式中时间节省方式**********************/
    using dataset = vector<pair<Point<512>, std::string >>;
   dataset trainData;
    kdPoint<512> keypoint;
    void transformData(FaceBase rawData, dataset& data, kdPoint<512>* testpoint) {
        for (int idx = 0; idx < rawData.size(); idx++) {
            vector_feature feature = dataColletcion[i];
            std::vector<double> tmp;
            for(int j = 0; j < feature.size(); j++){
                tmp.clear();
                if(feature[j].first!="1156174130"){
                    tmp = feature[j].second.featureFace;
                    kdPoint<512> newPoint;
                    copy(tmp.begin(), tmp.end(), newPoint.begin());
                    data.push_back(make_pair(newPoint, feature[j].first));
                }else{
                    tmp = feature[j].second.featureFace;
                    copy(tmp.begin(), tmp.end(), testpoint->begin());
                }
            } 
        }
    }
    transformData(dataColletcion, trainData, &keypoint);
    KDTree<512, float> kd(trainData);
	clock_t startTime,endTime;
 	startTime = clock();//计时开始
	vector<float> nearestNeighbor = searchNearestNeighbor(goal, kdtree);
	endTime = clock();//计时结束
	std::cout << "kd tree run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
	#if 0
	for(int i =0; i< 512; i++){
		std::cout<<nearestNeighbor[i]<<" ";
	}
	std::cout<<std::endl;
	#endif
	startTime = clock();//计时开始
	std::pair<float, std::string>nearestNeighbor= serachCollectDataNameByloop(dataColletcion,
             															detFeature, gender);
	std::string person = nearestNeighbor.second;
	std::cout<<person<<std::endl;
	endTime = clock();//计时结束
	std::cout << "for recusive run time is: " <<(double)(endTime - startTime) / CLOCKS_PER_SEC << "s" << endl;
/****************测试循环获取和第二种kd二叉树获取方式中时间节省方式**********************/
#endif

}

#if 0
using dataset = vector<pair<Point<784>, unsigned int>>;

static const int kNumThreads = 8; // number of threads to use for kNN classification
static int numQueriesProcessed;
static int correctCount;
static mutex queryLock; // lock for global counters

// Perform kNN classification on data[start, end) using kd-tree, and update global counters
static void kNNQueryThread(int start, int end, const KDTree<784, unsigned int>& kd, size_t k, const dataset& data) {
    for (int i = start; i < end; i++) {
        const auto &p = data[i];
        unsigned int pred = kd.kNNValue(p.first, k);
        queryLock.lock();
        ++numQueriesProcessed;
        if (pred == p.second) ++correctCount;
        if (numQueriesProcessed % 500 == 0) cout << numQueriesProcessed << endl;
        queryLock.unlock();
    }
}

int main(int argc, char **argv) {
    // Load the MNIST dataset
    mnist_data *rawTrainData;
    unsigned int trainCnt;
    mnist_data *rawTestData;
    unsigned int testCnt;
    mnist_load("mnist_data/train-images-idx3-ubyte", "mnist_data/train-labels-idx1-ubyte", &rawTrainData, &trainCnt);
    mnist_load("mnist_data/t10k-images-idx3-ubyte", "mnist_data/t10k-labels-idx1-ubyte", &rawTestData, &testCnt);
    cout << "Finished loading data from disk!" << endl
        << "Training set size: " << trainCnt << endl
        << "Test set size: " << testCnt << endl;

    // Transform the loaded data to vector<pair<Point<784>, unsigned int>>
    dataset trainData;
    dataset testData;
    transformData(rawTrainData, trainCnt, trainData);
    transformData(rawTestData, testCnt, testData);
    cout << "Finished transforming dataset!" << endl;

    // Construct KD-Tree using training set
    KDTree<784, unsigned int> kd(trainData);
    cout << "Finished building KD-Tree!" << endl;

    // Sanity check on the training set
    cout << "Start Sanity Check: contains() should return true for training data, "
            << "and 1-NN training set accuracy should be perfect" << endl;
    bool sanityPass = true;
    for (int i = 0; i < 1000; i++) {
        if (!kd.contains(trainData[i].first) || kd.kNNValue(trainData[i].first, 1) != trainData[i].second) {
            sanityPass = false;
            break;
        }
    }
    if (sanityPass) cout << "Sanity check PASSED!" << endl;
    else cout << "Sanity check FAILED!" << endl;

    // Evaluate performance on test set
    size_t k = 3; // Number of nearest neighbors
    numQueriesProcessed = 0;
    correctCount = 0;
    int queriesPerThread = testCnt / kNumThreads;
    vector<thread> threads;
    cout << "Start evaluating kNN performance on the test set (" << "k = " << k << ")" << endl;
    auto c_start = clock();
    auto t_start = chrono::high_resolution_clock::now();

    for (int i = 0; i < kNumThreads; i++) {
        int start = i * queriesPerThread;
        int end = (i == kNumThreads-1) ? testCnt : start + queriesPerThread;
        threads.push_back(thread(kNNQueryThread, start, end, ref(kd), k, ref(testData)));
    }
    for (thread &t : threads) t.join();

    clock_t c_end = clock();
    auto t_end = chrono::high_resolution_clock::now();
    cout << "Test set accuracy: " << correctCount * 100.0 / testCnt << endl;
    cout << "CPU time elapsed in s: " << (double)(c_end - c_start) / CLOCKS_PER_SEC << endl;
    cout << "Wall time elapsed in s: " << chrono::duration<double>(t_end - t_start).count() << endl;

    free(rawTrainData);
    free(rawTestData);
    return 0;
}
#endif