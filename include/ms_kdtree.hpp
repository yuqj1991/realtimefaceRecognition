#ifndef KDtreeNode_INCLUDED_MS_HPP_
#define KDtreeNode_INCLUDED_MS_HPP_

#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>
#include "utils_config.hpp"
using namespace std;

/****************计算维度的标准方差*********************/
inline float computeVariance(std::vector<float> Dimfeature){
    float mean = 0.f, variance = 0.f;
    for(int i = 0; i < Dimfeature.size(); i++){
        mean += Dimfeature[i];
        variance += std::pow((Dimfeature[i]), 2.0);
    }
    mean *= float(1 / Dimfeature.size());
    variance *= float(1 / Dimfeature.size());
    variance = variance - std::pow(mean, 2.0);
    return variance;
}
/****************计算每个维度的中度值*******************/
inline float computeMedianValue(std::vector<std::pair<std::vector<float>, std::string > > points, int featureIdx){
    std::vector<float>dimfeature;
    int nrof_samples = points.size(); 
    for(int i = 0; i < nrof_samples; i++){
        dimfeature.push_back(points[i].first[featureIdx]);
    }
    std::sort(dimfeature.begin(), dimfeature.end());
    int pos = dimfeature.size() /2 ;
    return dimfeature[pos];
}
/****************计算样本中每个维度的方差值***************/
inline int choose_feature(std::vector<std::pair<std::vector<float>, std::string > > points){
    int nrof_samples = points.size(); 
    int N = points[0].first.size();
    std::vector<float> dimfeature;
    float variance_max = 0.f;
    int featureidx = 0;
    for(int i = 0; i < N; i++){
        dimfeature.clear();
        for(int j = 0; j < nrof_samples; j++)
            dimfeature.push_back(points[j].first[i]);
        float variance = computeVariance(dimfeature);
        if(variance_max < variance){
            variance_max = variance;
            featureidx = i;
        }
    }
    return featureidx;
}

struct KDtreeNode{
    std::pair<std::vector<float>, std::string > root;
    KDtreeNode* parent;
    KDtreeNode* leftChild;
    KDtreeNode* rightChild;
    float splitvalue;
    unsigned splitDim;
    //默认构造函数
    KDtreeNode(){parent = leftChild = rightChild = NULL;}
    //判断kd树是否为空
    bool isEmpty()
    {
        return root.first.empty();
    }
    //判断kd树是否只是一个叶子结点
    bool isLeaf()
    {
        return (!root.first.empty()) && 
            rightChild == NULL && leftChild == NULL;
    }
    //判断是否是树的根结点
    bool isRoot()
    {
        return (!isEmpty()) && parent == NULL;
    }
    //判断该子kd树的根结点是否是其父kd树的左结点
    bool isLeft()
    {
        return parent->leftChild->root == root;
    }
    //判断该子kd树的根结点是否是其父kd树的右结点
    bool isRight()
    {
        return parent->rightChild->root == root;
    }
};

void buildKdtree(KDtreeNode* tree, std::vector<std::pair<std::vector<float>, std::string > > points){
    //样本的数量
    unsigned samplesNum = points.size();
    //终止条件
    if (samplesNum == 0)
        return;
    if (samplesNum == 1)
    {
        tree->root = points[0];
        return;
    }
    std::cout<< "samplesNum: "<<samplesNum<<std::endl;
    //选择切分属性
    unsigned int splitAttribute = choose_feature(points);
    //选择切分值
    float splitValue = computeMedianValue(points, splitAttribute);
    std::cout<<"split dim: "<<splitAttribute<<" splitValue: "<<splitValue<<std::endl;
    std::vector<float> splitAttributeValues;
    for(unsigned i = 0; i<samplesNum; i++ )
        splitAttributeValues.push_back(points[i].first[splitAttribute]);
    
    /*******根据选定的切分属性和切分值，将数据集分为两个子集**/
    std::vector<std::pair<std::vector<float>, std::string > > subset_left;
    std::vector<std::pair<std::vector<float>, std::string > > subset_right;
    tree->splitDim = splitAttribute;
    tree->splitvalue = splitValue;
    for (unsigned i = 0; i < samplesNum; ++i){
        if (splitAttributeValues[i] == splitValue && tree->root.first.empty())
            tree->root = points[i];
        else{
            if (splitAttributeValues[i] < splitValue)
                subset_left.push_back(points[i]);
            else
                subset_right.push_back(points[i]);
        }
    }
    std::cout<<"subset_left size: "<<subset_left.size()<<" subset_right size: "<<subset_right.size()<<std::endl;
    /*******子集递归调用buildKDtreeNode函数*****************/
    tree->leftChild = new KDtreeNode;
    tree->leftChild->parent = tree;
    tree->rightChild = new KDtreeNode;
    tree->rightChild->parent = tree;
    buildKdtree(tree->leftChild, subset_left);
    buildKdtree(tree->rightChild, subset_right);
}

std::pair<float, std::string > searchNearestNeighbor(std::vector<float> goal, KDtreeNode *tree)
{
    /*第一步：在kd树中找出包含目标点的叶子结点：从根结点出发，
    递归的向下访问kd树，若目标点的当前维的坐标小于切分点的
    坐标，则移动到左子结点，否则移动到右子结点，直到子结点为
    叶结点为止,以此叶子结点为“当前最近点”*/

    unsigned k = tree->root.first.size();//计算出数据的维数
    KDtreeNode* currentTree = tree;
    std::pair<std::vector<float>, std::string > currentNearest = currentTree->root;
    while(!currentTree->isLeaf())
    {
        unsigned index = currentTree->splitvalue;//计算当前维
        if (currentTree->rightChild->isEmpty() || goal[index] < currentNearest.first[index]){
            currentTree = currentTree->leftChild;
        }else{
            currentTree = currentTree->rightChild;
        }
    }
    currentNearest = currentTree->root;

    /*第二步：递归地向上回退， 在每个结点进行如下操作：
    (a)如果该结点保存的实例比当前最近点距离目标点更近，则以该例点为“当前最近点”
    (b)当前最近点一定存在于某结点一个子结点对应的区域，检查该子结点的父结点的另
    一子结点对应区域是否有更近的点（即检查另一子结点对应的区域是否与以目标点为球
    心、以目标点与“当前最近点”间的距离为半径的球体相交）；如果相交，可能在另一
    个子结点对应的区域内存在距目标点更近的点，移动到另一个子结点，接着递归进行最
    近邻搜索；如果不相交，向上回退*/

    //当前最近邻与目标点的距离
    float currentDistance = computeDistance(goal, currentNearest.first, 0);

    //如果当前子kd树的根结点是其父结点的左孩子，则搜索其父结点的右孩子结点所代表的区域，反之亦反
    KDtreeNode* searchDistrict;
    if (currentTree->isLeft())
    {
        if (currentTree->parent->rightChild == NULL)
            searchDistrict = currentTree;
        else
            searchDistrict = currentTree->parent->rightChild;
    }else{
        searchDistrict = currentTree->parent->leftChild;
    }

    //如果搜索区域对应的子kd树的根结点不是整个kd树的根结点，继续回退搜索
    while (searchDistrict->parent != NULL){
        //搜索区域与目标点的最近距离
        float districtDistance = abs(goal[searchDistrict->splitvalue] - searchDistrict->parent->root.first[searchDistrict->splitvalue]);
        std::cout<<"districtDistance: "<<districtDistance<<std::endl;
        //如果“搜索区域与目标点的最近距离”比“当前最近邻与目标点的距离”短，表明搜索区域内可能存在距离目标点更近的点
        if (districtDistance < currentDistance ){//&& !searchDistrict->isEmpty()

            float parentDistance = computeDistance(goal, searchDistrict->parent->root.first, 0);
            std::cout<<"parentDistance: "<<parentDistance<<std::endl;
            if (parentDistance < currentDistance){
                currentDistance = parentDistance;
                currentTree = searchDistrict->parent;
                currentNearest = currentTree->root;
            }
            if (!searchDistrict->isEmpty()){
                float rootDistance = computeDistance(goal, searchDistrict->root.first, 0);
                std::cout<<"rootDistance: "<<rootDistance<<std::endl;
                if (rootDistance < currentDistance)
                {
                    currentDistance = rootDistance;
                    currentTree = searchDistrict;
                    currentNearest = currentTree->root;
                }
            }
            if (searchDistrict->leftChild != NULL && !searchDistrict->leftChild->isEmpty()){
                float leftDistance = computeDistance(goal, searchDistrict->leftChild->root.first, 0);
                std::cout<<"leftDistance: "<<leftDistance<<std::endl;
                if (leftDistance < currentDistance)
                {
                    currentDistance = leftDistance;
                    currentTree = searchDistrict;
                    currentNearest = currentTree->root;
                }
            }
            if (searchDistrict->rightChild != NULL && !searchDistrict->rightChild->isEmpty()){
                float rightDistance = computeDistance(goal, searchDistrict->rightChild->root.first, 0);
                std::cout<<"rightDistance: "<<rightDistance<<std::endl;
                if (rightDistance < currentDistance){
                    currentDistance = rightDistance;
                    currentTree = searchDistrict;
                    currentNearest = currentTree->root;
                }
            }
            std::cout<<"currentDistance: "<<currentDistance<<std::endl;
        }

        if (searchDistrict->parent->parent != NULL)
        {
            searchDistrict = searchDistrict->parent->isLeft()? 
                            searchDistrict->parent->parent->rightChild:
                            searchDistrict->parent->parent->leftChild;
        }else{
            searchDistrict = searchDistrict->parent;
        }
    }
    std::pair<float, std::string> result;
    result.first = currentDistance;
    result.second = currentNearest.second;
    return result;
}

#endif