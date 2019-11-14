#ifndef KDtreeNode_INCLUDED_MS_HPP_
#define KDtreeNode_INCLUDED_MS_HPP_

#include <cmath>
#include <vector>
#include <utility>
#include <algorithm>
#include "utils_config.hpp"
using namespace std;


struct KDtreeNode{
    std::pair<Prediction, std::string > root;
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

void buildKdtree(KDtreeNode* tree, std::vector<std::pair<Prediction, std::string > > points, unsigned depth){
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
    //选择切分属性,某一维
    unsigned int splitAttribute = choose_feature(points);
    //选择切分值，中值
    float splitValue = computeMedianValue(points, splitAttribute);
    Prediction splitAttributeValues;
    for(unsigned i = 0; i<samplesNum; i++ )
        splitAttributeValues.push_back(points[i].first[splitAttribute]);
    
    /*******根据选定的切分属性和切分值，将数据集分为两个子集**/
    std::vector<std::pair<Prediction, std::string > > subset_left;
    std::vector<std::pair<Prediction, std::string > > subset_right;
    tree->splitDim = splitAttribute;
    tree->splitvalue = splitValue;
    
    for (unsigned i = 0; i < samplesNum; ++i){
        if (splitAttributeValues[i] == splitValue && tree->root.first.empty()){
            tree->root = points[i];
        }else{
            if (splitAttributeValues[i] < splitValue)
                subset_left.push_back(points[i]);
            else
                subset_right.push_back(points[i]);
        }
    }
    /*******子集递归调用buildKDtreeNode函数*****************/
    tree->leftChild = new KDtreeNode;
    tree->leftChild->parent = tree;
    tree->rightChild = new KDtreeNode;
    tree->rightChild->parent = tree;
    buildKdtree(tree->leftChild, subset_left, depth + 1);
    buildKdtree(tree->rightChild, subset_right, depth + 1);
}

void getNearestNode(Prediction goal, KDtreeNode *tree, float *Distance, KDtreeNode *currentTree){
    float currentDistance = 0.f;
    float parentDistance = computeDistance(goal, tree->root.first, 0);
    std::cout<<"Distance: "<< *Distance<<", parentDistance: "<<parentDistance<<", "<<tree->root.second<<std::endl;
    if(*Distance > parentDistance){
        currentDistance = parentDistance;
        currentTree = tree;
    }else{
        currentDistance = *Distance;
    }
    std::cout<<"currentDistance: "<< currentDistance<<", "<<currentTree->root.second<<std::endl;
    if(tree->leftChild != NULL  && !tree->leftChild->isEmpty()){
        getNearestNode(goal, tree ->leftChild, &currentDistance, currentTree);
    }
    if(tree->rightChild != NULL  && !tree->rightChild->isEmpty()){
        getNearestNode(goal, tree ->rightChild, &currentDistance, currentTree);
    }
}

std::pair<float, std::string > searchNearestNeighbor(Prediction goal, KDtreeNode *tree)
{
    std::pair<float, std::string> finalResult;
    /*
    第一步：在kd树中找出包含目标点的叶子结点：从根结点出发，
    递归的向下访问kd树，若目标点的当前维的坐标小于切分点的
    坐标，则移动到左子结点，否则移动到右子结点，直到子结点为
    叶结点为止,以此叶子结点为“当前最近点”
    */
    KDtreeNode* currentTree = tree;
    std::pair<Prediction, std::string > currentNearest = currentTree->root;
    while(!currentTree->isLeaf())
    {
        unsigned index = currentTree->splitDim;//计算当前维
        if (currentTree->rightChild->isEmpty() || goal[index] < currentNearest.first[index]){
            currentTree = currentTree->leftChild;
        }else{
            currentTree = currentTree->rightChild;
        }
    }
    currentNearest = currentTree->root;

    /*
    第二步：递归地向上回退， 在每个结点进行如下操作：
    (a)如果该结点保存的实例比当前最近点距离目标点更近，则以该例点为“当前最近点”
    (b)当前最近点一定存在于某结点一个子结点对应的区域，检查该子结点的父结点的另
    一子结点对应区域是否有更近的点（即检查另一子结点对应的区域是否与以目标点为球
    心、以目标点与“当前最近点”间的距离为半径的球体相交）；如果相交，可能在另一
    个子结点对应的区域内存在距目标点更近的点，移动到另一个子结点，接着递归进行最
    近邻搜索；如果不相交，向上回退
    */
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
    std::cout<<"~~~~~~~~~~start serach~~~~~~~~~~~~~~~~~"<<std::endl;
    std::cout<<"raw currentDistance: "<<currentDistance<<", "<<currentNearest.second<<std::endl;
    //如果搜索区域对应的子kd树的根结点不是整个kd树的根结点，继续回退搜索
    while (searchDistrict->parent != NULL && !searchDistrict->parent->isEmpty()){
        //搜索区域与目标点的最近距离
        float districtDistance = abs(goal[searchDistrict->splitDim] - searchDistrict->parent->root.first[searchDistrict->splitDim]);
        std::cout<<"districtDistance: "<<districtDistance<<std::endl;
        //如果“搜索区域与目标点的最近距离”比“当前最近邻与目标点的距离”短，表明搜索区域内可能存在距离目标点更近的点
        if (districtDistance < currentDistance ){//&& !searchDistrict->isEmpty()

            float parentDistance = computeDistance(goal, searchDistrict->parent->root.first, 0);
            std::cout<<"parentDistance: "<<parentDistance<<", "<<searchDistrict->parent->root.second<<std::endl;
            if (parentDistance < currentDistance){
                currentDistance = parentDistance;
                currentTree = searchDistrict->parent;
                currentNearest = currentTree->root;
            }
            if (!searchDistrict->isEmpty()){
                float rootDistance = computeDistance(goal, searchDistrict->root.first, 0);
                std::cout<<"rootDistance: "<<rootDistance<<", "<<searchDistrict->root.second<<std::endl;
                if (rootDistance < currentDistance){
                    currentDistance = rootDistance;
                    currentTree = searchDistrict;
                    currentNearest = currentTree->root;
                }
            }
            if (searchDistrict->leftChild != NULL && !searchDistrict->leftChild->isEmpty()){
                float treeDistance = currentDistance;
                getNearestNode(goal, searchDistrict->leftChild, &treeDistance, currentTree);
                std::cout<<"left tree Distance: "<<treeDistance<<", currentDistance: "<<currentDistance<<std::endl;
                std::cout<<"name: "<<currentTree->root.second<<std::endl;
                if (treeDistance < currentDistance){
                    currentDistance = treeDistance;
                    currentNearest = currentTree->root;
                }            
            }
            if (searchDistrict->rightChild != NULL && !searchDistrict->rightChild->isEmpty()){
                float treeDistance = currentDistance;
                getNearestNode(goal, searchDistrict->rightChild, &treeDistance, currentTree);
                std::cout<<"right tree Distance: "<<treeDistance<<", currentDistance: "<<currentDistance<<std::endl;
                std::cout<<"name: "<<currentTree->root.second<<std::endl;
                if (treeDistance < currentDistance){
                    currentDistance = treeDistance;
                    currentNearest = currentTree->root;
                }
            } 
        }
        std::cout<<"**********loop************"<<std::endl;
        if (searchDistrict->parent->parent != NULL)
        {
            searchDistrict = searchDistrict->parent->isLeft()? 
                            searchDistrict->parent->parent->rightChild:
                            searchDistrict->parent->parent->leftChild;
        }else{
            searchDistrict = searchDistrict->parent;
        }
    }
    std::cout<<"****final currentDistance: "<<currentDistance<<", "<<currentNearest.second
        <<"******************"<<std::endl;
    std::cout<<std::endl;
    finalResult.first = currentDistance;
    finalResult.second = currentNearest.second;
    return finalResult;
}

void printKdTree(KDtreeNode *tree, unsigned depth)
{
    //for (unsigned i = 0; i < depth; ++i)
    cout << "\t";
            
    cout <<"depth: "<<depth<<", "<< tree->root.second << ",";
    cout << endl;
    if (tree->leftChild == NULL && tree->rightChild == NULL )//叶子节点
        return;
    else //非叶子节点
    {
        if (tree->leftChild != NULL)
        {
            cout << " left:";
            printKdTree(tree->leftChild, depth + 1);
        }    
        cout << endl;
        if (tree->rightChild != NULL)
        {
            //for (unsigned i = 0; i < depth + 1; ++i)
            //    cout << "\t";
            cout << "right:";
            printKdTree(tree->rightChild, depth + 1);
        }
        cout << endl;
    }
}

#endif