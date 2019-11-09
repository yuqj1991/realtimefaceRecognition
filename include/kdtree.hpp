#ifndef KDTREE_INCLUDED_HPP_
#define KDTREE_INCLUDED_HPP_
using namespace std;

#include "Point.hpp"
#include "BoundedPQueue.hpp"
#include <stdexcept>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <utility>
#include <algorithm>

template <std::size_t N, typename ElemType>
class KDTree {
public:

    // Constructs an empty KDTree.
    KDTree();

    // Efficiently build a balanced KD-tree from a large set of kdPoints
    KDTree(std::vector<std::pair<kdPoint<N>, ElemType>>& kdPoints);

    // Frees up all the dynamically allocated resources
    ~KDTree();

    // Deep-copies the contents of another KDTree into this one.
    KDTree(const KDTree& rhs);
    KDTree& operator=(const KDTree& rhs);

    // Returns the dimension of the kdPoints stored in this KDTree.
    std::size_t dimension() const;

    // Returns the number of elements in the kd-tree and whether the tree is empty
    std::size_t size() const;
    bool empty() const;

    // Returns whether the specified kdPoint is contained in the KDTree.
    bool contains(const kdPoint<N>& pt) const;

    /*
     * Inserts the kdPoint pt into the KDTree, associating it with the specified value.
     * If the element already existed in the tree, the new value will overwrite the existing one.
     */
    void insert(const kdPoint<N>& pt, const ElemType& value=ElemType());

    /*
     * Returns a reference to the value associated with kdPoint pt in the KDTree.
     * If the kdPoint does not exist, then it is added to the KDTree using the
     * default value of ElemType as its key.
     */
    ElemType& operator[](const kdPoint<N>& pt);

    /*
     * Returns a reference to the key associated with the kdPoint pt. If the kdPoint
     * is not in the tree, this function throws an out_of_range exception.
     */
    ElemType& at(const kdPoint<N>& pt);
    const ElemType& at(const kdPoint<N>& pt) const;

    /*
     * Given a kdPoint v and an integer k, finds the k kdPoints in the KDTree
     * nearest to v and returns the most common value associated with those
     * kdPoints. In the event of a tie, one of the most frequent value will be chosen.
     */
    ElemType kNNValue(const kdPoint<N>& key, std::size_t k) const;

private:
    struct Node {
        kdPoint<N> kd_Point;
        Node *left;
        Node *right;
        int level;  // level of the node in the tree, starts at 0 for the root
        ElemType value;
        Node(const kdPoint<N>& _pt, int _level, const ElemType& _value=ElemType()):
            kd_Point(_pt), left(NULL), right(NULL), level(_level), value(_value) {}
    };

    // Root node of the KD-Tree
    Node* root_;

    // Number of kdPoints in the KD-Tree
    std::size_t size_;

    /*
     * Recursively build a subtree that satisfies the KD-Tree invariant using kdPoints in [start, end)
     * At each level, we split kdPoints into two halves using the median of the kdPoints as pivot
     * The root of the subtree is at level 'currLevel'
     * O(n) time partitioning algorithm is used to locate the median element
     */
    Node* buildTree(typename std::vector<std::pair<kdPoint<N>, ElemType>>::iterator start,
                    typename std::vector<std::pair<kdPoint<N>, ElemType>>::iterator end, int currLevel);

    void buildKdTree(KDTree* tree, std::vector<std::pair<kdPoint<N>, ElemType>>& kdPoints, unsigned depth);

    /*
     * Returns the Node that contains kdPoint pt if it is present in subtree 'currNode'
     * Returns the Node below which pt should be inserted if pt is not in the subtree
     */
    Node* findNode(Node* currNode, const kdPoint<N>& pt) const;

    // Recursive helper method for kNNValue(pt, k)
    void nearestNeighborRecurse(const Node* currNode, const kdPoint<N>& key, BoundedPQueue<ElemType>& pQueue) const;

    /*
     * Recursive helper method for copy constructor and assignment operator
     * Deep copies tree 'root' and returns the root of the copied tree
     */
    Node* deepcopyTree(Node* root);

    // Recursively free up all resources of subtree rooted at 'currNode'
    void freeResource(Node* currNode);

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
    inline float computeMedianValue(std::vector<std::pair<kdPoint<N>, ElemType>>& kdPoints, int featureIdx){
        std::vector<float>dimfeature;
        int nrof_samples = kdPoints.size(); 
        for(int i = 0; i < nrof_samples; i++){
            dimfeature.push_back(kdPoints[i].first[featureIdx]);
        }
        std::sort(dimfeature.begin(), dimfeature.end());
        int pos = dimfeature.size() /2 ;
        return dimfeature[pos];
    }
    /****************计算样本中每个维度的方差值***************/
    inline int choose_feature(std::vector<std::pair<kdPoint<N>, ElemType>>& kdPoints){
        int nrof_samples = kdPoints.size(); 
        std::vector<float> dimfeature(nrof_samples);
        float variance_max = 0.f;
        int featureidx = 0;
        for(int i = 0; i < N; i++){
            dimfeature.clear();
            for(int j = 0; j < nrof_samples; j++)
                dimfeature[j] = kdPoints[j].first[i];
            float variance = computeVariance(dimfeature);
            if(variance_max < variance){
                variance_max = variance;
                featureidx = i;
            }
        }
        return featureidx;
    }
};

#endif