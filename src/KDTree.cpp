#include "kdtree.hpp"

template <std::size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree() :
    root_(NULL), size_(0) { }

template <std::size_t N, typename ElemType>
typename KDTree<N, ElemType>::Node* KDTree<N, ElemType>::deepcopyTree(typename KDTree<N, ElemType>::Node* root) {
    if (root == NULL) return NULL;
    Node* newRoot = new Node(*root);
    newRoot->left = deepcopyTree(root->left);
    newRoot->right = deepcopyTree(root->right);
    return newRoot;
}

template <std::size_t N, typename ElemType>
typename KDTree<N, ElemType>::Node* KDTree<N, ElemType>::buildTree(typename std::vector<std::pair<kdPoint<N>, ElemType>>::iterator start,
                                                                   typename std::vector<std::pair<kdPoint<N>, ElemType>>::iterator end, 
                                                                   int currLevel) {
    if (start >= end) return NULL;
    #if 1
    int axis = currLevel % N; // the axis to split on
    auto cmp = [axis](const std::pair<kdPoint<N>, ElemType>& p1, const std::pair<kdPoint<N>, ElemType>& p2) {
        return p1.first[axis] < p2.first[axis];
    };
    std::size_t len = end - start;
    auto mid = start + len / 2;
    std::nth_element(start, mid, end, cmp); // linear time partition
    #endif
    // move left (if needed) so that all the equal kdPoints are to the right
    // The tree will still be balanced as long as there aren't many kdPoints that are equal along each axis
    while (mid > start && (mid - 1)->first[axis] == mid->first[axis]) {
        --mid;
    }

    Node* newNode = new Node(mid->first, currLevel, mid->second);
    newNode->left = buildTree(start, mid, currLevel + 1);
    newNode->right = buildTree(mid + 1, end, currLevel + 1);
    return newNode;
}


template <std::size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree(std::vector<std::pair<kdPoint<N>, ElemType>>& kdPoints) {
    root_ = buildTree(kdPoints.begin(), kdPoints.end(), 0);
    size_ = kdPoints.size();
}

template <std::size_t N, typename ElemType>
KDTree<N, ElemType>::KDTree(const KDTree& rhs) {
    root_ = deepcopyTree(rhs.root_);
    size_ = rhs.size_;
}

template <std::size_t N, typename ElemType>
KDTree<N, ElemType>& KDTree<N, ElemType>::operator=(const KDTree& rhs) {
    if (this != &rhs) { // make sure we don't self-assign
        freeResource(root_);
        root_ = deepcopyTree(rhs.root_);
        size_ = rhs.size_;
    }
    return *this;
}

template <std::size_t N, typename ElemType>
void KDTree<N, ElemType>::freeResource(typename KDTree<N, ElemType>::Node* currNode) {
    if (currNode == NULL) return;
    freeResource(currNode->left);
    freeResource(currNode->right);
    delete currNode;
}

template <std::size_t N, typename ElemType>
KDTree<N, ElemType>::~KDTree() {
    freeResource(root_);
}

template <std::size_t N, typename ElemType>
std::size_t KDTree<N, ElemType>::dimension() const {
    return N;
}

template <std::size_t N, typename ElemType>
std::size_t KDTree<N, ElemType>::size() const {
    return size_;
}

template <std::size_t N, typename ElemType>
bool KDTree<N, ElemType>::empty() const {
    return size_ == 0;
}

template <std::size_t N, typename ElemType>
typename KDTree<N, ElemType>::Node* KDTree<N, ElemType>::findNode(typename KDTree<N, ElemType>::Node* currNode, const kdPoint<N>& pt) const {
    if (currNode == NULL || currNode->kdPoint == pt) return currNode;

    const kdPoint<N>& currkdPoint = currNode->kdPoint;
    int currLevel = currNode->level;
    if (pt[currLevel%N] < currkdPoint[currLevel%N]) { // recurse to the left side
        return currNode->left == NULL ? currNode : findNode(currNode->left, pt);
    } else { // recurse to the right side
        return currNode->right == NULL ? currNode : findNode(currNode->right, pt);
    }
}

template <std::size_t N, typename ElemType>
bool KDTree<N, ElemType>::contains(const kdPoint<N>& pt) const {
    auto node = findNode(root_, pt);
    return node != NULL && node->kdPoint == pt;
}

template <std::size_t N, typename ElemType>
void KDTree<N, ElemType>::insert(const kdPoint<N>& pt, const ElemType& value) {
    auto targetNode = findNode(root_, pt);
    if (targetNode == NULL) { // this means the tree is empty
        root_ = new Node(pt, 0, value);
        size_ = 1;
    } else {
        if (targetNode->kdPoint == pt) { // pt is already in the tree, simply update its value
            targetNode->value = value;
        } else { // construct a new node and insert it to the right place (child of targetNode)
            int currLevel = targetNode->level;
            Node* newNode = new Node(pt, currLevel + 1, value);
            if (pt[currLevel%N] < targetNode->kdPoint[currLevel%N]) {
                targetNode->left = newNode;
            } else {
                targetNode->right = newNode;
            }
            ++size_;
        }
    }
}

template <std::size_t N, typename ElemType>
const ElemType& KDTree<N, ElemType>::at(const kdPoint<N>& pt) const {
    auto node = findNode(root_, pt);
    if (node == NULL || node->kdPoint != pt) {
        throw std::out_of_range("kdPoint not found in the KD-Tree");
    } else {
        return node->value;
    }
}

template <std::size_t N, typename ElemType>
ElemType& KDTree<N, ElemType>::at(const kdPoint<N>& pt) {
    const KDTree<N, ElemType>& constThis = *this;
    return const_cast<ElemType&>(constThis.at(pt));
}

template <std::size_t N, typename ElemType>
ElemType& KDTree<N, ElemType>::operator[](const kdPoint<N>& pt) {
    auto node = findNode(root_, pt);
    if (node != NULL && node->kdPoint == pt) { // pt is already in the tree
        return node->value;
    } else { // insert pt with default ElemType value, and return reference to the new ElemType
        insert(pt);
        if (node == NULL) return root_->value; // the new node is the root
        else return (node->left != NULL && node->left->kdPoint == pt) ? node->left->value: node->right->value;
    }
}

template <std::size_t N, typename ElemType>
void KDTree<N, ElemType>::nearestNeighborRecurse(const typename KDTree<N, ElemType>::Node* currNode, 
                        const kdPoint<N>& key, BoundedPQueue<ElemType>& pQueue) const {
    if (currNode == NULL) return;
    const kdPoint<N>& currkdPoint = currNode->kdPoint;

    // Add the current kdPoint to the BPQ if it is closer to 'key' that some kdPoint in the BPQ
    pQueue.enqueue(currNode->value, Distance(currkdPoint, key));

    // Recursively search the half of the tree that contains kdPoint 'key'
    int currLevel = currNode->level;
    bool isLeftTree;
    if (key[currLevel%N] < currkdPoint[currLevel%N]) {
        nearestNeighborRecurse(currNode->left, key, pQueue);
        isLeftTree = true;
    } else {
        nearestNeighborRecurse(currNode->right, key, pQueue);
        isLeftTree = false;
    }

    if (pQueue.size() < pQueue.maxSize() || fabs(key[currLevel%N] - currkdPoint[currLevel%N]) < pQueue.worst()) {
        // Recursively search the other half of the tree if necessary
        if (isLeftTree) nearestNeighborRecurse(currNode->right, key, pQueue);
        else nearestNeighborRecurse(currNode->left, key, pQueue);
    }
}

template <std::size_t N, typename ElemType>
ElemType KDTree<N, ElemType>::kNNValue(const kdPoint<N>& key, std::size_t k) const {
    BoundedPQueue<ElemType> pQueue(k);
    if (empty()) return ElemType(); // default return value if KD-tree is empty

    // Recursively search the KD-tree with pruning
    nearestNeighborRecurse(root_, key, pQueue);

    // Count occurrences of all ElemType in the kNN set
    std::unordered_map<ElemType, int> counter;
    while (!pQueue.empty()) {
        ++counter[pQueue.dequeueMin()];
    }

    ElemType result;
    int cnt = -1;
    for (const auto &p : counter) {
        if (p.second > cnt) {
            result = p.first;
            cnt = p.second;
        }
    }
    return result;
}
