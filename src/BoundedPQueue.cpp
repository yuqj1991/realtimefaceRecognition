#include "BoundedPQueue.hpp"

/** BoundedPQueue class implementation details */

template <typename T>
BoundedPQueue<T>::BoundedPQueue(std::size_t maxSize) {
    maximumSize = maxSize;
}

// enqueue adds the element to the map, then deletes the last element of the
// map if there size exceeds the maximum size.
template <typename T>
void BoundedPQueue<T>::enqueue(const T& value, double priority) {
    // Add the element to the collection.
    elems.insert(std::make_pair(priority, value));

    // If there are too many elements in the queue, drop off the last one.
    if (size() > maxSize()) {
        typename std::multimap<double, T>::iterator last = elems.end();
        --last; // Now points to highest-priority element
        elems.erase(last);
    }
}

// dequeueMin copies the lowest element of the map (the one pointed at by
// begin()) and then removes it.
template <typename T>
T BoundedPQueue<T>::dequeueMin() {
    // Copy the best value.
    T result = elems.begin()->second;

    // Remove it from the map.
    elems.erase(elems.begin());

    return result;
}

// size() and empty() call directly down to the underlying map.
template <typename T>
std::size_t BoundedPQueue<T>::size() const {
    return elems.size();
}

template <typename T>
bool BoundedPQueue<T>::empty() const {
    return elems.empty();
}

// maxSize just returns the appropriate data member.
template <typename T>
std::size_t BoundedPQueue<T>::maxSize() const {
    return maximumSize;
}

// The best() and worst() functions check if the queue is empty,
// and if so return infinity.
template <typename T>
double BoundedPQueue<T>::best() const {
    return empty()? std::numeric_limits<double>::infinity() : elems.begin()->first;
}

template <typename T>
double BoundedPQueue<T>::worst() const {
    return empty()? std::numeric_limits<double>::infinity() : elems.rbegin()->first;
}