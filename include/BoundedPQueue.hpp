#ifndef BOUNDED_PQUEUE_INCLUDED_HPP_
#define BOUNDED_PQUEUE_INCLUDED_HPP_

#include <map>
#include <algorithm>
#include <limits>

template <typename T>
class BoundedPQueue {
public:
    // Constructor: BoundedPQueue(size_t maxSize);
    // Usage: BoundedPQueue<int> bpq(15);
    // --------------------------------------------------
    // Constructs a new, empty BoundedPQueue with
    // maximum size equal to the constructor argument.
    ///
    explicit BoundedPQueue(std::size_t maxSize);

    // void enqueue(const T& value, double priority);
    // Usage: bpq.enqueue("Hi!", 2.71828);
    // --------------------------------------------------
    // Enqueues a new element into the BoundedPQueue with
    // the specified priority. If this overflows the maximum
    // size of the queue, the element with the highest
    // priority will be deleted from the queue. Note that
    // this might be the element that was just added.
    void enqueue(const T& value, double priority);

    // T dequeueMin();
    // Usage: int val = bpq.dequeueMin();
    // --------------------------------------------------
    // Returns the element from the BoundedPQueue with the
    // smallest priority value, then removes that element
    // from the queue.
    T dequeueMin();

    // size_t size() const;
    // bool empty() const;
    // Usage: while (!bpq.empty()) { ... }
    // --------------------------------------------------
    // Returns the number of elements in the queue and whether
    // the queue is empty, respectively.
    std::size_t size() const;
    bool empty() const;

    // size_t maxSize() const;
    // Usage: size_t queueSize = bpq.maxSize();
    // --------------------------------------------------
    // Returns the maximum number of elements that can be
    // stored in the queue.
    std::size_t maxSize() const;

    // double best() const;
    // double worst() const;
    // Usage: double highestPriority = bpq.worst();
    // --------------------------------------------------
    // best() returns the smallest priority of an element
    // stored in the container (i.e. the priority of the
    // element that will be dequeued first using dequeueMin).
    // worst() returns the largest priority of an element
    // stored in the container.  If an element is enqueued
    // with a priority above this value, it will automatically
    // be deleted from the queue.  Both functions return
    // numeric_limits<double>::infinity() if the queue is
    // empty.
    double best()  const;
    double worst() const;

private:
    // This class is layered on top of a multimap mapping from priorities
    // to elements with those priorities.
    std::multimap<double, T> elems;
    std::size_t maximumSize;
};

#endif // BOUNDED_PQUEUE_INCLUDED
