#ifndef kdPoint_INCLUDED_RESIDEO_HPP_
#define kdPoint_INCLUDED_RESIDEO_HPP_

#include <cmath>
#include <algorithm>

template <std::size_t N>
class kdPoint {
public:

    // Types representing iterators that can traverse and optionally modify the elements of the kdPoint.
    typedef double* iterator;
    typedef const double* const_iterator;

    // Returns N, the dimension of the kdPoint.
    std::size_t size() const;

    // Queries or retrieves the value of the kdPoint at a particular kdPoint. The index is assumed to be in-range.
    double& operator[](std::size_t index);
    double operator[](std::size_t index) const;

    iterator begin();
    iterator end();
    const_iterator begin() const;
    const_iterator end() const;

private:
    double coords[N];
};

template <std::size_t N>
double eucliDistance(const kdPoint<N>& one, const kdPoint<N>& two) {
    double result = 0.0;
    for (std::size_t i = 0; i < N; ++i)
        result += (one[i] - two[i]) * (one[i] - two[i]);
    return result;
}

template <std::size_t N>
bool operator==(const kdPoint<N>& one, const kdPoint<N>& two) {
    return std::equal(one.begin(), one.end(), two.begin());
}

template <std::size_t N>
bool operator!=(const kdPoint<N>& one, const kdPoint<N>& two) {
    return !(one == two);
}

#endif