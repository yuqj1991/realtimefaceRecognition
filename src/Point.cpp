#include "Point.hpp"

template <std::size_t N>
std::size_t kdPoint<N>::size() const {
    return N;
}

template <std::size_t N>
double& kdPoint<N>::operator[] (std::size_t index) {
    return coords[index];
}

template <std::size_t N>
double kdPoint<N>::operator[] (std::size_t index) const {
    return coords[index];
}

template <std::size_t N>
typename kdPoint<N>::iterator kdPoint<N>::begin() {
    return coords;
}

template <std::size_t N>
typename kdPoint<N>::const_iterator kdPoint<N>::begin() const {
    return coords;
}

template <std::size_t N>
typename kdPoint<N>::iterator kdPoint<N>::end() {
    return begin() + size();
}

template <std::size_t N>
typename kdPoint<N>::const_iterator kdPoint<N>::end() const {
    return begin() + size();
}

