#include "core/types.hpp"

#include <cassert>

namespace core {

void State::resize(int n_links) {
    assert(n_links >= 0 && n_links <= kMaxLinks);
    theta.setZero(n_links);
    omega.setZero(n_links);
    x = 0.0;
    xdot = 0.0;
}

int State::n_links() const {
    return static_cast<int>(theta.size());
}

int State::flat_size(int n_links) {
    return 2 * n_links + 2;
}

Eigen::VectorXd State::flatten() const {
    // Returns a heap-backed VectorXd on purpose: this crosses into LQR and RL,
    // where sizes are not bounded by kMaxLinks and the call is not hot.
    const int n = n_links();
    assert(omega.size() == theta.size());

    Eigen::VectorXd v(flat_size(n));
    v(0) = x;
    v(1) = xdot;
    v.segment(2, n) = theta;
    v.segment(2 + n, n) = omega;
    return v;
}

void State::unflatten(const Eigen::Ref<const Eigen::VectorXd>& v, int n_links) {
    // A silent size mismatch here shifts every gain onto the wrong signal.
    assert(v.size() == flat_size(n_links));

    resize(n_links);
    x = v(0);
    xdot = v(1);
    theta = v.segment(2, n_links);
    omega = v.segment(2 + n_links, n_links);
}

}  // namespace core
