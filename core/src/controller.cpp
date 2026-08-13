#include "core/controller.hpp"

#include <algorithm>
#include <cassert>

namespace core {

int Gains::n_links() const {
    return static_cast<int>(kp_theta.size());
}

int Gains::size(int n_links) {
    return 2 * n_links + 3;
}

Gains Gains::from_flat(const Eigen::Ref<const Eigen::VectorXd>& v, int n_links) {
    // This is the RL action boundary; a layout mismatch produces a controller that
    // is wrong but still looks plausible.
    assert(n_links >= 0 && n_links <= kMaxLinks);
    assert(v.size() == size(n_links));

    Gains g;
    g.kp_x = v(0);
    g.kd_x = v(1);
    g.ki_x = v(2);
    g.kp_theta = v.segment(3, n_links);
    g.kd_theta = v.segment(3 + n_links, n_links);
    return g;
}

Eigen::VectorXd Gains::to_flat() const {
    const int n = n_links();
    assert(kd_theta.size() == kp_theta.size());

    Eigen::VectorXd v(size(n));
    v(0) = kp_x;
    v(1) = kd_x;
    v(2) = ki_x;
    v.segment(3, n) = kp_theta;
    v.segment(3 + n, n) = kd_theta;
    return v;
}

Eigen::VectorXd augmented_state(const State& state, double integral_x, const Vec& theta_ref) {
    const int n = state.n_links();
    assert(state.omega.size() == state.theta.size());
    assert(theta_ref.size() == 0 || theta_ref.size() == n);

    Eigen::VectorXd z(Gains::size(n));
    z(0) = state.x;
    z(1) = state.xdot;
    z(2) = integral_x;
    for (int i = 0; i < n; ++i) {
        const double reference = (theta_ref.size() == 0) ? 0.0 : theta_ref[i];
        z(3 + i) = wrap_angle(state.theta[i] - reference);
    }
    z.segment(3 + n, n) = state.omega;
    return z;
}

Controller::Controller(const Gains& gains, double dt, double integral_limit)
    : gains_(gains), dt_(dt), integral_limit_(integral_limit), integral_x_(0.0) {
    // dt must equal the real control period -- ki_x carries 1/s and kd_x carries s,
    // so a rate mismatch between tuning and deployment makes them numerically wrong
    // even with perfect physics.
    assert(dt > 0.0);
    assert(integral_limit >= 0.0);
    theta_ref_.setZero(gains_.n_links());  // all upright until told otherwise
}

void Controller::reset() {
    integral_x_ = 0.0;
}

double Controller::compute(double x, double xdot, const Vec& theta, const Vec& omega) {
    // Arguments are measured state, not truth.
    assert(theta.size() == gains_.kp_theta.size());
    assert(omega.size() == gains_.kd_theta.size());
    assert(theta.size() == omega.size());

    // Clamping the accumulator is the simplest anti-windup that works; without any,
    // actuator saturation lets error pile up and recovery overshoots badly. Whatever
    // scheme is here has to be the one the Pi runs.
    integral_x_ = std::clamp(integral_x_ + x * dt_, -integral_limit_, integral_limit_);

    // Wrapped deviation from the target equilibrium, not the raw angle. With the
    // default all-upright reference this is identical to theta itself.
    double link_terms = 0.0;
    for (int i = 0; i < theta.size(); ++i) {
        const double error = wrap_angle(theta[i] - theta_ref_[i]);
        link_terms += gains_.kp_theta[i] * error + gains_.kd_theta[i] * omega[i];
    }

    return -(gains_.kp_x * x + gains_.kd_x * xdot + gains_.ki_x * integral_x_ + link_terms);
}

void Controller::set_gains(const Gains& gains) {
    // Does not reset the integral -- the caller decides, since mid-episode gain
    // changes should not discard accumulated position error.
    gains_ = gains;
    if (theta_ref_.size() != gains_.n_links()) {
        theta_ref_.setZero(gains_.n_links());
    }
}

const Gains& Controller::gains() const {
    return gains_;
}

void Controller::set_theta_ref(const Vec& theta_ref) {
    assert(theta_ref.size() == gains_.n_links());
    theta_ref_ = theta_ref;
}

const Vec& Controller::theta_ref() const {
    return theta_ref_;
}

double Controller::integral() const {
    return integral_x_;
}

}  // namespace core
