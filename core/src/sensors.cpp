#include "core/sensors.hpp"

#include <cassert>
#include <cmath>

namespace core {

// kTwoPi comes from types.hpp -- it was a file-local constant here until the angle
// wrapping in controller.cpp needed the same value.

SensorModel::SensorModel(const EncoderSpec& spec, int n_links, double dt)
    : n_(n_links), dt_(dt) {
    assert(n_links >= 1 && n_links <= kMaxLinks);
    assert(dt > 0.0);

    previous_theta_.setZero(n_);
    omega_filter_state_.setZero(n_);

    // set_spec derives filter_alpha_ and the delay lengths; doing it in one place
    // keeps construction and domain randomisation from drifting apart.
    set_spec(spec);
}

void SensorModel::reset(const State& truth) {
    assert(truth.n_links() == n_);

    for (int i = 0; i < n_; ++i) {
        // Seed with the *quantised* angle, which is what every later reading will
        // be. Seeding with raw truth leaves a sub-count step between the seed and
        // the first reading, and differentiating that injects a spurious rate of
        // up to half a count per dt into the opening of every episode.
        const double seed = quantise(truth.theta[i]);
        previous_theta_[i] = seed;
        theta_delay_[i].reset(seed);
    }
    omega_filter_state_.setZero();
}

double SensorModel::quantise(double angle) const {
    // 2^bits counts per revolution (AS5600 -> 12, AS5048A -> 14).
    assert(spec_.bits > 0);
    const double step = kTwoPi / std::exp2(static_cast<double>(spec_.bits));
    return std::round(angle / step) * step;
}

double SensorModel::differentiate(double value, double previous, double& filter_state) const {
    // Deliberately simple. Whatever runs here has to be reimplementable exactly on
    // the Pi, and a better estimator that cannot be is worth less than a crude one
    // that can.
    const double raw = (value - previous) / dt_;
    filter_state += filter_alpha_ * (raw - filter_state);
    return filter_state;
}

Measurement SensorModel::measure(const State& truth, double believed_x, double believed_xdot,
                                 std::mt19937& rng) {
    assert(truth.n_links() == n_);

    Measurement m;
    // x and xdot are the actuator's dead-reckoned belief and pass through
    // untouched -- no sensor observes them.
    m.x = believed_x;
    m.xdot = believed_xdot;
    m.theta.resize(n_);
    m.omega.resize(n_);

    // A zero noise_std is the default and a legitimate configuration, but it is not
    // a legal distribution parameter, so the draw is skipped rather than scaled.
    const bool noisy = spec_.noise_std > 0.0;
    std::normal_distribution<double> noise(0.0, noisy ? spec_.noise_std : 1.0);

    for (int i = 0; i < n_; ++i) {
        double reading = quantise(truth.theta[i]);
        if (noisy) {
            reading += noise(rng);
        }

        // Differentiate the delayed, quantised, noisy angle -- not the true one.
        // Differentiating clean state and then delaying it would hide the exact
        // effect kd_theta is most sensitive to.
        reading = theta_delay_[i].push(reading);

        m.theta[i] = reading;
        m.omega[i] = differentiate(reading, previous_theta_[i], omega_filter_state_[i]);
        previous_theta_[i] = reading;
    }

    return m;
}

void SensorModel::set_spec(const EncoderSpec& spec) {
    spec_ = spec;

    // First-order low-pass at cutoff f_c: alpha = dt / (dt + 1/(2 pi f_c)).
    // A non-positive cutoff means "no filtering" -- tau = 0, alpha = 1 -- rather
    // than a division by zero.
    const double tau = spec_.derivative_cutoff_hz > 0.0
                           ? 1.0 / (kTwoPi * spec_.derivative_cutoff_hz)
                           : 0.0;
    filter_alpha_ = dt_ / (dt_ + tau);

    for (auto& line : theta_delay_) {
        line.set_delay(spec_.latency_steps);
    }
}

}  // namespace core
