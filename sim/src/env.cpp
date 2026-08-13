#include "sim/env.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace sim {
namespace {

core::Vec filled(const core::Vec& given, int n, double fallback) {
    if (given.size() == n) {
        return given;
    }
    assert(given.size() == 0);
    return core::Vec::Constant(n, fallback);
}

}  // namespace

Env::Env(const EnvConfig& config, std::uint64_t seed)
    : config_(config),
      plant_(config.plant, config.gravity),
      actuator_(config.limits, config.encoder.latency_steps),
      sensors_(config.encoder, config.plant.n_links(), config.dt),
      rng_(static_cast<std::mt19937::result_type>(seed)) {
    assert(config_.dt > 0.0);
    assert(config_.control_decimation >= 1);
    assert(config_.episode_seconds > 0.0);

    const int n = config_.plant.n_links();
    theta_ref_ = filled(config_.theta_ref, n, 0.0);        // upright
    theta_start_ = filled(config_.theta_start, n, core::kPi);  // hanging

    max_steps_ = static_cast<int>(config_.episode_seconds / config_.dt);
    observation_.assign(observation_size(n), 0.0);
    observed_theta_.setZero(n);
    observed_omega_.setZero(n);
}

int Env::observation_size() const {
    return observation_size(config_.plant.n_links());
}

const std::vector<double>& Env::reset() {
    const int n = config_.plant.n_links();
    std::uniform_real_distribution<double> theta_noise(-config_.theta_start_spread,
                                                      config_.theta_start_spread);
    std::uniform_real_distribution<double> omega_noise(-config_.omega_start_spread,
                                                       config_.omega_start_spread);
    std::uniform_real_distribution<double> x_noise(-config_.x_start_spread,
                                                   config_.x_start_spread);

    truth_.resize(n);
    truth_.x = x_noise(rng_);
    truth_.xdot = 0.0;
    for (int i = 0; i < n; ++i) {
        truth_.theta[i] = theta_start_[i] + theta_noise(rng_);
        truth_.omega[i] = omega_noise(rng_);
    }

    // The actuator's belief starts where the cart actually is -- on hardware this is
    // what homing against the limit switches establishes.
    actuator_.reset(truth_.x, truth_.xdot);
    sensors_.reset(truth_);

    step_index_ = 0;
    finished_ = false;
    write_observation();
    return observation_;
}

double Env::step_reward(double action) const {
    const int n = config_.plant.n_links();

    // +1 per link at its target, -1 opposite, smooth everywhere between. Uses the
    // cosine of the error rather than the error itself, so there is no wrap to get
    // wrong and no discontinuity for the optimiser to trip over.
    double aligned = 0.0;
    for (int i = 0; i < n; ++i) {
        aligned += std::cos(truth_.theta[i] - theta_ref_[i]);
    }
    aligned /= n;

    return aligned - config_.position_weight * truth_.x * truth_.x -
           config_.action_weight * action * action;
}

StepResult Env::step(double action) {
    assert(!finished_ && "call reset() before step()");

    action = std::clamp(action, -1.0, 1.0);
    const double command = action * config_.limits.max_accel;
    const double half_rail = 0.5 * config_.limits.rail_length;

    StepResult result;

    // The command is held across the decimation window -- a zero-order hold, exactly
    // as the hardware would apply it.
    for (int k = 0; k < config_.control_decimation; ++k) {
        const double achieved = actuator_.apply(command, config_.dt);

        double peak_belt_force = 0.0;
        plant_.step(truth_, achieved, config_.dt, &peak_belt_force);
        ++step_index_;

        // The sensing chain must advance on every plant step, not once per policy
        // step: its latency is measured in plant steps, and skipping updates would
        // silently shorten the delay it models.
        if (config_.use_sensor_model) {
            const core::Measurement m =
                sensors_.measure(truth_, actuator_.position(), actuator_.velocity(), rng_);
            observed_theta_ = m.theta;
            observed_omega_ = m.omega;
        }

        result.reward += step_reward(action);

        // Termination is judged on truth, not on the actuator's belief -- the rail
        // end is a physical fact, and after step loss the belief is exactly the thing
        // that has stopped being trustworthy.
        if (std::abs(truth_.x) > half_rail) {
            result.off_rail = true;
            result.done = true;
            result.reward -= config_.rail_penalty;
            break;
        }
        if (config_.terminate_on_step_loss && actuator_.would_lose_steps(peak_belt_force)) {
            result.step_loss = true;
            result.done = true;
            result.reward -= config_.rail_penalty;
            break;
        }
        if (step_index_ >= max_steps_) {
            result.truncated = true;
            break;
        }
    }

    finished_ = result.done || result.truncated;
    write_observation();
    return result;
}

void Env::write_observation() {
    const int n = config_.plant.n_links();
    const double half_rail = 0.5 * config_.limits.rail_length;

    // Carriage terms come from the actuator's dead-reckoned belief, never from truth:
    // no sensor observes the cart, and the belief is exactly what the Pi will have.
    observation_[0] = actuator_.position() / half_rail;
    observation_[1] = actuator_.velocity() / config_.limits.max_velocity;

    for (int i = 0; i < n; ++i) {
        const double theta = config_.use_sensor_model ? observed_theta_[i] : truth_.theta[i];
        const double omega = config_.use_sensor_model ? observed_omega_[i] : truth_.omega[i];
        observation_[2 + 3 * i] = std::sin(theta);
        observation_[3 + 3 * i] = std::cos(theta);
        observation_[4 + 3 * i] = omega / config_.omega_scale;
    }
}

}  // namespace sim
