#pragma once

#include <cstdint>
#include <random>
#include <vector>

#include "core/actuator.hpp"
#include "core/plant.hpp"
#include "core/sensors.hpp"

namespace sim {

// One episode of the swing-up-and-balance task, wrapping core.
//
// The policy sees the state and emits a carriage acceleration directly -- no gain
// vector, no mode switching, no handoff. Swing-up and balance are one learned
// behaviour, which is what makes this the smallest system that does the whole job.
//
// Desktop only. Nothing here ships to the Pi; the trained network does.

struct EnvConfig {
    core::PendulumParams plant;    // required -- there is no sensible default
    core::StepperLimits limits;
    core::EncoderSpec encoder;

    double dt = 1.0 / 500.0;       // plant integration period
    double episode_seconds = 8.0;

    // The policy acts every N plant steps, holding its command in between. This is
    // not a performance tweak -- at 500 Hz a discount of 0.99 looks 0.2 s ahead,
    // which cannot represent a swing-up that takes seconds. Decimating to 100 Hz
    // makes the same discount cover a full second, and it is also a zero-order hold,
    // so it is what the hardware does anyway.
    int control_decimation = 5;

    // Target configuration, one angle per link: 0 is upright, pi is hanging. Empty
    // means all upright. This is the whole of the "which links up, which down" goal
    // -- it enters the reward and the observation, and changes nothing structural.
    core::Vec theta_ref;

    // Where each episode starts. Default is hanging at rest, which is the task.
    core::Vec theta_start;         // empty => all pi (hanging)
    double theta_start_spread = 0.05;  // rad, uniform
    double omega_start_spread = 0.0;   // rad/s
    double x_start_spread = 0.0;       // m

    // false: the policy sees true angles and exact rates -- trains faster and is the
    // right place to start. true: everything goes through SensorModel, so angles are
    // quantised, noisy and delayed, and rates are filtered differences. Flip this on
    // before believing any sim-to-real claim.
    bool use_sensor_model = false;

    // Step loss is unrecoverable on open-loop hardware -- the belief diverges from
    // truth permanently with no way to detect it -- so it ends the episode rather
    // than costing reward.
    bool terminate_on_step_loss = true;

    // Reward weights. The angle term is normalised per link, so upright scores +1
    // per step regardless of link count.
    double position_weight = 0.1;
    double action_weight = 0.01;
    double rail_penalty = 100.0;

    double omega_scale = 10.0;     // observation normalisation, rad/s
};

struct StepResult {
    double reward = 0.0;
    bool done = false;       // episode ended for a real reason
    bool truncated = false;  // ran out of time; the state was still fine
    bool off_rail = false;
    bool step_loss = false;
};

class Env {
public:
    Env(const EnvConfig& config, std::uint64_t seed);

    // [x/(rail/2), xdot/max_velocity, then per link: sin, cos, omega/omega_scale]
    //
    // sin/cos rather than the angle itself: it removes the wrap discontinuity at pi
    // entirely, so nothing downstream has to know that 3pi and pi are the same pose.
    static int observation_size(int n_links) { return 2 + 3 * n_links; }
    int observation_size() const;
    static constexpr int kActionSize = 1;

    // Begins a new episode and returns the first observation.
    const std::vector<double>& reset();

    // action is in [-1, 1] and scales to +/- max_accel. Advances control_decimation
    // plant steps, accumulating reward across them.
    StepResult step(double action);

    const std::vector<double>& observation() const { return observation_; }
    const core::State& truth() const { return truth_; }
    const EnvConfig& config() const { return config_; }
    int plant_steps() const { return step_index_; }
    double elapsed() const { return step_index_ * config_.dt; }
    int max_plant_steps() const { return max_steps_; }

private:
    void write_observation();
    double step_reward(double action) const;

    EnvConfig config_;
    core::Plant plant_;
    core::Actuator actuator_;
    core::SensorModel sensors_;
    std::mt19937 rng_;

    core::Vec theta_ref_;
    core::Vec theta_start_;

    core::State truth_;
    core::Vec observed_theta_;
    core::Vec observed_omega_;
    std::vector<double> observation_;

    int step_index_ = 0;
    int max_steps_ = 0;
    bool finished_ = true;
};

}  // namespace sim
