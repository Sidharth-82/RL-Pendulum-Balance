#include "core/actuator.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>

namespace core {

Actuator::Actuator(const StepperLimits& limits, int latency_steps) : limits_(limits) {
    command_delay_.set_delay(latency_steps);
    reset();
}

void Actuator::reset(double position, double velocity) {
    position_ = position;
    velocity_ = velocity;
    command_delay_.reset(0.0);
}

double Actuator::apply(double accel_cmd, double dt) {
    assert(dt > 0.0);

    // 1-2: delayed command, clamped by the step-rate slew limit.
    const double cmd = std::clamp(command_delay_.push(accel_cmd),
                                  -limits_.max_accel, limits_.max_accel);

    // 3: integrate into velocity, clamped by the top step rate.
    const double previous_velocity = velocity_;
    velocity_ = std::clamp(velocity_ + cmd * dt,
                           -limits_.max_velocity, limits_.max_velocity);

    // 4: what the carriage actually did, which is what Plant::accel must see --
    // handing it the command instead would credit the pendulum with authority the
    // motor never delivered.
    const double achieved = (velocity_ - previous_velocity) / dt;

    // 5: because position_ advances from clamped motion, the dead-reckoned belief
    // already accounts for saturation -- exactly like firmware that knows how many
    // pulses it emitted. That leaves step loss as the only source of divergence
    // between belief and truth.
    //
    // Trapezoidal, matching Plant::step's x += xdot*dt + (1/2) a dt^2 exactly. A
    // semi-implicit update (position += velocity_ * dt) differs by (1/2) a dt^2 per
    // step, which sounds negligible and is not: it accumulates across an episode and
    // makes belief drift from truth with no step loss involved, quietly breaking the
    // invariant this comment claims.
    position_ += 0.5 * (previous_velocity + velocity_) * dt;

    return achieved;
}

double Actuator::available_force() const {
    // Linear droop: flat at holding_force up to the knee, falling to zero at
    // max_velocity. Crude, but a speed-independent force model makes a simulated
    // stepper look far stronger than the real one precisely where it matters.
    const double speed = std::abs(velocity_);
    const double knee = std::min(limits_.force_knee_speed, limits_.max_velocity);
    if (speed <= knee) {
        return limits_.holding_force;
    }

    const double span = limits_.max_velocity - knee;
    if (span <= 0.0) {
        return 0.0;  // degenerate limits: no droop region to interpolate across
    }

    const double force = limits_.holding_force * (limits_.max_velocity - speed) / span;
    return std::clamp(force, 0.0, limits_.holding_force);
}

bool Actuator::would_lose_steps(double belt_force) const {
    return std::abs(belt_force) > available_force();
}

bool Actuator::off_rail() const {
    return std::abs(position_) > 0.5 * limits_.rail_length;
}

double Actuator::position() const {
    return position_;
}

double Actuator::velocity() const {
    return velocity_;
}

double Actuator::quantise(double position) const {
    // The carriage cannot sit between steps; at coarse belt pitches this is a real
    // resolution floor on position control.
    if (limits_.steps_per_metre <= 0.0) {
        return position;  // resolution unconfigured -- treat as continuous
    }
    return std::round(position * limits_.steps_per_metre) / limits_.steps_per_metre;
}

void Actuator::set_limits(const StepperLimits& limits) {
    limits_ = limits;
}

}  // namespace core
