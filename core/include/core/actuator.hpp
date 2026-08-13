#pragma once

#include "core/delay_line.hpp"
#include "core/types.hpp"

namespace core {

// Envelope of a belt-driven stepper. Everything here is a hardware limit the
// controller must stay inside, not a modelling nicety -- a policy trained without
// these will happily command accelerations the motor cannot produce.
struct StepperLimits {
    double max_velocity = 0.5;      // m/s, from max usable step rate / steps_per_metre
    double max_accel = 5.0;         // m/s^2, step-rate slew limit before steps are lost
    double steps_per_metre = 0.0;   // microsteps per rev * revs per metre of belt travel

    // Available force falls off with speed as back-EMF cuts into torque. A linear
    // droop from holding_force at rest to zero at max_velocity is crude but far
    // better than pretending force is speed-independent, which is what makes a
    // simulated stepper look far stronger than the real one at high speed.
    double holding_force = 10.0;    // N at zero speed
    double force_knee_speed = 0.2;  // m/s where the droop begins

    double rail_length = 1.0;       // m of usable travel; running out is a termination
};

// Converts a commanded carriage acceleration into what the hardware actually
// delivers, and reports when the command would have desynchronised the motor.
//
// A step/dir stepper is an open-loop position source with no encoder. If it is
// commanded past its envelope it silently loses steps, and the controller's belief
// about carriage position diverges from reality with no way to detect it. That is
// the dominant sim-to-real failure mode for this hardware, so step loss is a hard
// termination during training rather than a soft penalty -- the policy has to
// learn to stay inside the envelope, not to trade against it.
class Actuator {
public:
    explicit Actuator(const StepperLimits& limits, int latency_steps = 0);

    void reset(double position = 0.0, double velocity = 0.0);

    // Applies one control step: delays the command, clamps it to max_accel,
    // integrates to velocity, clamps to max_velocity, and returns the
    // acceleration actually achieved -- which is what should be handed to
    // Plant::accel, not the raw command.
    double apply(double accel_cmd, double dt);

    // Force available at the current speed, from the droop model above.
    double available_force() const;

    // True when the belt force the plant reported exceeds what the motor can hold
    // at this speed. Test against the peak across the RK4 stages.
    bool would_lose_steps(double belt_force) const;

    // True when the carriage has run past the usable rail. Also a termination:
    // there is no recovery once it hits the end stop.
    bool off_rail() const;

    double position() const;
    double velocity() const;

    // Quantises a position to whole microsteps. The carriage cannot sit between
    // steps, and at coarse belt pitches this is a real resolution floor on
    // position control.
    double quantise(double position) const;

    void set_limits(const StepperLimits& limits);  // domain randomisation between episodes

private:
    StepperLimits limits_;
    DelayLine command_delay_;
    double position_ = 0.0;
    double velocity_ = 0.0;
};

}  // namespace core
