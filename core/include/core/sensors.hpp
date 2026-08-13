#pragma once

#include <array>
#include <random>
#include <algorithm>

#include "core/delay_line.hpp"
#include "core/types.hpp"

namespace core {

// Magnetic rotary encoder on each joint. AS5600 is 12-bit, AS5048A is 14-bit.
//
// Chosen over potentiometers because they add no joint friction (which compounds
// with N), never wear, are absolute so no homing is needed, and do not require an
// ADC -- the Pi has none.
//
// Part numbers and filter corner come from config.json under "encoder". The
// initialisers below are test fixtures.
struct EncoderSpec {
    int bits = 12;             // quantisation, counts = 2^bits per revolution
    double noise_std = 0.0;    // rad, per reading
    int latency_steps = 0;     // sensing lag in control steps

    // Corner of the first-order low-pass on the differentiated angle. This is the
    // most consequential number in this struct: theta is quantised and noisy, and
    // differentiating it amplifies both. Without a filter here, kd_theta tunes
    // against a clean derivative that does not exist on hardware and the real
    // system chatters. Whatever cutoff is used in training must be the one the Pi
    // runs.
    double derivative_cutoff_hz = 50.0;
};

// What the controller actually sees.
struct Measurement {
    double x = 0.0;
    double xdot = 0.0;
    Vec theta;
    Vec omega;
};

// Turns true plant state into what the hardware would report: quantised, noisy,
// delayed, with velocities estimated by filtered differentiation rather than
// known exactly.
//
// This class is why gains trained in simulation have any chance of transferring.
// The rigid-body dynamics are the accurate part of the model; the sensing chain
// is where the sim-to-real gap actually lives.
//
// Carriage position is regulated but never *measured*. An open-loop stepper knows
// where it is by counting the steps it emitted, which is exact -- not an estimate --
// right up until steps are lost, after which the belief diverges from truth
// permanently and with no correction path. Hence step loss being a termination
// rather than a penalty, and hence limit switches on the physical build: they give
// homing a physical reference and provide a stop that does not depend on the
// controller's belief being right.
//
// The belief therefore comes from Actuator, not from here, and is passed in so the
// data flow is visible at the call site rather than assembled silently by the env.
class SensorModel {
public:
    SensorModel(const EncoderSpec& spec, int n_links, double dt);

    // Seeds filter and delay state from the true initial state so the first few
    // measurements are not transients of the estimator itself.
    void reset(const State& truth);

    // One measurement. Link angles are quantised, noised and delayed from truth;
    // carriage position and velocity are the actuator's dead-reckoned belief and
    // pass through untouched, since no sensor observes them.
    //
    // rng is passed in rather than held so a rollout is reproducible from a seed.
    Measurement measure(const State& truth, double believed_x, double believed_xdot,
                        std::mt19937& rng);

    void set_spec(const EncoderSpec& spec);  // domain randomisation between episodes

private:
    // Rounds an angle to the nearest encoder count.
    double quantise(double angle) const;

    // First-order low-pass on the backward difference of successive quantised
    // angles. Deliberately simple: whatever runs here has to be reimplementable
    // exactly on the Pi, and a fancier estimator that cannot be is worth less
    // than a crude one that can.
    double differentiate(double value, double previous, double& filter_state) const;

    EncoderSpec spec_;
    int n_ = 0;
    double dt_ = 0.0;
    double filter_alpha_ = 0.0;  // derived from derivative_cutoff_hz and dt_

    Vec previous_theta_;
    Vec omega_filter_state_;
    std::array<DelayLine, kMaxLinks> theta_delay_;
};

}  // namespace core
