#pragma once

#include <array>

#include "core/types.hpp"

namespace core {

// Fixed-capacity ring buffer for transport delay, used for both sensing lag and
// actuation lag.
//
// Latency is the single most destabilising element for this plant: the derivative
// gains are what loop delay eats first, and an inverted pendulum has no margin to
// spare. Modelling it explicitly -- and randomising it during training -- is what
// keeps kd_theta from being tuned against a plant that reacts instantly.
//
// Storage is a fixed std::array so this never allocates, which matters because
// the same class runs inside the Pi's real-time loop.
class DelayLine {
public:
    // delay_steps of 0 passes values through unchanged. Default-constructible on
    // purpose: SensorModel holds a std::array of these, which requires it.
    DelayLine(int delay_steps = 0, double initial = 0.0);

    void reset(double value = 0.0);

    // Pushes a new sample and returns the one from delay_steps ago.
    double push(double value);

    int delay_steps() const;

    // Resamples the delay between episodes for domain randomisation. Refills the
    // buffer rather than reinterpreting stale contents -- changing the delay
    // mid-episode would otherwise inject a spurious step into the signal.
    void set_delay(int steps, double fill_value = 0.0);

private:
    std::array<double, kMaxDelaySteps> buffer_{};
    int delay_steps_ = 0;
    int head_ = 0;
};

}  // namespace core
