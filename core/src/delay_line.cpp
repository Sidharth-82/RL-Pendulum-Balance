#include "core/delay_line.hpp"

#include <algorithm>

namespace core {

DelayLine::DelayLine(int delay_steps, double initial) {
    delay_steps_ = std::clamp(delay_steps, 0, kMaxDelaySteps - 1);
    reset(initial);
}

void DelayLine::reset(double value) {
    // Filling rather than just clearing matters: a partially stale buffer injects
    // a spurious step into the delayed signal at the start of an episode.
    buffer_.fill(value);
    head_ = 0;
}

double DelayLine::push(double value) {
    if (delay_steps_ == 0) {
        return value;
    }
    // Read before write: head_ holds the oldest sample, which is the one from
    // delay_steps_ pushes ago.
    const double delayed = buffer_[head_];
    buffer_[head_] = value;
    head_ = (head_ + 1) % delay_steps_;
    return delayed;
}

int DelayLine::delay_steps() const {
    return delay_steps_;
}

void DelayLine::set_delay(int steps, double fill_value) {
    // Refills rather than reinterpreting stale contents -- changing the delay
    // would otherwise inject a spurious step into the signal. reset() also puts
    // head_ back in range, which the new, possibly shorter, length requires.
    delay_steps_ = std::clamp(steps, 0, kMaxDelaySteps - 1);
    reset(fill_value);
}

}  // namespace core
