// DelayLine is the smallest class here and the easiest to get subtly wrong: a
// delay line that silently passes values straight through looks like a working
// simulator right up until the gains it produced meet real loop lag.

#include "core/delay_line.hpp"

#include "check.hpp"

int main() {
    check::section("passthrough at zero delay");
    {
        core::DelayLine line(0);
        CHECK(line.delay_steps() == 0);
        CHECK_NEAR(line.push(1.0), 1.0, 0.0);
        CHECK_NEAR(line.push(-4.5), -4.5, 0.0);
        CHECK_NEAR(line.push(0.0), 0.0, 0.0);
    }

    check::section("one-step delay returns the previous sample");
    {
        // The tightest case: at delay 1 a read-after-write bug returns the value
        // just pushed, which is indistinguishable from passthrough.
        core::DelayLine line(1, 0.0);
        CHECK_NEAR(line.push(5.0), 0.0, 0.0);
        CHECK_NEAR(line.push(6.0), 5.0, 0.0);
        CHECK_NEAR(line.push(7.0), 6.0, 0.0);
    }

    check::section("n-step delay shifts by exactly n");
    {
        constexpr int kDelay = 4;
        core::DelayLine line(kDelay, 0.0);
        CHECK(line.delay_steps() == kDelay);

        // The fill value comes out for the first kDelay pushes, then the input
        // sequence reappears, offset by kDelay and otherwise untouched.
        for (int i = 0; i < kDelay; ++i) {
            CHECK_NEAR(line.push(static_cast<double>(i + 1)), 0.0, 0.0);
        }
        for (int i = 0; i < 20; ++i) {
            const double expected = static_cast<double>(i + 1);
            CHECK_NEAR(line.push(static_cast<double>(i + 1 + kDelay)), expected, 0.0);
        }
    }

    check::section("construction fill");
    {
        core::DelayLine line(3, 2.5);
        CHECK_NEAR(line.push(0.0), 2.5, 0.0);
        CHECK_NEAR(line.push(0.0), 2.5, 0.0);
        CHECK_NEAR(line.push(0.0), 2.5, 0.0);
        CHECK_NEAR(line.push(0.0), 0.0, 0.0);
    }

    check::section("reset refills the whole buffer");
    {
        core::DelayLine line(3, 0.0);
        line.push(1.0);
        line.push(2.0);  // buffer now partly stale

        line.reset(9.0);
        CHECK_NEAR(line.push(0.0), 9.0, 0.0);
        CHECK_NEAR(line.push(0.0), 9.0, 0.0);
        CHECK_NEAR(line.push(0.0), 9.0, 0.0);
        CHECK_NEAR(line.push(0.0), 0.0, 0.0);
    }

    check::section("set_delay lengthens, shortens and refills");
    {
        core::DelayLine line(2, 0.0);
        line.push(1.0);
        line.push(2.0);
        line.push(3.0);  // head is mid-buffer here

        line.set_delay(5, -1.0);
        CHECK(line.delay_steps() == 5);
        for (int i = 0; i < 5; ++i) {
            CHECK_NEAR(line.push(static_cast<double>(i)), -1.0, 0.0);
        }
        CHECK_NEAR(line.push(99.0), 0.0, 0.0);

        // Shrinking is where a stale head bites: the write index must come back
        // inside the new, shorter window or the first push lands in a dead slot.
        line.set_delay(2, 7.0);
        CHECK(line.delay_steps() == 2);
        CHECK_NEAR(line.push(10.0), 7.0, 0.0);
        CHECK_NEAR(line.push(11.0), 7.0, 0.0);
        CHECK_NEAR(line.push(12.0), 10.0, 0.0);
        CHECK_NEAR(line.push(13.0), 11.0, 0.0);
    }

    check::section("delay is clamped to the buffer");
    {
        // Domain randomisation feeds set_delay from an RNG, so an out-of-range draw
        // must clamp rather than index past the fixed-capacity array.
        core::DelayLine line(10 * core::kMaxDelaySteps, 0.0);
        CHECK(line.delay_steps() < core::kMaxDelaySteps);
        CHECK(line.delay_steps() > 0);

        line.set_delay(-5, 0.0);
        CHECK(line.delay_steps() == 0);
        CHECK_NEAR(line.push(3.0), 3.0, 0.0);

        line.set_delay(1000000, 0.0);
        CHECK(line.delay_steps() < core::kMaxDelaySteps);
        for (int i = 0; i < 4 * core::kMaxDelaySteps; ++i) {
            line.push(static_cast<double>(i));  // must not run off the end
        }
    }

    check::section("longest representable delay round-trips");
    {
        const int longest = core::kMaxDelaySteps - 1;
        core::DelayLine line(longest, 0.0);
        CHECK(line.delay_steps() == longest);
        for (int i = 0; i < longest; ++i) {
            CHECK_NEAR(line.push(static_cast<double>(i + 1)), 0.0, 0.0);
        }
        CHECK_NEAR(line.push(0.0), 1.0, 0.0);
    }

    return check::summary("delay_line");
}
