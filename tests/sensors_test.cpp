// SensorModel is where the sim-to-real gap lives, so it is the component whose
// bugs are least visible: every failure mode here still produces a plausible
// trajectory and a controller that trains happily against a plant it will never
// meet. The checks are correspondingly literal -- exact sample offsets, exact
// filter coefficients, exact pass-through.

#include "core/sensors.hpp"

#include "check.hpp"

#include <cmath>
#include <numeric>
#include <random>
#include <vector>

namespace {

constexpr double kDt = 1.0 / 500.0;
constexpr double kTwoPi = 6.283185307179586476925286766559;

core::EncoderSpec fine_spec() {
    core::EncoderSpec spec;
    spec.bits = 16;
    spec.noise_std = 0.0;
    spec.latency_steps = 0;
    spec.derivative_cutoff_hz = 50.0;
    return spec;
}

double count_size(const core::EncoderSpec& spec) {
    return kTwoPi / std::exp2(static_cast<double>(spec.bits));
}

core::State single_link(double theta, double omega = 0.0) {
    core::State s;
    s.resize(1);
    s.theta[0] = theta;
    s.omega[0] = omega;
    return s;
}

}  // namespace

int main() {
    std::mt19937 rng(12345);

    check::section("angles come back in radians, near truth");
    {
        core::EncoderSpec spec = fine_spec();
        core::SensorModel sensor(spec, 1, kDt);

        const core::State truth = single_link(0.3);
        sensor.reset(truth);
        const core::Measurement m = sensor.measure(truth, 0.25, -0.5, rng);

        // Within half a count of the true angle. A quantiser that returns counts
        // rather than radians, or divides by 360 rather than 2*pi, misses by
        // orders of magnitude rather than by a rounding step.
        CHECK(std::abs(m.theta[0] - 0.3) <= 0.5 * count_size(spec) + 1e-15);

        // Carriage state is the actuator's belief and must pass through untouched.
        CHECK_NEAR(m.x, 0.25, 0.0);
        CHECK_NEAR(m.xdot, -0.5, 0.0);
    }

    check::section("quantisation lands on a whole count");
    {
        core::EncoderSpec spec = fine_spec();
        spec.bits = 12;  // AS5600
        core::SensorModel sensor(spec, 1, kDt);
        const double step = count_size(spec);

        for (int i = -20; i <= 20; ++i) {
            const core::State truth = single_link(i * 0.0731);
            sensor.reset(truth);
            const core::Measurement m = sensor.measure(truth, 0.0, 0.0, rng);
            const double counts = m.theta[0] / step;
            CHECK_NEAR(counts, std::round(counts), 1e-9);
            CHECK(std::abs(m.theta[0] - i * 0.0731) <= 0.5 * step + 1e-12);
        }
    }

    check::section("reset removes the start-of-episode transient");
    {
        // Without seeding, previous_theta_ and the delay line both start at zero
        // while truth does not, and the first differentiation turns that step into
        // an enormous fictitious rate -- arriving exactly when the controller is
        // least able to absorb it.
        core::SensorModel sensor(fine_spec(), 1, kDt);
        const core::State truth = single_link(0.5);
        sensor.reset(truth);

        for (int i = 0; i < 5; ++i) {
            const core::Measurement m = sensor.measure(truth, 0.0, 0.0, rng);
            CHECK_NEAR(m.omega[0], 0.0, 1e-12);
            CHECK(std::abs(m.theta[0] - 0.5) <= 0.5 * count_size(fine_spec()) + 1e-15);
        }
    }

    check::section("filter coefficient matches dt / (dt + 1/(2 pi fc))");
    {
        core::EncoderSpec spec = fine_spec();
        core::SensorModel sensor(spec, 1, kDt);

        sensor.reset(single_link(0.0));
        const core::Measurement first = sensor.measure(single_link(0.01), 0.0, 0.0, rng);

        // One step of a first-order low-pass from zero state is exactly alpha times
        // the raw backward difference. The tolerance is tight enough to reject a
        // four-digit stand-in for pi, which shifts alpha in the fifth figure.
        const double alpha = kDt / (kDt + 1.0 / (kTwoPi * spec.derivative_cutoff_hz));
        const double raw = first.theta[0] / kDt;
        CHECK_NEAR(first.omega[0], alpha * raw, 1e-9);

        // Held at the same angle the raw rate is zero, so the state decays by
        // (1 - alpha) per step.
        const core::Measurement second = sensor.measure(single_link(0.01), 0.0, 0.0, rng);
        CHECK_NEAR(second.omega[0], (1.0 - alpha) * first.omega[0], 1e-12);
    }

    check::section("a non-positive cutoff means unfiltered, not divide-by-zero");
    {
        core::EncoderSpec spec = fine_spec();
        spec.derivative_cutoff_hz = 0.0;
        core::SensorModel sensor(spec, 1, kDt);

        sensor.reset(single_link(0.0));
        const core::Measurement m = sensor.measure(single_link(0.01), 0.0, 0.0, rng);
        CHECK(std::isfinite(m.omega[0]));
        CHECK_NEAR(m.omega[0], m.theta[0] / kDt, 1e-12);  // alpha == 1
    }

    check::section("rate estimate converges on a constant true rate");
    {
        constexpr double kRate = 1.0;  // rad/s
        core::SensorModel sensor(fine_spec(), 1, kDt);
        sensor.reset(single_link(0.0));

        std::vector<double> tail;
        for (int i = 1; i <= 300; ++i) {
            const core::Measurement m = sensor.measure(single_link(kRate * i * kDt), 0.0, 0.0, rng);
            if (i > 200) {
                tail.push_back(m.omega[0]);
                CHECK(std::abs(m.omega[0] - kRate) < 0.15);  // per-sample ripple
            }
        }
        const double mean = std::accumulate(tail.begin(), tail.end(), 0.0) / tail.size();
        CHECK_NEAR(mean, kRate, 0.02);
    }

    check::section("latency shifts the reading by exactly N samples");
    {
        constexpr int kLatency = 4;
        core::EncoderSpec prompt = fine_spec();
        core::EncoderSpec laggy = fine_spec();
        laggy.latency_steps = kLatency;

        core::SensorModel fast(prompt, 1, kDt);
        core::SensorModel slow(laggy, 1, kDt);

        const core::State start = single_link(0.0);
        fast.reset(start);
        slow.reset(start);

        std::vector<double> fast_theta;
        std::vector<double> slow_theta;
        for (int i = 0; i < 40; ++i) {
            const core::State truth = single_link(0.003 * i + 0.0004 * i * i);
            fast_theta.push_back(fast.measure(truth, 0.0, 0.0, rng).theta[0]);
            slow_theta.push_back(slow.measure(truth, 0.0, 0.0, rng).theta[0]);
        }
        for (int i = kLatency; i < 40; ++i) {
            CHECK_NEAR(slow_theta[i], fast_theta[i - kLatency], 0.0);
        }
        // And it must actually be delayed, not merely equal on a slow ramp.
        CHECK(std::abs(slow_theta[39] - fast_theta[39]) > 1e-6);
    }

    check::section("set_spec re-lengths the delay lines");
    {
        // The randomisation hook. Iterating the delay-line array by value here
        // updates copies, leaving the sensing lag pinned at whatever the
        // constructor set -- which silently removes latency from the randomisation
        // that latency robustness is supposed to come from.
        constexpr int kLatency = 4;
        core::EncoderSpec laggy = fine_spec();
        laggy.latency_steps = kLatency;

        core::SensorModel fast(fine_spec(), 1, kDt);
        core::SensorModel changed(fine_spec(), 1, kDt);
        changed.set_spec(laggy);

        const core::State start = single_link(0.0);
        fast.reset(start);
        changed.reset(start);

        std::vector<double> fast_theta;
        std::vector<double> changed_theta;
        for (int i = 0; i < 30; ++i) {
            const core::State truth = single_link(0.005 * i);
            fast_theta.push_back(fast.measure(truth, 0.0, 0.0, rng).theta[0]);
            changed_theta.push_back(changed.measure(truth, 0.0, 0.0, rng).theta[0]);
        }
        for (int i = kLatency; i < 30; ++i) {
            CHECK_NEAR(changed_theta[i], fast_theta[i - kLatency], 0.0);
        }
        CHECK(std::abs(changed_theta[29] - fast_theta[29]) > 1e-6);
    }

    check::section("links are measured independently and in order");
    {
        core::SensorModel sensor(fine_spec(), 3, kDt);

        core::State truth;
        truth.resize(3);
        truth.theta[0] = 0.20;
        truth.theta[1] = -0.70;
        truth.theta[2] = 1.30;
        sensor.reset(truth);

        const core::Measurement m = sensor.measure(truth, 0.0, 0.0, rng);
        CHECK(m.theta.size() == 3);
        CHECK(m.omega.size() == 3);
        const double half = 0.5 * count_size(fine_spec()) + 1e-15;
        CHECK(std::abs(m.theta[0] - 0.20) <= half);
        CHECK(std::abs(m.theta[1] + 0.70) <= half);
        CHECK(std::abs(m.theta[2] - 1.30) <= half);
    }

    check::section("noise is applied, unbiased, and reproducible from a seed");
    {
        core::EncoderSpec noisy = fine_spec();
        noisy.noise_std = 0.01;

        core::SensorModel sensor(noisy, 1, kDt);
        const core::State truth = single_link(0.4);

        std::mt19937 rng_a(777);
        sensor.reset(truth);
        std::vector<double> first_pass;
        double sum = 0.0;
        for (int i = 0; i < 4000; ++i) {
            const double v = sensor.measure(truth, 0.0, 0.0, rng_a).theta[0];
            first_pass.push_back(v);
            sum += v;
        }
        CHECK(std::abs(sum / first_pass.size() - 0.4) < 0.002);  // ~3 standard errors

        double spread = 0.0;
        for (const double v : first_pass) {
            spread = std::max(spread, std::abs(v - 0.4));
        }
        CHECK(spread > noisy.noise_std);  // noise really is being added

        std::mt19937 rng_b(777);
        core::SensorModel replay(noisy, 1, kDt);
        replay.reset(truth);
        for (int i = 0; i < 4000; ++i) {
            CHECK_NEAR(replay.measure(truth, 0.0, 0.0, rng_b).theta[0], first_pass[i], 0.0);
        }
    }

    return check::summary("sensors");
}
