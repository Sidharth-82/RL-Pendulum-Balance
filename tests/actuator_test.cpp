// Actuator is the boundary between what the controller asks for and what a
// belt-driven stepper can actually deliver. Every check here is about that gap:
// the command is not the achieved acceleration, the dead-reckoned position is not
// the commanded position, and the available force is not constant with speed.

#include "core/actuator.hpp"
#include "core/plant.hpp"

#include "check.hpp"

#include <cmath>
#include <random>

namespace {

constexpr double kDt = 1.0 / 500.0;

core::StepperLimits default_limits() {
    core::StepperLimits limits;  // 0.5 m/s, 5 m/s^2, 10 N holding, knee 0.2 m/s, 1 m rail
    limits.steps_per_metre = 1000.0;
    return limits;
}

}  // namespace

int main() {
    const core::StepperLimits limits = default_limits();

    check::section("unsaturated command passes through");
    {
        core::Actuator actuator(limits);
        CHECK_NEAR(actuator.position(), 0.0, 0.0);
        CHECK_NEAR(actuator.velocity(), 0.0, 0.0);

        const double achieved = actuator.apply(2.0, kDt);
        CHECK_NEAR(achieved, 2.0, 1e-12);
        CHECK_NEAR(actuator.velocity(), 2.0 * kDt, 1e-15);
    }

    check::section("acceleration is clamped to the slew limit");
    {
        core::Actuator actuator(limits);
        CHECK_NEAR(actuator.apply(1.0e6, kDt), limits.max_accel, 1e-12);

        core::Actuator reverse(limits);
        CHECK_NEAR(reverse.apply(-1.0e6, kDt), -limits.max_accel, 1e-12);
    }

    check::section("velocity saturates and the achieved acceleration goes to zero");
    {
        core::Actuator actuator(limits);
        for (int i = 0; i < 200; ++i) {
            actuator.apply(limits.max_accel, kDt);
        }
        CHECK_NEAR(actuator.velocity(), limits.max_velocity, 1e-12);

        // Still commanding full acceleration, but there is nowhere left to go --
        // and this is the value the plant must be handed, not the command.
        CHECK_NEAR(actuator.apply(limits.max_accel, kDt), 0.0, 1e-12);
    }

    check::section("dead reckoning follows the clamped motion");
    {
        // Unsaturated: position is the exact sum of the post-update velocities,
        // a*dt^2 * k(k+1)/2.
        core::Actuator actuator(limits);
        constexpr int kCount = 20;
        constexpr double kAccel = 2.0;
        for (int i = 0; i < kCount; ++i) {
            actuator.apply(kAccel, kDt);
        }
        // Trapezoidal, so the belief matches Plant::step's own position update term
        // for term: x += xdot*dt + (1/2) a dt^2.
        double belief = 0.0;
        double velocity = 0.0;
        for (int i = 0; i < kCount; ++i) {
            belief += velocity * kDt + 0.5 * kAccel * kDt * kDt;
            velocity += kAccel * kDt;
        }
        CHECK_NEAR(actuator.position(), belief, 1e-12);

        // Saturated: the belief must advance at the clamped velocity, not at the
        // velocity the command implies. This is what makes step loss the only
        // source of divergence between belief and truth.
        core::Actuator saturated(limits);
        for (int i = 0; i < 200; ++i) {
            saturated.apply(limits.max_accel, kDt);
        }
        const double before = saturated.position();
        saturated.apply(limits.max_accel, kDt);
        CHECK_NEAR(saturated.position() - before, limits.max_velocity * kDt, 1e-12);
    }

    check::section("belief tracks plant truth exactly while no steps are lost");
    {
        // The invariant the class is built on: dead reckoning diverges from reality
        // only through step loss. That requires the actuator and the plant to
        // integrate position identically -- two different integrators would drift
        // apart on their own, and the drift would look exactly like step loss.
        core::Actuator actuator(limits);
        core::Plant plant(core::PendulumParams::uniform_rods(1, 0.1, 0.3, 0.0, 0.5));
        core::State truth;
        truth.resize(1);
        truth.theta[0] = 0.05;

        std::mt19937 rng(4242);
        std::uniform_real_distribution<double> command(-limits.max_accel, limits.max_accel);

        for (int i = 0; i < 2000; ++i) {
            const double achieved = actuator.apply(command(rng), kDt);
            plant.step(truth, achieved, kDt);
            CHECK_NEAR(actuator.position(), truth.x, 1e-12);
            CHECK_NEAR(actuator.velocity(), truth.xdot, 1e-12);
        }
    }

    check::section("command latency delays the whole envelope");
    {
        constexpr int kLatency = 3;
        core::Actuator actuator(limits, kLatency);
        for (int i = 0; i < kLatency; ++i) {
            CHECK_NEAR(actuator.apply(4.0, kDt), 0.0, 1e-15);
            CHECK_NEAR(actuator.velocity(), 0.0, 1e-15);
        }
        CHECK_NEAR(actuator.apply(4.0, kDt), 4.0, 1e-12);
    }

    check::section("reset restores pose and clears the pending command");
    {
        core::Actuator actuator(limits, 2);
        actuator.apply(5.0, kDt);
        actuator.apply(5.0, kDt);
        actuator.reset(0.1, -0.2);
        CHECK_NEAR(actuator.position(), 0.1, 0.0);
        CHECK_NEAR(actuator.velocity(), -0.2, 0.0);
        // The commands queued before the reset must not arrive after it.
        CHECK_NEAR(actuator.apply(0.0, kDt), 0.0, 1e-15);
        CHECK_NEAR(actuator.apply(0.0, kDt), 0.0, 1e-15);
    }

    check::section("off_rail measures from the centre of the rail");
    {
        core::Actuator actuator(limits);
        const double half = 0.5 * limits.rail_length;

        actuator.reset(0.0, 0.0);
        CHECK(!actuator.off_rail());  // the origin is the middle of the rail
        actuator.reset(half * 0.98, 0.0);
        CHECK(!actuator.off_rail());
        actuator.reset(-half * 0.98, 0.0);
        CHECK(!actuator.off_rail());
        actuator.reset(half * 1.02, 0.0);
        CHECK(actuator.off_rail());
        actuator.reset(-half * 1.02, 0.0);
        CHECK(actuator.off_rail());
    }

    check::section("quantise returns metres, not step counts");
    {
        core::Actuator actuator(limits);
        const double step = 1.0 / limits.steps_per_metre;

        CHECK_NEAR(actuator.quantise(0.12345), 0.123, 1e-12);
        CHECK_NEAR(actuator.quantise(-0.12345), -0.123, 1e-12);
        CHECK_NEAR(actuator.quantise(0.0), 0.0, 0.0);

        // Never further than half a step from the input, and stable under a second
        // application -- both fail loudly if the divide back to metres is missing.
        for (int i = -50; i <= 50; ++i) {
            const double v = i * 0.00713;
            const double q = actuator.quantise(v);
            CHECK(std::abs(q - v) <= 0.5 * step + 1e-15);
            CHECK_NEAR(actuator.quantise(q), q, 1e-12);
        }

        // Unconfigured resolution is the struct default and must not divide by zero.
        core::StepperLimits unconfigured = limits;
        unconfigured.steps_per_metre = 0.0;
        core::Actuator continuous(unconfigured);
        CHECK_NEAR(continuous.quantise(0.12345), 0.12345, 0.0);
    }

    check::section("available force droops with speed");
    {
        core::Actuator actuator(limits);

        actuator.reset(0.0, 0.0);
        CHECK_NEAR(actuator.available_force(), limits.holding_force, 1e-12);

        // Flat out to the knee.
        actuator.reset(0.0, limits.force_knee_speed * 0.5);
        CHECK_NEAR(actuator.available_force(), limits.holding_force, 1e-12);
        actuator.reset(0.0, limits.force_knee_speed);
        CHECK_NEAR(actuator.available_force(), limits.holding_force, 1e-12);

        // Linear from the knee to zero at top speed.
        const double mid = 0.5 * (limits.force_knee_speed + limits.max_velocity);
        actuator.reset(0.0, mid);
        CHECK_NEAR(actuator.available_force(), 0.5 * limits.holding_force, 1e-12);
        actuator.reset(0.0, limits.max_velocity);
        CHECK_NEAR(actuator.available_force(), 0.0, 1e-12);

        // Symmetric in direction, and never increasing with speed.
        actuator.reset(0.0, -mid);
        CHECK_NEAR(actuator.available_force(), 0.5 * limits.holding_force, 1e-12);

        double previous = limits.holding_force + 1.0;
        for (int i = 0; i <= 100; ++i) {
            actuator.reset(0.0, limits.max_velocity * i / 100.0);
            const double f = actuator.available_force();
            CHECK(f <= previous + 1e-12);
            CHECK(f >= -1e-12);
            previous = f;
        }
    }

    check::section("degenerate limits stay finite");
    {
        // Domain randomisation could draw a knee at or past top speed; the linear
        // interpolation's denominator vanishes there.
        core::StepperLimits degenerate = limits;
        degenerate.force_knee_speed = limits.max_velocity;
        core::Actuator actuator(degenerate);

        actuator.reset(0.0, 0.5 * limits.max_velocity);
        CHECK(std::isfinite(actuator.available_force()));
        actuator.reset(0.0, limits.max_velocity * 1.5);
        CHECK(std::isfinite(actuator.available_force()));
        CHECK(actuator.available_force() >= 0.0);
    }

    check::section("step loss triggers on the force the motor cannot hold");
    {
        core::Actuator actuator(limits);
        actuator.reset(0.0, 0.0);
        const double available = actuator.available_force();
        CHECK(!actuator.would_lose_steps(available * 0.99));
        CHECK(!actuator.would_lose_steps(-available * 0.99));
        CHECK(actuator.would_lose_steps(available * 1.01));
        CHECK(actuator.would_lose_steps(-available * 1.01));

        // The same force that is safe at rest desynchronises the motor near top
        // speed, which is the entire point of the droop model.
        actuator.reset(0.0, 0.9 * limits.max_velocity);
        CHECK(actuator.would_lose_steps(available * 0.99));
    }

    check::section("set_limits takes effect");
    {
        core::StepperLimits tight = limits;
        tight.max_accel = 1.0;

        core::Actuator actuator(limits);
        actuator.set_limits(tight);
        actuator.reset(0.0, 0.0);
        CHECK_NEAR(actuator.apply(100.0, kDt), tight.max_accel, 1e-12);
    }

    return check::summary("actuator");
}
