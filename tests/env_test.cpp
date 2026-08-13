// Env contract. Cheap checks, but they cover the things that silently poison a
// training run rather than crashing it: an observation that is the wrong size or
// scale, a reward whose sign is backwards, a reset that does not actually reset.

#include "sim/env.hpp"

#include "check.hpp"

#include <cmath>

namespace {

sim::EnvConfig base_config(int n_links = 1) {
    sim::EnvConfig c;
    c.plant = core::PendulumParams::uniform_rods(n_links, 0.12, 0.25, 0.0, 0.70);
    c.limits.max_accel = 30.0;
    c.limits.max_velocity = 3.0;
    c.limits.rail_length = 2.0;
    c.limits.steps_per_metre = 4000.0;
    c.episode_seconds = 2.0;
    c.theta_start_spread = 0.0;  // deterministic unless a test asks otherwise
    return c;
}

}  // namespace

int main() {
    check::section("observation layout and size");
    {
        for (int n = 1; n <= 3; ++n) {
            sim::Env env(base_config(n), 1);
            CHECK(env.observation_size() == 2 + 3 * n);
            const std::vector<double>& obs = env.reset();
            CHECK(static_cast<int>(obs.size()) == 2 + 3 * n);

            // Starts hanging: sin(pi) = 0, cos(pi) = -1.
            for (int i = 0; i < n; ++i) {
                CHECK_NEAR(obs[2 + 3 * i], 0.0, 1e-12);
                CHECK_NEAR(obs[3 + 3 * i], -1.0, 1e-12);
                CHECK_NEAR(obs[4 + 3 * i], 0.0, 1e-12);
            }
            CHECK_NEAR(obs[0], 0.0, 1e-12);
            CHECK_NEAR(obs[1], 0.0, 1e-12);
        }
    }

    check::section("sin/cos makes the pose single-valued across revolutions");
    {
        // The reason angles are not fed in raw. Swing-up runs theta through several
        // turns, so the same physical pose recurs at theta, theta +/- 2pi, ... Feeding
        // the angle directly would present those as wildly different inputs; sin/cos
        // collapses them to one, with no wrapping code anywhere downstream.
        for (const double base : {0.0, 0.7, core::kPi, -1.4}) {
            sim::EnvConfig c = base_config(1);
            c.theta_start = core::Vec::Constant(1, base);
            sim::Env reference(c, 1);
            const std::vector<double> want = reference.reset();

            for (const int turns : {-3, -1, 1, 2}) {
                c.theta_start = core::Vec::Constant(1, base + turns * core::kTwoPi);
                sim::Env spun(c, 1);
                const std::vector<double>& got = spun.reset();
                CHECK_NEAR(got[2], want[2], 1e-12);
                CHECK_NEAR(got[3], want[3], 1e-12);
            }
        }
    }

    check::section("observations stay bounded and finite under a violent command");
    {
        sim::EnvConfig c = base_config(1);
        sim::Env env(c, 7);
        env.reset();

        sim::StepResult r;
        for (int i = 0; i < 400 && !r.done && !r.truncated; ++i) {
            r = env.step((i / 12) % 2 == 0 ? 1.0 : -1.0);
            const std::vector<double>& obs = env.observation();
            for (const double v : obs) {
                CHECK(std::isfinite(v));
            }
            CHECK(std::abs(obs[2]) <= 1.0 + 1e-12);
            CHECK(std::abs(obs[3]) <= 1.0 + 1e-12);
            CHECK(std::abs(obs[0]) <= 1.0 + 1e-9);  // normalised by half the rail
        }
    }

    check::section("reward is maximal at the target and minimal opposite it");
    {
        sim::EnvConfig c = base_config(1);
        c.control_decimation = 1;
        c.position_weight = 0.0;
        c.action_weight = 0.0;

        // Upright start against an upright target: +1 per step.
        c.theta_start = core::Vec::Constant(1, 0.0);
        sim::Env upright(c, 1);
        upright.reset();
        CHECK_NEAR(upright.step(0.0).reward, 1.0, 1e-3);

        // Hanging against the same target: -1 per step.
        c.theta_start = core::Vec::Constant(1, core::kPi);
        sim::Env hanging(c, 1);
        hanging.reset();
        CHECK_NEAR(hanging.step(0.0).reward, -1.0, 1e-3);

        // And with the target moved to hanging, the same pose scores +1. This is the
        // whole of the per-link up/down goal: a reward parameter, nothing structural.
        c.theta_ref = core::Vec::Constant(1, core::kPi);
        sim::Env matched(c, 1);
        matched.reset();
        CHECK_NEAR(matched.step(0.0).reward, 1.0, 1e-3);
    }

    check::section("action is clamped and scaled to the acceleration limit");
    {
        sim::EnvConfig c = base_config(1);
        c.control_decimation = 1;
        sim::Env env(c, 3);
        env.reset();
        env.step(100.0);  // far outside [-1, 1]

        // One step at full command: velocity is a*dt and nothing larger.
        CHECK_NEAR(env.truth().xdot, c.limits.max_accel * c.dt, 1e-9);
    }

    check::section("running off the rail ends the episode and is penalised");
    {
        sim::EnvConfig c = base_config(1);
        c.terminate_on_step_loss = false;
        c.episode_seconds = 60.0;  // long enough that the rail is what stops it
        sim::Env env(c, 5);
        env.reset();

        sim::StepResult r;
        for (int i = 0; i < 100000 && !r.done && !r.truncated; ++i) {
            r = env.step(1.0);  // full acceleration, one direction, forever
        }
        CHECK(r.done);
        CHECK(r.off_rail);
        CHECK(r.reward < -c.rail_penalty / 2.0);
        CHECK(std::abs(env.truth().x) > 0.5 * c.limits.rail_length);
    }

    check::section("episodes truncate at the configured length");
    {
        sim::EnvConfig c = base_config(1);
        c.episode_seconds = 1.0;
        sim::Env env(c, 9);
        env.reset();

        sim::StepResult r;
        while (!r.done && !r.truncated) {
            r = env.step(0.0);  // no command: it just hangs there
        }
        CHECK(r.truncated);
        CHECK(!r.done);
        CHECK(env.plant_steps() == env.max_plant_steps());
        CHECK_NEAR(env.elapsed(), 1.0, 1e-9);
    }

    check::section("reset restores the start state and the rng is seeded");
    {
        sim::EnvConfig c = base_config(1);
        c.theta_start_spread = 0.1;

        sim::Env a(c, 12345);
        sim::Env b(c, 12345);
        const std::vector<double> first_a = a.reset();
        const std::vector<double> first_b = b.reset();
        for (std::size_t i = 0; i < first_a.size(); ++i) {
            CHECK_NEAR(first_a[i], first_b[i], 0.0);  // same seed, same episode
        }

        sim::StepResult r;
        for (int i = 0; i < 50 && !r.done && !r.truncated; ++i) {
            r = a.step(0.7);
        }
        CHECK(a.plant_steps() > 0);
        a.reset();
        CHECK(a.plant_steps() == 0);
        CHECK(std::abs(a.truth().xdot) < 1e-12);
    }

    check::section("the sensing chain can be switched on without changing the shape");
    {
        sim::EnvConfig c = base_config(2);
        c.use_sensor_model = true;
        c.encoder.noise_std = 0.001;
        c.encoder.latency_steps = 3;

        sim::Env env(c, 21);
        const std::vector<double>& obs = env.reset();
        CHECK(static_cast<int>(obs.size()) == env.observation_size());
        sim::StepResult r;
        for (int i = 0; i < 100 && !r.done && !r.truncated; ++i) {
            r = env.step((i % 2 == 0) ? 0.3 : -0.3);
        }
        for (const double v : env.observation()) {
            CHECK(std::isfinite(v));
        }
    }

    return check::summary("env");
}
