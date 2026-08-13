// The one class compiled into both the simulator and the Pi binary. Anything that
// makes the two behave differently -- a gain read from the wrong slot, an integral
// that scales with the wrong dt -- breaks transfer without breaking either build.

#include "core/controller.hpp"

#include "check.hpp"

namespace {

constexpr double kDt = 1.0 / 500.0;

core::Gains make_gains(int n) {
    core::Gains g;
    g.kp_x = 2.0;
    g.kd_x = 0.5;
    g.ki_x = 0.25;
    g.kp_theta.resize(n);
    g.kd_theta.resize(n);
    for (int i = 0; i < n; ++i) {
        g.kp_theta[i] = 10.0 * (i + 1);
        g.kd_theta[i] = 1.0 * (i + 1);
    }
    return g;
}

}  // namespace

int main() {
    check::section("gain vector size is 2n + 3");
    {
        CHECK(core::Gains::size(1) == 5);
        CHECK(core::Gains::size(3) == 9);
        CHECK(make_gains(3).n_links() == 3);
    }

    check::section("to_flat packs [kp_x, kd_x, ki_x, kp_theta, kd_theta]");
    {
        const core::Gains g = make_gains(3);
        const Eigen::VectorXd v = g.to_flat();

        CHECK(v.size() == 9);
        CHECK_NEAR(v(0), 2.0, 0.0);
        CHECK_NEAR(v(1), 0.5, 0.0);
        CHECK_NEAR(v(2), 0.25, 0.0);
        CHECK_NEAR(v(3), 10.0, 0.0);
        CHECK_NEAR(v(4), 20.0, 0.0);
        CHECK_NEAR(v(5), 30.0, 0.0);
        CHECK_NEAR(v(6), 1.0, 0.0);
        CHECK_NEAR(v(7), 2.0, 0.0);
        CHECK_NEAR(v(8), 3.0, 0.0);
    }

    check::section("from_flat inverts to_flat");
    {
        for (int n = 1; n <= core::kMaxLinks; ++n) {
            const core::Gains original = make_gains(n);
            const core::Gains restored = core::Gains::from_flat(original.to_flat(), n);

            CHECK(restored.n_links() == n);
            CHECK_NEAR(restored.kp_x, original.kp_x, 0.0);
            CHECK_NEAR(restored.kd_x, original.kd_x, 0.0);
            CHECK_NEAR(restored.ki_x, original.ki_x, 0.0);
            for (int i = 0; i < n; ++i) {
                CHECK_NEAR(restored.kp_theta[i], original.kp_theta[i], 0.0);
                CHECK_NEAR(restored.kd_theta[i], original.kd_theta[i], 0.0);
            }
        }
    }

    check::section("control law matches the written form");
    {
        core::Gains g = make_gains(1);
        g.ki_x = 0.0;  // isolate the proportional and derivative paths
        g.kp_theta[0] = 30.0;
        g.kd_theta[0] = 3.0;

        core::Controller controller(g, kDt, 1.0);

        core::Vec theta(1);
        core::Vec omega(1);
        theta[0] = 0.05;
        omega[0] = 0.30;

        // -(2*0.1 + 0.5*0.2 + 30*0.05 + 3*0.3)
        CHECK_NEAR(controller.compute(0.1, 0.2, theta, omega), -2.7, 1e-12);
    }

    check::section("every link contributes its own pair of gains");
    {
        core::Gains g = make_gains(3);
        g.ki_x = 0.0;
        core::Controller controller(g, kDt, 1.0);

        core::Vec theta(3);
        core::Vec omega(3);
        theta << 0.01, 0.02, 0.03;
        omega << 0.10, 0.20, 0.30;

        // link terms: 10*.01 + 20*.02 + 30*.03 + 1*.1 + 2*.2 + 3*.3 = 2.8
        // carriage:   2*0.1 + 0.5*0.2 = 0.3
        CHECK_NEAR(controller.compute(0.1, 0.2, theta, omega), -3.1, 1e-12);

        // Asymmetric angles: swapping two links must change the command, which a
        // gain vector read in the wrong order would not.
        core::Vec swapped(3);
        swapped << 0.03, 0.02, 0.01;
        CHECK(std::abs(controller.compute(0.0, 0.0, swapped, omega) -
                       controller.compute(0.0, 0.0, theta, omega)) > 1e-6);
    }

    check::section("command opposes the error");
    {
        core::Gains g = make_gains(1);
        g.ki_x = 0.0;
        core::Controller controller(g, kDt, 1.0);

        core::Vec zero(1);
        zero.setZero();
        CHECK(controller.compute(0.5, 0.0, zero, zero) < 0.0);
        CHECK(controller.compute(-0.5, 0.0, zero, zero) > 0.0);

        core::Vec tilt(1);
        core::Vec still(1);
        tilt[0] = 0.2;
        still.setZero();
        CHECK(controller.compute(0.0, 0.0, tilt, still) < 0.0);
    }

    check::section("integral accumulates at the configured dt");
    {
        core::Controller controller(make_gains(1), kDt, 10.0);
        core::Vec zero(1);
        zero.setZero();

        CHECK_NEAR(controller.integral(), 0.0, 0.0);
        for (int i = 0; i < 100; ++i) {
            controller.compute(1.0, 0.0, zero, zero);
        }
        // A constructor that drops dt leaves this at zero and quietly disables the
        // integral term for every value of ki_x.
        CHECK_NEAR(controller.integral(), 100 * kDt, 1e-12);

        // And it must unwind when the error changes sign.
        for (int i = 0; i < 50; ++i) {
            controller.compute(-1.0, 0.0, zero, zero);
        }
        CHECK_NEAR(controller.integral(), 50 * kDt, 1e-12);
    }

    check::section("anti-windup clamps the accumulator");
    {
        constexpr double kLimit = 0.05;
        core::Controller controller(make_gains(1), kDt, kLimit);
        core::Vec zero(1);
        zero.setZero();

        for (int i = 0; i < 1000; ++i) {
            controller.compute(1.0, 0.0, zero, zero);
        }
        // Unclamped this would reach 2.0. A constructor that drops the limit clamps
        // to zero instead, which is just as wrong in the other direction.
        CHECK_NEAR(controller.integral(), kLimit, 1e-15);

        for (int i = 0; i < 2000; ++i) {
            controller.compute(-1.0, 0.0, zero, zero);
        }
        CHECK_NEAR(controller.integral(), -kLimit, 1e-15);
    }

    check::section("integral feeds the command with the right weight");
    {
        core::Gains g = make_gains(1);
        g.kp_x = 0.0;
        g.kd_x = 0.0;
        g.ki_x = 4.0;
        g.kp_theta[0] = 0.0;
        g.kd_theta[0] = 0.0;
        core::Controller controller(g, kDt, 10.0);

        core::Vec zero(1);
        zero.setZero();
        const double command = controller.compute(1.0, 0.0, zero, zero);
        // One step of accumulation, then -ki_x * integral.
        CHECK_NEAR(controller.integral(), kDt, 1e-15);
        CHECK_NEAR(command, -4.0 * kDt, 1e-15);
    }

    check::section("augmented_state and to_flat agree with the control law");
    {
        // The layout lock. Gains::to_flat and augmented_state are two independent
        // orderings that must stay in step, and the failure when they drift is
        // silent: every link gain lands one slot off, which still produces a
        // stable-looking controller for the wrong plant. Evaluating the control
        // law both ways -- through compute(), and as a plain dot product -- makes
        // any future reordering of either one fail here immediately.
        //
        // This is also the contract the LQR baseline relies on: it designs a gain
        // vector over exactly this state ordering, so the Riccati result maps onto
        // Gains as the identity.
        for (int n = 1; n <= core::kMaxLinks; ++n) {
            const core::Gains g = make_gains(n);
            core::Controller controller(g, kDt, 10.0);

            core::State state;
            state.resize(n);
            state.x = 0.13;
            state.xdot = -0.27;
            for (int i = 0; i < n; ++i) {
                state.theta[i] = 0.05 * (i + 1);
                state.omega[i] = -0.11 * (i + 1);
            }

            for (int repeat = 0; repeat < 4; ++repeat) {
                const double command =
                    controller.compute(state.x, state.xdot, state.theta, state.omega);
                // compute() accumulates the integral before using it, so the packed
                // state must read the integral back afterwards to line up.
                const Eigen::VectorXd z = core::augmented_state(state, controller.integral());
                CHECK(z.size() == core::Gains::size(n));
                CHECK_NEAR(command, -g.to_flat().dot(z), 1e-12);
            }
        }
    }

    check::section("theta_ref moves the setpoint, and the error wraps");
    {
        core::Gains g = make_gains(1);
        g.kp_x = 0.0;
        g.kd_x = 0.0;
        g.ki_x = 0.0;
        g.kp_theta[0] = 30.0;
        g.kd_theta[0] = 0.0;

        core::Controller controller(g, kDt, 10.0);
        core::Vec still(1);
        still.setZero();

        // Default is all-upright.
        CHECK(controller.theta_ref().size() == 1);
        CHECK_NEAR(controller.theta_ref()[0], 0.0, 0.0);

        core::Vec hanging(1);
        hanging[0] = core::kPi;
        controller.set_theta_ref(hanging);

        // At the setpoint the tilt error is zero, so the command is zero -- even
        // though the raw angle is pi.
        core::Vec at_ref(1);
        at_ref[0] = core::kPi;
        CHECK_NEAR(controller.compute(0.0, 0.0, at_ref, still), 0.0, 1e-12);

        // The same equilibrium approached from the other side. Unwrapped, this would
        // read as an error of -2pi and command a huge acceleration; wrapped, it is
        // the same point.
        core::Vec other_side(1);
        other_side[0] = -core::kPi;
        CHECK_NEAR(controller.compute(0.0, 0.0, other_side, still), 0.0, 1e-12);

        // And after several revolutions, which is what swing-up actually produces.
        core::Vec spun(1);
        spun[0] = core::kPi + 4.0 * core::kPi;
        CHECK_NEAR(controller.compute(0.0, 0.0, spun, still), 0.0, 1e-12);

        // A small deviation either side gives an opposing command of the right size.
        core::Vec tilted(1);
        tilted[0] = core::kPi + 0.01;
        CHECK_NEAR(controller.compute(0.0, 0.0, tilted, still), -0.3, 1e-12);
        tilted[0] = core::kPi - 0.01;
        CHECK_NEAR(controller.compute(0.0, 0.0, tilted, still), 0.3, 1e-12);

        // augmented_state has to agree, or the LQR layout lock breaks the moment a
        // non-upright equilibrium is targeted.
        core::State state;
        state.resize(1);
        state.theta[0] = core::kPi + 0.01;
        const Eigen::VectorXd z = core::augmented_state(state, 0.0, hanging);
        CHECK_NEAR(z(3), 0.01, 1e-12);
    }

    check::section("wrap_angle folds into (-pi, pi]");
    {
        CHECK_NEAR(core::wrap_angle(0.0), 0.0, 0.0);
        CHECK_NEAR(core::wrap_angle(0.5), 0.5, 1e-15);
        CHECK_NEAR(core::wrap_angle(core::kPi), core::kPi, 1e-12);
        CHECK_NEAR(core::wrap_angle(-core::kPi), core::kPi, 1e-12);
        CHECK_NEAR(core::wrap_angle(core::kTwoPi), 0.0, 1e-12);
        CHECK_NEAR(core::wrap_angle(1.5 * core::kPi), -0.5 * core::kPi, 1e-12);
        CHECK_NEAR(core::wrap_angle(-1.5 * core::kPi), 0.5 * core::kPi, 1e-12);
        CHECK_NEAR(core::wrap_angle(10.0 * core::kTwoPi + 0.25), 0.25, 1e-10);
    }

    check::section("reset clears the integral, set_gains does not");
    {
        core::Controller controller(make_gains(1), kDt, 10.0);
        core::Vec zero(1);
        zero.setZero();

        for (int i = 0; i < 100; ++i) {
            controller.compute(1.0, 0.0, zero, zero);
        }
        CHECK(controller.integral() > 0.0);

        core::Gains other = make_gains(1);
        other.kp_x = 99.0;
        controller.set_gains(other);
        CHECK_NEAR(controller.gains().kp_x, 99.0, 0.0);
        // Mid-episode gain changes must not discard accumulated position error.
        CHECK(controller.integral() > 0.0);

        controller.reset();
        CHECK_NEAR(controller.integral(), 0.0, 0.0);
    }

    return check::summary("controller");
}
