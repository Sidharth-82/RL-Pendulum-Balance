// LQR baseline, checked one function at a time.
//
// Written ahead of the implementation on purpose: every section runs inside a
// try/catch, so a function that still throws logic_error reports PEND and the rest
// of the run continues. Run this after each function you finish and the pending
// count goes down. Nothing here fails because something is unimplemented -- only
// because something is implemented and wrong.
//
// Order matches the dependency chain, and so does the difficulty: the first two
// sections check arithmetic you can verify in your head, and the linearize section
// checks against a closed form derived independently below.

#include "baseline/lqr.hpp"

#include "check.hpp"

#include <cmath>
#include <stdexcept>

namespace {

constexpr double kDt = 1.0 / 500.0;
constexpr double kMass = 0.1;          // kg
constexpr double kLength = 0.3;        // m
constexpr double kCarriageMass = 0.5;  // kg

// Augmented index of each named state, for readability below. The ordering is
// [x, xdot, integral_x, theta_0..n-1, omega_0..n-1] -- core::augmented_state.
constexpr int kX = 0;
constexpr int kXdot = 1;
constexpr int kIntegral = 2;
constexpr int kTheta0 = 3;

core::Plant uniform_plant(int n) {
    return core::Plant(
        core::PendulumParams::uniform_rods(n, kMass, kLength, 0.0, kCarriageMass));
}

// Runs a section, treating "not implemented" as pending rather than failure.
template <typename Body>
void section(const std::string& name, Body&& body) {
    check::section(name);
    try {
        body();
    } catch (const std::logic_error& e) {
        check::skip(e.what());
    }
}

}  // namespace

int main() {
    // ---- LqrTolerances ------------------------------------------------------
    section("state_penalty is 1/tol^2 in the augmented ordering", [] {
        baseline::LqrTolerances tol;  // struct defaults
        for (int n = 1; n <= 3; ++n) {
            const Eigen::VectorXd q = tol.state_penalty(n);
            CHECK(q.size() == core::Gains::size(n));

            CHECK_NEAR(q(kX), 1.0 / (tol.x * tol.x), 1e-12);
            CHECK_NEAR(q(kXdot), 1.0 / (tol.xdot * tol.xdot), 1e-12);
            CHECK_NEAR(q(kIntegral), 1.0 / (tol.integral_x * tol.integral_x), 1e-12);
            for (int i = 0; i < n; ++i) {
                CHECK_NEAR(q(kTheta0 + i), 1.0 / (tol.theta * tol.theta), 1e-12);
                CHECK_NEAR(q(kTheta0 + n + i), 1.0 / (tol.omega * tol.omega), 1e-12);
            }
            CHECK((q.array() > 0.0).all());
        }
    });

    section("input_penalty is 1/accel^2", [] {
        baseline::LqrTolerances tol;
        CHECK_NEAR(tol.input_penalty(), 1.0 / (tol.accel * tol.accel), 1e-12);
    });

    section("from_envelope reads the hardware limits", [] {
        core::StepperLimits limits;
        limits.max_velocity = 0.42;
        limits.max_accel = 6.5;
        limits.rail_length = 1.2;

        const baseline::LqrTolerances tol =
            baseline::LqrTolerances::from_envelope(limits, 0.08, 0.9);

        CHECK_NEAR(tol.xdot, limits.max_velocity, 1e-12);
        CHECK_NEAR(tol.theta, 0.08, 1e-12);
        CHECK_NEAR(tol.omega, 0.9, 1e-12);

        // Both of these carry headroom rather than sitting on the limit, and for the
        // same reason: the rail end and the step-rate slew limit are terminations,
        // not soft bounds. A tolerance equal to the limit prices a hard failure at a
        // cost of 1, on par with a few degrees of tilt.
        CHECK(tol.x > 0.0);
        CHECK(tol.x < 0.5 * limits.rail_length);
        CHECK(tol.accel > 0.0);
        CHECK(tol.accel < limits.max_accel);
        CHECK(tol.integral_x > 0.0);
    });

    // ---- linearize ----------------------------------------------------------
    section("linearize: shape and the exactly-known carriage block", [] {
        // These entries come straight out of Plant::step's own arithmetic:
        //   x'    = x + xdot*dt + (1/2) a dt^2
        //   xdot' = xdot + a dt
        // so they are exact, not approximations, and the tolerance is tight.
        for (int n = 1; n <= 3; ++n) {
            const core::Plant plant = uniform_plant(n);
            const baseline::LinearModel model = baseline::linearize(plant, kDt);
            const int m = core::Gains::size(n);

            CHECK(model.A.rows() == m);
            CHECK(model.A.cols() == m);
            CHECK(model.B.size() == m);
            CHECK(model.n_links == n);
            CHECK_NEAR(model.dt, kDt, 0.0);

            CHECK_NEAR(model.A(kX, kX), 1.0, 1e-9);
            CHECK_NEAR(model.A(kX, kXdot), kDt, 1e-9);
            CHECK_NEAR(model.A(kXdot, kX), 0.0, 1e-9);
            CHECK_NEAR(model.A(kXdot, kXdot), 1.0, 1e-9);
            CHECK_NEAR(model.B(kX), 0.5 * kDt * kDt, 1e-12);
            CHECK_NEAR(model.B(kXdot), kDt, 1e-9);
        }
    });

    section("linearize: integrator row and column", [] {
        for (int n = 1; n <= 3; ++n) {
            const baseline::LinearModel model = baseline::linearize(uniform_plant(n), kDt);
            const int m = core::Gains::size(n);

            // integral_x[k+1] = integral_x[k] + dt * x[k], using x at the START of
            // the step so the model matches Controller::compute's accumulate order.
            CHECK_NEAR(model.A(kIntegral, kX), kDt, 1e-12);
            CHECK_NEAR(model.A(kIntegral, kIntegral), 1.0, 1e-12);
            CHECK_NEAR(model.A(kIntegral, kXdot), 0.0, 1e-12);
            CHECK_NEAR(model.B(kIntegral), 0.0, 1e-12);

            // Nothing in the physics depends on the integral, so its column is zero
            // apart from the integrator's own memory.
            for (int i = 0; i < m; ++i) {
                if (i != kIntegral) {
                    CHECK_NEAR(model.A(i, kIntegral), 0.0, 1e-12);
                }
            }
        }
    });

    section("linearize: carriage rows do not depend on the links", [] {
        // The links exert no influence on carriage motion -- the carriage is a
        // prescribed-motion boundary, driven only by the command. If these are
        // non-zero the augmented index map is wrong.
        for (int n = 1; n <= 3; ++n) {
            const baseline::LinearModel model = baseline::linearize(uniform_plant(n), kDt);
            for (int j = kTheta0; j < core::Gains::size(n); ++j) {
                CHECK_NEAR(model.A(kX, j), 0.0, 1e-9);
                CHECK_NEAR(model.A(kXdot, j), 0.0, 1e-9);
            }
        }
    });

    section("linearize: N=1 pendulum block against the closed form", [] {
        // Independent derivation. At N=1 the linearized link subsystem is
        //   d/dt [theta; omega] = M [theta; omega] + [0; -c] a
        //   M = [[0, 1], [lambda^2, 0]],  lambda^2 = 3g/2L,  c = 3/2L
        // and because M^2 = lambda^2 I, the exact discretization is closed form:
        //   exp(M dt)          = [[cosh, sinh/lambda], [lambda sinh, cosh]]
        //   integral of exp    = [[sinh/lambda, (cosh-1)/lambda^2],
        //                         [cosh-1,      sinh/lambda]]
        // all evaluated at lambda*dt. RK4 matches exp() to O(dt^5), so the agreement
        // should be near rounding -- this is a far sharper check than the
        // first-order picture A ~ I + A_c dt, which is only good to 1e-4 here.
        const double lambda_sq = 3.0 * core::kGravity / (2.0 * kLength);
        const double lambda = std::sqrt(lambda_sq);
        const double c = 3.0 / (2.0 * kLength);
        const double ch = std::cosh(lambda * kDt);
        const double sh = std::sinh(lambda * kDt);

        const baseline::LinearModel model = baseline::linearize(uniform_plant(1), kDt);
        const int theta = kTheta0;
        const int omega = kTheta0 + 1;

        CHECK_NEAR(model.A(theta, theta), ch, 1e-9);
        CHECK_NEAR(model.A(theta, omega), sh / lambda, 1e-9);
        CHECK_NEAR(model.A(omega, theta), lambda * sh, 1e-8);
        CHECK_NEAR(model.A(omega, omega), ch, 1e-9);

        CHECK_NEAR(model.B(theta), -c * (ch - 1.0) / lambda_sq, 1e-11);
        CHECK_NEAR(model.B(omega), -c * sh / lambda, 1e-10);

        // The same numbers to first order, as a sanity anchor while debugging:
        //   A(omega, theta) ~ (3g/2L) dt = 0.0981     A(theta, omega) ~ dt = 0.002
        //   B(omega)        ~ -(3/2L) dt = -0.01
        CHECK(std::abs(model.A(omega, theta) - lambda_sq * kDt) < 1e-5);
        CHECK(std::abs(model.B(omega) + c * kDt) < 1e-6);
    });

    section("linearize: link kinematics at N=2 and N=3", [] {
        // theta_i' picks up dt * omega_i and nothing from omega_j, j != i.
        for (int n = 2; n <= 3; ++n) {
            const baseline::LinearModel model = baseline::linearize(uniform_plant(n), kDt);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    const double expected = (i == j) ? kDt : 0.0;
                    CHECK_NEAR(model.A(kTheta0 + i, kTheta0 + n + j), expected, 1e-6);
                }
            }
        }
    });

    section("linearize about the hanging equilibrium flips the signs", [] {
        // Same derivation as above but about theta = pi, where sin(pi + d) = -d and
        // cos(pi + d) = -1. The subsystem becomes M = [[0, 1], [-lambda^2, 0]], so
        // M^2 = -lambda^2 I and the exponential is trigonometric rather than
        // hyperbolic -- a stable oscillation instead of a divergence, with the input
        // sign reversed because the carriage now pushes the mass the other way.
        //
        // This is the check that the equilibrium argument does what it claims. It is
        // also the mechanism behind the "which links up, which down" goal: every
        // combination of 0 and pi is a valid target.
        const double lambda_sq = 3.0 * core::kGravity / (2.0 * kLength);
        const double lambda = std::sqrt(lambda_sq);
        const double c = 3.0 / (2.0 * kLength);
        const double co = std::cos(lambda * kDt);
        const double si = std::sin(lambda * kDt);

        core::Vec hanging(1);
        hanging[0] = core::kPi;
        const baseline::LinearModel model = baseline::linearize(uniform_plant(1), kDt, hanging);

        const int theta = kTheta0;
        const int omega = kTheta0 + 1;

        CHECK_NEAR(model.A(theta, theta), co, 1e-9);
        CHECK_NEAR(model.A(theta, omega), si / lambda, 1e-9);
        CHECK_NEAR(model.A(omega, theta), -lambda * si, 1e-8);
        CHECK_NEAR(model.A(omega, omega), co, 1e-9);
        // Looser than the equivalent upright check (1e-11), and necessarily so.
        // This entry is ~1e-5, extracted by differencing two angles that sit near pi
        // and differ by ~1e-11. The ULP at pi is 4.4e-16, so roughly 4-5 significant
        // digits of that difference survive -- a cancellation limit of finite
        // differencing about a large base point, not an error in the model. Upright
        // does not suffer it because the base point there is zero.
        CHECK_NEAR(model.B(theta), c * (1.0 - co) / lambda_sq, 1e-9);
        CHECK_NEAR(model.B(omega), c * si / lambda, 1e-10);

        // The carriage block is unchanged -- it does not care where the links are.
        CHECK_NEAR(model.A(kX, kXdot), kDt, 1e-9);
        CHECK_NEAR(model.B(kXdot), kDt, 1e-9);
        CHECK_NEAR(model.A(kIntegral, kX), kDt, 1e-12);
    });

    section("hanging is stabilisable, and trivially so", [] {
        // A sanity check on the whole 2^N-equilibria idea: the solver should handle a
        // non-upright target, and hanging is the easy one -- it is open-loop stable,
        // so the closed loop should sit comfortably inside the unit circle.
        core::Vec hanging(1);
        hanging[0] = core::kPi;

        const baseline::LqrDesign d =
            baseline::design(uniform_plant(1), kDt, baseline::LqrTolerances{}, hanging);

        CHECK(d.riccati.converged);
        CHECK(d.spectral_radius < 1.0);
        CHECK(d.gains.to_flat().allFinite());

        // Upright needs authority to fight gravity; hanging does not. The angle gain
        // should be far smaller here, and that difference is the whole reason the
        // equilibrium has to be an input to the design.
        const baseline::LqrDesign upright =
            baseline::design(uniform_plant(1), kDt, baseline::LqrTolerances{});
        CHECK(std::abs(d.gains.kp_theta[0]) < std::abs(upright.gains.kp_theta[0]));
    });

    // ---- solve_dare ---------------------------------------------------------
    section("solve_dare on a scalar system with a hand-checkable answer", [] {
        // A = B = Q = R = 1 reduces the DARE to p = p - p^2/(1+p) + 1, i.e.
        // p^2 - p - 1 = 0, so P is the golden ratio and k = P/(1+P) = 1/phi.
        // The closed loop is 1 - k = 1/phi^2 = 0.381966, comfortably stable.
        const double phi = 0.5 * (1.0 + std::sqrt(5.0));

        Eigen::MatrixXd A(1, 1);
        A << 1.0;
        Eigen::VectorXd B(1);
        B << 1.0;
        Eigen::MatrixXd Q(1, 1);
        Q << 1.0;

        const baseline::RiccatiResult r = baseline::solve_dare(A, B, Q, 1.0);
        CHECK(r.converged);
        CHECK(r.iterations > 0);
        CHECK_NEAR(r.P(0, 0), phi, 1e-8);
        CHECK_NEAR(r.k(0), phi / (1.0 + phi), 1e-8);
        CHECK(r.residual < 1e-10);
    });

    section("solve_dare on the plant: P is symmetric positive definite", [] {
        const baseline::LinearModel model = baseline::linearize(uniform_plant(1), kDt);
        const baseline::LqrTolerances tol;
        const Eigen::MatrixXd Q = tol.state_penalty(1).asDiagonal();

        const baseline::RiccatiResult r =
            baseline::solve_dare(model.A, model.B, Q, tol.input_penalty());

        CHECK(r.converged);
        CHECK(r.residual < 1e-8);
        CHECK(r.k.size() == core::Gains::size(1));
        CHECK((r.P - r.P.transpose()).norm() / r.P.norm() < 1e-10);

        const Eigen::LLT<Eigen::MatrixXd> llt(r.P);
        CHECK(llt.info() == Eigen::Success);  // positive definite
        CHECK(r.P.diagonal().minCoeff() > 0.0);
    });

    // ---- to_gains and poles -------------------------------------------------
    section("to_gains is the identity on the augmented ordering", [] {
        for (int n = 1; n <= 3; ++n) {
            Eigen::VectorXd k(core::Gains::size(n));
            for (int i = 0; i < k.size(); ++i) {
                k(i) = 0.5 * (i + 1);
            }
            const core::Gains g = baseline::to_gains(k, n);
            CHECK(g.n_links() == n);
            CHECK((g.to_flat() - k).norm() < 1e-15);
        }
    });

    section("closed_loop_poles are inside the unit circle", [] {
        const baseline::LinearModel model = baseline::linearize(uniform_plant(1), kDt);
        const baseline::LqrTolerances tol;
        const Eigen::MatrixXd Q = tol.state_penalty(1).asDiagonal();
        const baseline::RiccatiResult r =
            baseline::solve_dare(model.A, model.B, Q, tol.input_penalty());

        const std::vector<std::complex<double>> poles =
            baseline::closed_loop_poles(model, r.k);
        CHECK(poles.size() == static_cast<std::size_t>(core::Gains::size(1)));

        double radius = 0.0;
        for (const auto& p : poles) {
            radius = std::max(radius, std::abs(p));
        }
        CHECK(radius < 1.0);
        // A pole parked at exactly 1 is the integrator failing to be stabilised,
        // not a solver bug -- so check it is clear of the circle, not just inside.
        CHECK(radius < 0.9999);
    });

    // ---- end to end ---------------------------------------------------------
    section("design produces a stabilising gain set", [] {
        for (int n = 1; n <= 2; ++n) {
            const baseline::LqrDesign d =
                baseline::design(uniform_plant(n), kDt, baseline::LqrTolerances{});

            CHECK(d.riccati.converged);
            CHECK(d.spectral_radius < 1.0);
            CHECK(d.gains.n_links() == n);
            CHECK(d.gains.to_flat().allFinite());

            // Sign sanity: leaning forward must command forward acceleration to
            // drive the carriage under the pendulum. kp_theta is negated inside
            // Controller::compute, so the gain itself is negative.
            CHECK(d.gains.kp_theta[0] < 0.0);
        }
    });

    section("envelope_report stabilises a small tilt inside the limits", [] {
        const core::Plant plant = uniform_plant(1);
        const core::StepperLimits limits;
        const baseline::LqrDesign d =
            baseline::design(plant, kDt, baseline::LqrTolerances::from_envelope(limits, 0.05, 0.5));

        const baseline::EnvelopeReport report =
            baseline::envelope_report(plant, d.gains, limits, kDt, 0.05, 3.0);

        CHECK(report.stabilised);
        CHECK(std::abs(report.settled_tilt) < 0.01);
        CHECK(report.peak_accel_cmd > 0.0);
        CHECK(std::isfinite(report.peak_accel_cmd));
        CHECK(std::isfinite(report.peak_belt_force));
        CHECK(report.peak_excursion < 0.5 * limits.rail_length);
        CHECK(report.within_envelope ==
              (report.peak_accel_cmd <= limits.max_accel &&
               report.peak_velocity <= limits.max_velocity &&
               report.peak_excursion <= 0.5 * limits.rail_length));
    });

    return check::summary("lqr");
}
