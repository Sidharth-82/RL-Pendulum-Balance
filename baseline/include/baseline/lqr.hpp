#pragma once

#include <complex>
#include <vector>

#include <Eigen/Dense>

#include "core/actuator.hpp"
#include "core/controller.hpp"
#include "core/plant.hpp"

namespace baseline {

constexpr double kEps = 1e-6;

// LQR baseline: linearize about upright, solve the Riccati equation, emit gains in
// the same 2N+3 layout the RL policy will output.
//
// Worth having before PPO for three separate reasons, in descending order of value:
// it answers "is this plant stabilizable at all inside the actuator envelope" from a
// solve rather than from a training run that quietly fails to converge; it gives PPO
// an initialization that is already stabilizing, which is a fundamentally easier
// starting point than noise on an open-loop-unstable plant; and it is the number the
// policy has to beat.
//
// Where LQR structurally cannot win is latency, quantisation, saturation and
// parameter spread -- none of which appear in the linear model. The gap between the
// gains designed here and what survives the full sensing chain IS the sim-to-real
// problem, so measuring it is a deliverable of this module, not an afterthought.
//
// Everything here is heap-backed MatrixXd/VectorXd rather than core's inline-storage
// Vec and Mat: the augmented state is 2N+3, which exceeds kMaxLinks, and none of this
// runs in a hot loop. Same reasoning as State::flatten.

// Largest deviation in each state that is considered acceptable, plus the largest
// acceptable command. Bryson's rule turns these into the cost matrices:
//   Q = diag(1 / max_i^2),   R = 1 / accel_max^2
//
// Expressed as tolerances rather than as raw weights because the tolerances have
// units and physical meaning -- and because from_envelope() can then derive most of
// them from the hardware, which is what makes "stabilizable inside the actuator
// envelope" a property of the design instead of something checked afterwards.
struct LqrTolerances {
    double x = 0.10;           // m, carriage excursion
    double xdot = 0.20;        // m/s
    // m s. Chosen by sweep against a rail off level by 1 degree, 2026-08-12, and it
    // does NOT behave the way "integral action costs phase margin" suggests. Weakening
    // this term drives the integrator's own pole toward z = 1 -- the closed-loop
    // spectral radius went 0.9976 -> 0.99998 across 0.05 -> 20, and the Riccati
    // iteration count ran into its cap. The real trade is against kp_theta: a weaker
    // integral leaves less authority on position, so kp_theta falls and the
    // recoverable tilt widens (2.78 deg -> 3.72 deg) while disturbance rejection slows
    // from seconds to never. 0.2 takes most of the basin (3.32 deg) at zero measured
    // steady-state error.
    //
    // The cost that would argue for going looser -- phase lag eating stability margin
    // -- is invisible to this design, which is delay-free by construction. Revisit
    // once the gains have been run through SensorModel and the actuation delay.
    double integral_x = 0.20;
    double theta = 0.05;       // rad, tilt (~3 degrees)
    double omega = 0.50;       // rad/s
    double accel = 2.00;       // m/s^2, commanded carriage acceleration

    // Derives the carriage and command tolerances from the hardware envelope, so a
    // domain-randomised StepperLimits re-derives its own weights rather than reusing
    // weights tuned for a different motor. Angular tolerances stay as arguments --
    // no hardware limit implies them.
    static LqrTolerances from_envelope(const core::StepperLimits& limits, double max_tilt,
                                       double max_tilt_rate);

    // Diagonal of Q, in the augmented ordering. Length Gains::size(n_links).
    Eigen::VectorXd state_penalty(int n_links) const;

    // R, a scalar because the plant has exactly one input.
    double input_penalty() const;
};

// Discrete-time linear model of the augmented system, at the control period.
//
//   z = [x, xdot, integral_x, theta, omega]        (2N+3, == core::augmented_state)
//   z[k+1] = A z[k] + B a[k]
//
// Discrete rather than continuous because the controller genuinely runs at a fixed
// rate with a zero-order hold on the command, and because a discrete design needs no
// Hamiltonian eigendecomposition -- Eigen has no ordered Schur form, so the
// continuous route means hand-writing stable-subspace extraction.
struct LinearModel {
    Eigen::MatrixXd A;
    Eigen::VectorXd B;
    double dt = 0.0;
    int n_links = 0;
};

// Builds A and B by central-differencing Plant::step itself, then appending the
// integrator row analytically.
//
// Differencing the real step map rather than writing an analytic Jacobian means the
// model is automatically the one the controller actually faces -- including the
// zero-order hold and RK4's own behaviour -- and there is no second derivation to
// keep in sync with the plant. The equilibrium is exact (theta = omega = 0, a = 0 is
// a fixed point of step()), so the differences are clean.
//
// The analytic form is still worth knowing, because it is what the N=1 test checks
// against:  D w' = g diag(h) theta - L omega - a h,  giving 3g/2L and -3/2L for a
// uniform rod.
//
// theta_ref selects which equilibrium to linearize about. Every combination of 0 and
// pi is an equilibrium -- gravity torque vanishes at both -- so an N-link chain has
// 2^N, and the state of the returned model is the DEVIATION from the chosen one.
// Empty means all-upright. Linearizing about hanging flips the signs exactly as you
// would expect: cos(lambda dt) in place of cosh, and a stable oscillation rather than
// a divergence.
//
// Whether a given equilibrium is stabilisable at all is answered by running
// solve_dare against it and checking the spectral radius -- worth doing before
// designing anything that assumes a particular configuration is reachable.
LinearModel linearize(const core::Plant& plant, double dt,
                      const core::Vec& theta_ref = core::Vec());

struct RiccatiResult {
    Eigen::MatrixXd P;   // cost-to-go, symmetric positive definite
    Eigen::VectorXd k;   // feedback row, length 2N+3; the command is -k.dot(z)
    int iterations = 0;
    bool converged = false;
    double residual = 0.0;  // relative Riccati residual, ||.|| / ||P||
};

// Backward iteration on the discrete algebraic Riccati equation:
//   P <- A'PA - A'PB (R + B'PB)^-1 B'PA + Q,  from P = Q
//
// Converges linearly, which is fine at this size -- each iteration is one small
// matrix product. Never infer success from the fact that it returned: check
// `converged`, and check `residual`.
RiccatiResult solve_dare(const Eigen::MatrixXd& A, const Eigen::VectorXd& B,
                         const Eigen::MatrixXd& Q, double R, double tolerance = 1e-12,
                         int max_iterations = 200000);

// Identity mapping, given that the augmented ordering was chosen to match
// Gains::to_flat. Named anyway so the assumption lives in one place and is testable
// -- see the layout lock in controller_test.
core::Gains to_gains(const Eigen::VectorXd& k, int n_links);

// Eigenvalues of A - B k'. All must lie strictly inside the unit circle.
//
// Watch for a pole sitting at exactly 1.0: that is the integrator state failing to
// be stabilized, not a solver bug.
std::vector<std::complex<double>> closed_loop_poles(const LinearModel& model,
                                                    const Eigen::VectorXd& k);

// What the designed gains actually demand of the hardware, from a representative
// initial tilt.
//
// Run on the NONLINEAR plant with ideal state feedback and without pushing the
// command through Actuator: clamping the command would hide the very violation this
// is looking for. Once the unclamped answer is known, re-running through Actuator is
// the follow-up question -- does it still stabilise when saturated.
struct EnvelopeReport {
    double peak_accel_cmd = 0.0;    // m/s^2
    double peak_velocity = 0.0;     // m/s
    double peak_excursion = 0.0;    // m from centre
    double peak_belt_force = 0.0;   // N
    double settled_tilt = 0.0;      // rad at the end of the run
    bool within_envelope = false;   // every peak inside StepperLimits
    bool stabilised = false;        // tilt actually decayed
};

EnvelopeReport envelope_report(const core::Plant& plant, const core::Gains& gains,
                               const core::StepperLimits& limits, double dt,
                               double initial_tilt, double duration);

// The whole pipeline: linearize, solve, map, and report the closed-loop spectrum.
struct LqrDesign {
    core::Gains gains;
    LinearModel model;
    RiccatiResult riccati;
    double spectral_radius = 0.0;  // largest |eigenvalue| of A - B k'; must be < 1
};

LqrDesign design(const core::Plant& plant, double dt, const LqrTolerances& tolerances,
                 const core::Vec& theta_ref = core::Vec());

}  // namespace baseline
