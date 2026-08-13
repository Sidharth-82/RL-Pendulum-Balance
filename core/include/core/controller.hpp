#pragma once

#include "core/types.hpp"

namespace core {

// The 2N+3 gains of the parallel state-feedback law.
//
//   a_cmd = -( kp_x x + kd_x xdot + ki_x integral(x)
//              + sum_i [ kp_theta_i theta_i + kd_theta_i omega_i ] )
//
// Parallel rather than cascaded: with N+1 degrees of freedom and one actuator
// there is no sensible way to nest N loops, and this form is just state feedback,
// so it extends to any N and can be seeded from an LQR solution.
//
// Deliberately no integral term on angle. Integral action adds phase lag exactly
// where an inverted pendulum needs lead, and any steady angle offset is absorbed
// by the position loop anyway.
struct Gains {
    double kp_x = 0.0;
    double kd_x = 0.0;
    double ki_x = 0.0;  // small; compensates a rail that is not level
    Vec kp_theta;
    Vec kd_theta;

    int n_links() const;

    static int size(int n_links);  // 2n + 3

    // Flat form is the RL action space and the LQR output. Ordering:
    //   [kp_x, kd_x, ki_x, kp_theta_0..n-1, kd_theta_0..n-1]
    // Must stay consistent with State::flatten -- a mismatch here produces a
    // controller that is wrong but stable-looking, which is the worst kind.
    static Gains from_flat(const Eigen::Ref<const Eigen::VectorXd>& v, int n_links);
    Eigen::VectorXd to_flat() const;
};

// The state vector to_flat() is a dot product against:
//   [x, xdot, integral_x, theta_0..n-1, omega_0..n-1]
// so that  compute(...) == -to_flat().dot(augmented_state(...)).
//
// Note this is NOT State::flatten's ordering -- integral_x is inserted at index 2,
// which shifts every link gain by one. That offset is exactly the mismatch the
// comment above warns about, so it is spelled once here rather than open-coded at
// each call site, and locked by a test that evaluates the control law both ways.
//
// It is also the state ordering the LQR baseline designs against: augmenting the
// plant with the integrator makes the Riccati gain vector map onto Gains as the
// identity, which is the whole reason for the augmentation.
//
// theta_ref is the target equilibrium; the theta entries carry the wrapped
// deviation from it, because that is what the gains multiply. Pass an empty Vec
// for the all-upright equilibrium.
Eigen::VectorXd augmented_state(const State& state, double integral_x,
                                const Vec& theta_ref = Vec());

// Parallel state-feedback controller emitting a commanded carriage acceleration.
//
// This is the one class that is compiled into both the simulator and the Pi
// binary. Keep it free of anything platform-specific: no I/O, no allocation, no
// clock reads. Gains tuned in simulation only transfer if the code consuming
// them is byte-identical, which is the entire reason core/ links nothing but
// Eigen.
//
// Command acceleration rather than velocity: the pendulum responds to carriage
// acceleration, so commanding velocity would put an extra integrator in the plant
// and change the loop shaping. Integration to a step rate belongs in the actuator.
class Controller {
public:
    // dt must equal the real control period. Two of the three position gains are
    // dimensional in time (ki_x in 1/s, kd_x in s), so a mismatch between the
    // rate used for tuning and the rate on hardware makes the gains numerically
    // wrong even with perfect physics.
    Controller(const Gains& gains, double dt, double integral_limit);

    // Clears integral state. Call between episodes, and on the Pi whenever
    // control is re-engaged, or stale integral windup drives the first command.
    void reset();

    // Commanded carriage acceleration, m/s^2.
    //
    // Takes *measured* state, not truth -- in simulation the sensor model sits
    // between the plant and this call, which is what stops kd_theta being tuned
    // against a noise-free derivative it will never see on hardware.
    double compute(double x, double xdot, const Vec& theta, const Vec& omega);

    // Integral clamp is the anti-windup scheme. ki_x is meaningless without one:
    // the actuator saturates, error keeps accumulating, and recovery overshoots
    // badly. Whatever scheme you pick here has to be the one the Pi runs.
    void set_gains(const Gains& gains);
    const Gains& gains() const;

    // Target equilibrium, one angle per link. Every combination of 0 and pi is an
    // equilibrium of this plant -- gravity torque vanishes at both -- so an N-link
    // chain has 2^N of them, and "link 1 up, link 2 down" is a legitimate setpoint
    // rather than a special case. Defaults to all-upright.
    //
    // The deviation is wrapped into (-pi, pi] before the gains see it. That matters
    // during swing-up, where theta runs through several revolutions: an unwrapped
    // deviation from a pi setpoint is nonsense at exactly the moment the balancing
    // controller takes over.
    //
    // Does not reset the integral, for the same reason set_gains does not.
    void set_theta_ref(const Vec& theta_ref);
    const Vec& theta_ref() const;

    double integral() const;

private:
    Gains gains_;
    Vec theta_ref_;
    double dt_ = 0.0;
    double integral_limit_ = 0.0;
    double integral_x_ = 0.0;
};

}  // namespace core
