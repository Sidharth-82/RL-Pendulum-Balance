#pragma once

#include <cmath>

#include <Eigen/Dense>

namespace core {

inline constexpr double kPi = 3.14159265358979323846;
inline constexpr double kTwoPi = 6.28318530717958647693;

// Folds an angle into (-pi, pi].
//
// Needed wherever an angle is compared against a setpoint. During swing-up theta
// runs unbounded -- several full revolutions -- so a raw (theta - theta_ref)
// against a pi setpoint is meaningless exactly when the controller is about to
// take over. Every equilibrium of this plant has each link at 0 or pi, so a
// setpoint of pi is a normal case, not an exotic one.
inline double wrap_angle(double angle) {
    double wrapped = std::fmod(angle + kPi, kTwoPi);
    if (wrapped <= 0.0) {
        wrapped += kTwoPi;
    }
    return wrapped - kPi;
}

// Compile-time ceiling on the link count. Storage is sized to this, so raising it
// costs memory in every matrix; the arithmetic still runs at the runtime size.
// 6 is generous -- a triple pendulum is already research-grade.
inline constexpr int kMaxLinks = 6;

// Longest delay any DelayLine can represent, in control steps. At 500 Hz this is
// 200 ms, well beyond any plausible sensing or actuation lag.
inline constexpr int kMaxDelaySteps = 100;

inline constexpr double kGravity = 9.80665;

// Runtime-sized, inline-stored. The Dynamic sizes mean one type serves any link
// count; the Max template arguments mean the storage is a fixed inline array, so
// nothing here ever touches the heap.
using Vec = Eigen::Matrix<double, Eigen::Dynamic, 1, 0, kMaxLinks, 1>;
using Mat = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, 0, kMaxLinks, kMaxLinks>;

// Full system state. Carriage and links are kept separate because the dynamics
// read far more clearly that way; the stacked [x, xdot, theta, omega] form is
// only needed at the LQR and RL boundaries, where flatten()/unflatten() convert
// explicitly rather than the whole codebase carrying an awkward layout.
struct State {
    double x = 0.0;     // carriage position along the rail from the center, m
    double xdot = 0.0;  // carriage velocity, m/s
    Vec theta;          // link angles measured from upright, rad
    Vec omega;          // link angular rates, rad/s

    // Allocates theta/omega to n links and zeroes everything.
    void resize(int n_links);

    int n_links() const;

    // Stacked ordering is [x, xdot, theta_0..theta_{n-1}, omega_0..omega_{n-1}],
    // length 2n+2. Keep this ordering consistent with Gains::to_flat -- the LQR
    // gain vector and the state vector have to agree or the Riccati solution is
    // silently wrong rather than obviously wrong.
    Eigen::VectorXd flatten() const;
    void unflatten(const Eigen::Ref<const Eigen::VectorXd>& v, int n_links);

    static int flat_size(int n_links);
};

}  // namespace core
