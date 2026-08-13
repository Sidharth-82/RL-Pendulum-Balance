#pragma once

#include "core/types.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <format>

namespace core {

// Physical description of the chain. All per-link vectors are length n_links.
struct PendulumParams {
    double carriage_mass = 0.5;  // kg, everything that translates with the belt

    Vec masses;         // kg, per link
    Vec lengths;        // m, joint-to-joint per link
    Vec com_dists;      // m, proximal joint to that link's centre of mass
    Vec inertias;       // kg m^2, about each link's own centre of mass
    Vec joint_damping;  // N m s / rad, resisting rotation relative to the parent link

    int n_links() const;

    // Identical thin uniform rods: com_dist = L/2, inertia = m L^2 / 12.
    // The usual starting point, and the case with a hand-checkable closed form.
    static PendulumParams uniform_rods(int n_links, double mass, double length,
                                       double damping, double carriage_mass);

    // Throws if vector lengths disagree, or if any mass/length/inertia is
    // non-positive. Cheap, and it turns a domain-randomisation bug that would
    // otherwise surface as NaNs deep in a rollout into an immediate error.
    void validate() const;
};

// Output of one dynamics evaluation.
struct Derivatives {
    Vec alpha;                // angular accelerations, rad/s^2
    double belt_force = 0.0;  // longitudinal force the belt must transmit, N
};

// N-link inverted pendulum on a translating carriage.
//
// The carriage is a prescribed-motion boundary condition rather than a free body:
// a belt-driven stepper is a position source, so its acceleration is commanded,
// not solved for. The belt force needed to enforce that motion comes back out so
// the caller can detect step loss -- the failure mode an open-loop stepper has no
// way to notice on its own.
//
//     A(th) alpha = g (h .* sin th) - a (h .* cos th) - C(th) (om .* om) + Q
//
//     A_ik = D_ik cos(th_i - th_k)
//     C_ik = D_ik sin(th_i - th_k)
//
// A is a mass matrix, hence symmetric positive definite -- solve it with LDLT,
// not a general LU.
class Plant {
public:
    explicit Plant(const PendulumParams& params, double gravity = kGravity);

    int n_links() const;
    const PendulumParams& params() const;

    // One dynamics evaluation at the given carriage acceleration.
    //
    // Evaluate sin/cos once per link and expand the differences via
    //   cos(a-b) = cos a cos b + sin a sin b
    //   sin(a-b) = sin a cos b - cos a sin b
    // rather than calling trig per matrix entry. At N=3 that is 6 transcendentals
    // instead of 18, and the transcendentals dominate this function.
    Derivatives accel(const Vec& theta, const Vec& omega, double carriage_accel) const;

    // One RK4 step at zero-order-hold carriage acceleration, advancing theta,
    // omega, x and xdot in place.
    //
    // RK4 rather than Euler because an inverted pendulum is open-loop unstable:
    // an integrator that injects energy makes the plant harder to stabilise than
    // the real hardware, and gains tuned against it transfer wrong.
    //
    // peak_belt_force, if given, receives the maximum |force| across the four
    // stages -- that is what a step-loss check should be tested against, not the
    // value at any single stage.
    void step(State& state, double carriage_accel, double dt,
              double* peak_belt_force = nullptr) const;

    // Total mechanical energy. With zero damping and zero carriage acceleration
    // this is conserved, which is the real check on the derivation -- far stronger
    // than eyeballing a trajectory, and it localises sign errors in the D_ik cross
    // terms that are entirely invisible at N=1.
    //
    //   T = 0.5 M_total xdot^2 + xdot sum_j h_j cos th_j om_j
    //       + 0.5 sum_jk D_jk cos(th_j - th_k) om_j om_k
    //   V = g sum_j h_j cos th_j
    double energy(const Vec& theta, const Vec& omega, double xdot) const;

private:
    // Fills h_, D_ and total_mass_ from params_. Configuration-independent, so
    // this runs once in the constructor and never in the hot loop.
    //
    //   beyond_i = sum_{j>i} m_j
    //   h_i      = m_i l_i + L_i beyond_i
    //   D_ii     = I_i + m_i l_i^2 + L_i^2 beyond_i
    //   D_ik     = L_min(i,k) h_max(i,k)        (i != k, symmetric)
    void build_inertia_terms();

    // Joint i resists rotation relative to link i-1, with the carriage taken as
    // non-rotating (omega_{-1} = 0). Each joint torque therefore enters two
    // equations with opposite sign:
    //   tau_i = b_i (om_i - om_{i-1});  Q_i = -tau_i + tau_{i+1},  tau_n = 0
    Vec damping_torques(const Vec& omega) const;

    PendulumParams params_;
    double gravity_ = kGravity;
    int n_ = 0;

    Vec h_;                     // gravity / carriage-acceleration coupling per link
    Mat D_;                     // configuration-independent part of the mass matrix
    double total_mass_ = 0.0;   // carriage plus all links
};

}  // namespace core
