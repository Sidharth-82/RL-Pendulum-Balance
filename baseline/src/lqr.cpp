#include "baseline/lqr.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>

namespace baseline {

LqrTolerances LqrTolerances::from_envelope(const core::StepperLimits& limits, double max_tilt,
                                           double max_tilt_rate) {
    // TODO: Bryson from the hardware.
    //   x     <- a fraction of rail_length/2 (leave headroom; the rail end is a
    //            termination, not a soft limit, so do not tolerate reaching it)
    //   xdot  <- max_velocity
    //   accel <- max_accel
    //   theta <- max_tilt, omega <- max_tilt_rate
    // integral_x has no hardware analogue: keep the struct default unless steady-state
    // offset turns out to matter, since integral action costs phase margin this plant
    // has none of.
    // Half of the usable half-rail. The full half-length is where the carriage runs
    // out of track and the episode terminates, so treating that excursion as merely
    // "tolerable" would price a hard failure at a cost of 1.
    LqrTolerances ret_lqr_tol{.x = 0.25 * limits.rail_length, .xdot = limits.max_velocity,
                              .theta = max_tilt, .omega = max_tilt_rate,
                              .accel = 0.5*limits.max_accel};

    return ret_lqr_tol;
}

Eigen::VectorXd LqrTolerances::state_penalty(int n_links) const {
    // TODO: 1/tol^2 per entry, in the augmented ordering
    //   [x, xdot, integral_x, theta * n, omega * n]
    // Length is core::Gains::size(n_links). Assert every tolerance is positive --
    // a zero here is a division by zero that reaches Q as an infinity and comes back
    // as a NaN gain, which is far harder to trace from the far end.
    // Assert on the tolerance, not on the penalty: a zero tolerance divides to
    // +inf, and inf passes a > 0 test happily before poisoning the Riccati solve
    // with a NaN gain several functions downstream.
    const auto penalty = [](double tolerance) {
        assert(tolerance > 0.0);
        return 1.0 / (tolerance * tolerance);
    };

    Eigen::VectorXd ret_vec(core::Gains::size(n_links));
    ret_vec[0] = penalty(x);
    ret_vec[1] = penalty(xdot);
    ret_vec[2] = penalty(integral_x);
    for (int i = 0; i < n_links; ++i) {
        ret_vec[3 + i] = penalty(theta);
        ret_vec[3 + n_links + i] = penalty(omega);
    }

    return ret_vec;
}

double LqrTolerances::input_penalty() const {
    // TODO: 1 / accel^2.
    return 1/(accel * accel);
}

LinearModel linearize(const core::Plant& plant, double dt, const core::Vec& theta_ref) {
    const int n = plant.n_links();
    const int m = core::Gains::size(n);  // one definition of the augmented size
    assert(theta_ref.size() == 0 || theta_ref.size() == n);

    // Base point: the equilibrium being linearized about. Everything below measures
    // deviation from this, which is why the resulting gains apply to the wrapped
    // error rather than to the raw angle.
    core::State base;
    base.resize(n);
    if (theta_ref.size() == n) {
        base.theta = theta_ref;
    }

    // A non-equilibrium reference would leave a constant term the linear model has no
    // way to represent, so verify rather than trust: stepping the base point with no
    // command must leave it where it started.
    core::State drift = base;
    plant.step(drift, 0.0, dt);
    assert((drift.flatten() - base.flatten()).norm() < 1e-9);

    const Eigen::VectorXd base_flat = base.flatten();

    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(m,m);
    Eigen::VectorXd B = Eigen::VectorXd::Zero(m);
    Eigen::VectorXd e = base_flat;
    core::State plus;
    core::State minus;
    core::State plus_u = base;
    core::State minus_u = base;

    const auto aug = [](int j){
        return j < 2 ? j : j + 1;
    };

    for(int j = 0; j < core::State::flat_size(n); ++j){
        e(j) = base_flat(j) + kEps;
        plus.unflatten(e, n);
        plant.step(plus, 0.0, dt);
        e(j) = base_flat(j) - kEps;
        minus.unflatten(e,n);
        plant.step(minus, 0.0, dt);
        e(j) = base_flat(j);
        Eigen::VectorXd column = (plus.flatten() - minus.flatten()) / (2.0 * kEps);

        for (int i = 0; i < column.size(); ++i){
            A(aug(i), aug(j)) = column(i);
        }
    }

    plant.step(plus_u, kEps, dt);
    plant.step(minus_u, -kEps, dt);

    const Eigen::VectorXd input_column =
        (plus_u.flatten() - minus_u.flatten()) / (2.0 * kEps);
    for (int i = 0; i < input_column.size(); ++i) {
        B(aug(i)) = input_column(i);
    }
    A(2,0) = dt;
    A(2,2) = 1.0;

    LinearModel ret_model{.A = A, .B = B, .dt = dt, .n_links = n};
    
    return ret_model;
}

RiccatiResult solve_dare(const Eigen::MatrixXd& A, const Eigen::VectorXd& B,
                         const Eigen::MatrixXd& Q, double R, double tolerance,
                         int max_iterations) {
    // TODO: iterate P <- A'PA - A'PB (R + B'PB)^-1 B'PA + Q from P = Q until the
    // change in P stops moving, then k = (R + B'PB)^-1 B'PA.
    //
    // B is a single column, so R + B'PB is a scalar -- no matrix inverse anywhere in
    // this loop, just a divide. Guard it anyway: it is positive by construction only
    // while P stays positive semidefinite.
    //
    // Report, do not assume:
    //   - converged = (change fell below tolerance before max_iterations)
    //   - residual  = ||A'PA - P - A'PB(R+B'PB)^-1 B'PA + Q|| / ||P||
    // Convergence is linear and the rate is set by the closed-loop poles, so a slow
    // plant at 500 Hz can want thousands of iterations. That is fine -- each one is a
    // 15x15 product -- but it means "hit max_iterations" is a real outcome to report
    // rather than a can't-happen.
    
    auto n = static_cast<int>(A.rows());
    Eigen::MatrixXd P = Eigen::MatrixXd::Zero(n, n);   // cost-to-go, symmetric positive definite
    Eigen::VectorXd k = Eigen::VectorXd::Zero(n);   // feedback row, length 2N+3; the command is -k.dot(z)
    
    int i = 0;
    double change = tolerance + 1;
    bool converged = false;
    double residual = 0.0;

    while(i < max_iterations && !converged){
        auto denom = 1.0/(R + B.dot(P * B));
        auto prev_P = P;
        P = A.transpose()*P*A
            - A.transpose() * P * B * denom * B.transpose() * P * A + Q;
            
        change = (P - prev_P).norm() / std::max(1.0, P.norm());
        converged = (change < tolerance);
        ++i;
    }
    


    // k and the residual are read off the FINAL P, with a denominator recomputed
    // from it. The denom carried out of the loop belongs to the previous iterate;
    // that discrepancy vanishes only if the loop actually converged, so reusing it
    // would make the max_iterations case quietly wrong rather than merely
    // unconverged.
    const double final_denom = 1.0 / (R + B.dot(P * B));
    assert(final_denom > 0.0);  // positive only while P stays positive semidefinite

    k = (B.transpose() * P * A).transpose() * final_denom;

    // Relative, not absolute: with Bryson weights Q carries entries in the hundreds,
    // so ||P|| is large and a bare norm would mean a different thing at every choice
    // of tolerances. Computed once here rather than per iteration -- it is three
    // more matrix products, and the loop can run for thousands of passes.
    const Eigen::MatrixXd gap =
        A.transpose() * P * A - P
        - A.transpose() * P * B * final_denom * B.transpose() * P * A + Q;
    residual = gap.norm() / std::max(1.0, P.norm());

    RiccatiResult ret_riccati{
        .P = P, .k = k, .iterations = i, .converged = converged, .residual = residual};
    return ret_riccati;
}

core::Gains to_gains(const Eigen::VectorXd& k, int n_links) {
    // TODO: assert k.size() == core::Gains::size(n_links), then
    // core::Gains::from_flat(k, n_links). The augmented ordering was chosen to make
    // this the identity; if it ever stops being the identity, that is a decision to
    // make here and nowhere else.
    assert(k.size() == core::Gains::size(n_links));

    return core::Gains::from_flat(k, n_links);
}

std::vector<std::complex<double>> closed_loop_poles(const LinearModel& model,
                                                    const Eigen::VectorXd& k) {
    // TODO: Eigen::EigenSolver on (model.A - model.B * k.transpose()), return
    // eigenvalues(). EigenSolver, not SelfAdjointEigenSolver -- the closed-loop
    // matrix is not symmetric and the eigenvalues are genuinely complex.
    
    // No eigenvectors wanted, so do not compute them -- that is the bulk of the work
    // in a non-symmetric decomposition.
    Eigen::EigenSolver<Eigen::MatrixXd> solver(model.A - model.B * k.transpose(), false);
    assert(solver.info() == Eigen::Success);

    // Dynamically sized: there are 2N+3 poles, not 2.
    const Eigen::VectorXcd eigenval = solver.eigenvalues();
    std::vector<std::complex<double>> ret_vec(eigenval.data(), eigenval.data() + eigenval.size());
    return ret_vec;
}

EnvelopeReport envelope_report(const core::Plant& plant, const core::Gains& gains,
                               const core::StepperLimits& limits, double dt,
                               double initial_tilt, double duration) {
    // TODO: close the loop on the nonlinear plant from theta = initial_tilt, all else
    // zero, and record the peaks.
    //
    //   Controller controller(gains, dt, integral_limit);
    //   loop:
    //     a = controller.compute(state.x, state.xdot, state.theta, state.omega);
    //     plant.step(state, a, dt, &peak_belt_force);
    //
    // Ideal state feedback, no SensorModel, and the command goes straight to the
    // plant WITHOUT passing through Actuator -- clamping it would hide the violation
    // this function exists to find. Track peak |a|, |xdot|, |x| and the peak belt
    // force across the RK4 stages, then compare against limits.
    //
    // stabilised: the tilt actually decayed rather than the run merely ending.
    // Check the tail, not the final sample -- a pendulum passing through zero on its
    // way over the top looks stabilised for one instant.
    //
    // Once this reads clean, the follow-up run is the same loop with the command
    // pushed through Actuator and Actuator::would_lose_steps checked against the peak
    // belt force. That answers the question this one deliberately does not: whether
    // it still stabilises when saturated.
    // Past horizontal: a linear gain law is not recovering from there, so the run
    // has nothing left to tell us.
    constexpr double kPastHorizontal = 1.5707963267948966;  // pi/2

    double integral_limit = 1e6; //Deliberately high to measure unclamped demand
    double peak_belt_force = 0.0;
    int steps = duration/dt;
    double a = 0.0;
    bool diverged = false;
    EnvelopeReport ret_env_rep;

    core::Controller controller(gains, dt, integral_limit);
    core::State state;

    state.resize(plant.n_links());
    state.theta[0] = initial_tilt;

    for(int i = 0; i < steps; ++i){
        a = controller.compute(state.x, state.xdot, state.theta, state.omega);

        ret_env_rep.peak_accel_cmd = std::max(ret_env_rep.peak_accel_cmd, std::abs(a));

        plant.step(state, a, dt, &peak_belt_force);

        ret_env_rep.peak_belt_force = std::max(ret_env_rep.peak_belt_force,
                                               std::abs(peak_belt_force));

        ret_env_rep.peak_excursion = std::max(ret_env_rep.peak_excursion, std::abs(state.x));

        ret_env_rep.peak_velocity = std::max(ret_env_rep.peak_velocity, std::abs(state.xdot));
        
        const double tilt = state.theta.cwiseAbs().maxCoeff();

        // Tail window, anchored to the end of the fixed run rather than trailing
        // the current step: the question is where the trajectory settled once the
        // transient had time to finish, and a moving window would keep re-reading
        // the fall.
        if (i >= steps * 0.8) {
            ret_env_rep.settled_tilt = std::max(ret_env_rep.settled_tilt, tilt);
        }

        // Bail on divergence, never on success. Peaks -- excursion especially --
        // keep building long after the tilt is caught, because the carriage returns
        // to centre far more slowly than the pendulum comes upright. Stopping when
        // it "looks upright" would truncate the measurement this function exists to
        // take. Once it has genuinely diverged there is nothing left to learn, and
        // continuing only integrates garbage into the peaks.
        if (!std::isfinite(a) || !std::isfinite(state.x) || !std::isfinite(state.xdot) ||
            !state.theta.allFinite() || !state.omega.allFinite() ||
            tilt > kPastHorizontal || std::abs(state.x) > 0.5 * limits.rail_length) {
            diverged = true;
            // The tail window was never reached, so report the tilt where it failed
            // rather than leaving a zero that reads as "settled perfectly".
            ret_env_rep.settled_tilt = tilt;
            break;
        }
    }

    // Decay measured against where it started, with an absolute floor so a tiny
    // initial tilt does not demand unreasonable precision. "Ended near zero" is not
    // enough on its own -- a run on its way over the top passes through zero.
    const double decayed = std::max(0.1 * std::abs(initial_tilt), 1e-3);
    ret_env_rep.stabilised = !diverged && ret_env_rep.settled_tilt < decayed;

    // Belt force is deliberately not part of this verdict. Whether that force costs
    // steps depends on available_force() at the speed being travelled, which is an
    // Actuator question -- reported here, judged in the saturated follow-up run.
    ret_env_rep.within_envelope = ret_env_rep.peak_accel_cmd <= limits.max_accel &&
                                  ret_env_rep.peak_velocity <= limits.max_velocity &&
                                  ret_env_rep.peak_excursion <= 0.5 * limits.rail_length;

    return ret_env_rep;
}

LqrDesign design(const core::Plant& plant, double dt, const LqrTolerances& tolerances,
                 const core::Vec& theta_ref) {
    // TODO: linearize -> state_penalty/input_penalty -> solve_dare -> to_gains, then
    // fill spectral_radius from closed_loop_poles. Thin wiring only; every decision
    // belongs in the function that owns it.
    LinearModel lin_model = linearize(plant, dt, theta_ref);
    
    Eigen::VectorXd Q = tolerances.state_penalty(lin_model.n_links);
    double R = tolerances.input_penalty();

    RiccatiResult ric_res = solve_dare(lin_model.A, lin_model.B,Q.asDiagonal(),R);

    auto poles_vec = closed_loop_poles(lin_model, ric_res.k);

    auto max_abs_it = std::max_element(poles_vec.begin(), poles_vec.end(),
        [](const std::complex<double>& a, const std::complex<double>& b) {
            return std::abs(a) < std::abs(b);
        }
    );

    auto gains = to_gains(ric_res.k, lin_model.n_links);

    LqrDesign ret_lqr_design{.gains = gains,
                             .model = lin_model,
                             .riccati = ric_res, 
                             .spectral_radius = std::abs(*max_abs_it)};
                            
    return ret_lqr_design;
}

}  // namespace baseline
