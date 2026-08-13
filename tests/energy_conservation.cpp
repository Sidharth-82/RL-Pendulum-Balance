// Multi-link plant validation. Three checks, each closing a hole the others leave
// open -- the layering matters, so read this before adding a fourth.
//
//   1. Energy against first principles. plant.energy() is recomputed here from
//      link geometry alone: every centre of mass located and differentiated
//      directly, no h_ and no D_. This is the ONLY check that validates
//      build_inertia_terms, and without it check 2 is far weaker than it looks --
//      accel() and energy() both read h_ and D_, so a wrong-but-symmetric D_
//      yields a self-consistent system that conserves its own energy perfectly.
//
//   2. Energy conservation under RK4. With zero damping and a stationary carriage,
//      T + V must hold. Given check 1 has pinned h_ and D_, this tests accel()
//      against them: the cos(th_i - th_k) expansion in A, the antisymmetric C, the
//      omega-squared coupling. All three are invisible at N=1, which is the entire
//      reason this file exists alongside physics_validation.
//
//   3. Work-energy on the belt force. Drive the carriage and the plant is no longer
//      conservative -- the belt does work on it at a rate F*xdot. Integrating that
//      must reproduce the energy change. belt_force feeds the step-loss termination
//      and is otherwise entirely untested.
//
// Plus damping monotonicity: joint damping may only remove energy. A sign error in
// the reaction term of damping_torques makes an amplifier, which is a fast way to
// train a policy against a plant that does not exist.
//
// Parameters are deliberately NON-UNIFORM. D_ik = L_min(i,k) * h_max(i,k) is
// asymmetric in the two indices, so with identical links a transposed index reads
// exactly the same and every check here passes on broken code.

#include "core/plant.hpp"

#include "check.hpp"
#include "plot_html.hpp"

#include <cmath>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

constexpr double kDt = 1.0 / 2000.0;
constexpr double kDuration = 2.0;
constexpr int kSteps = static_cast<int>(kDuration / kDt);
constexpr int kMaxPlotPoints = 900;

constexpr double kPi = 3.14159265358979323846;
constexpr double kDrive = 3.0;      // m/s^2 carriage acceleration amplitude
constexpr double kDriveHz = 1.5;
constexpr double kDamping = 2.0e-3; // N m s / rad, for the monotonicity scenario

// Tolerances. The first is at rounding level because both sides are a few dozen
// flops. The conservation bound sits roughly 20x above what the code achieves, so
// it catches a regression without tripping on a compiler changing its mind about
// fma contraction.
constexpr double kGeometryTol = 1e-12;     // relative
constexpr double kConservationTol = 1e-7;  // relative
constexpr double kMonotoneSlack = 1e-9;    // J, per step

// The work-energy residual is limited by the trapezoid quadrature of the power,
// not by the belt force -- it is O(dt^2) where everything else here is O(dt^4), so
// a tight absolute bound would only be measuring the quadrature. This value is a
// coarse backstop against a grossly wrong force; the real check is that the
// residual falls off as dt^2, which a wrong force would not do.
constexpr double kWorkEnergyTol = 1e-4;    // relative

core::PendulumParams non_uniform(int n) {
    // Nothing here is a scaled copy of anything else: distinct masses, distinct
    // lengths, centres of mass that are not at L/2, and inertias that are not the
    // thin-rod value. Any accidental symmetry would hide an index bug.
    const double m[] = {0.12, 0.08, 0.05};
    const double len[] = {0.30, 0.22, 0.16};
    const double com[] = {0.11, 0.09, 0.06};
    const double inertia[] = {1.17e-3, 4.20e-4, 1.39e-4};

    core::PendulumParams p;
    p.carriage_mass = 0.6;
    p.masses.resize(n);
    p.lengths.resize(n);
    p.com_dists.resize(n);
    p.inertias.resize(n);
    p.joint_damping.setZero(n);
    for (int i = 0; i < n; ++i) {
        p.masses[i] = m[i];
        p.lengths[i] = len[i];
        p.com_dists[i] = com[i];
        p.inertias[i] = inertia[i];
    }
    p.validate();
    return p;
}

// Total mechanical energy from geometry alone -- the independent implementation.
// Walks the chain accumulating joint position and joint velocity, locates each
// centre of mass, and sums 1/2 m v^2 + 1/2 I w^2 directly.
double reference_energy(const core::PendulumParams& p, double gravity, const core::Vec& theta,
                        const core::Vec& omega, double xdot) {
    const int n = p.n_links();

    double kinetic = 0.5 * p.carriage_mass * xdot * xdot;
    double potential = 0.0;

    double joint_y = 0.0;    // height of the current joint above the carriage
    double joint_vx = xdot;  // velocity of the current joint
    double joint_vy = 0.0;

    for (int i = 0; i < n; ++i) {
        const double s = std::sin(theta[i]);
        const double c = std::cos(theta[i]);

        const double com_y = joint_y + p.com_dists[i] * c;
        const double com_vx = joint_vx + p.com_dists[i] * c * omega[i];
        const double com_vy = joint_vy - p.com_dists[i] * s * omega[i];

        kinetic += 0.5 * p.masses[i] * (com_vx * com_vx + com_vy * com_vy) +
                   0.5 * p.inertias[i] * omega[i] * omega[i];
        potential += gravity * p.masses[i] * com_y;

        joint_y += p.lengths[i] * c;
        joint_vx += p.lengths[i] * c * omega[i];
        joint_vy -= p.lengths[i] * s * omega[i];
    }
    return kinetic + potential;
}

// Energy scale for relative comparisons: the full swing of potential energy.
double energy_scale(const core::PendulumParams& p, double gravity) {
    double sum = 0.0;
    double beyond = 0.0;
    for (int i = p.n_links() - 1; i >= 0; --i) {
        sum += p.masses[i] * p.com_dists[i] + p.lengths[i] * beyond;
        beyond += p.masses[i];
    }
    return 2.0 * gravity * sum;
}

core::State initial_state(int n) {
    const double theta[] = {0.40, -0.30, 0.60};
    const double omega[] = {0.00, 1.50, -1.00};

    core::State s;
    s.resize(n);
    for (int i = 0; i < n; ++i) {
        s.theta[i] = theta[i];
        s.omega[i] = omega[i];
    }
    return s;
}

// Rate at which the joint dampers remove energy, from the definition rather than
// from the plant: joint i resists rotation relative to link i-1, with the carriage
// non-rotating, so it burns b_i (om_i - om_{i-1})^2. Sign-definite by construction,
// which is exactly what a plant that drops the reaction term is not.
double dissipation_rate(const core::PendulumParams& p, const core::Vec& omega) {
    double rate = 0.0;
    for (int i = 0; i < p.n_links(); ++i) {
        const double relative = (i == 0) ? omega[0] : omega[i] - omega[i - 1];
        rate += p.joint_damping[i] * relative * relative;
    }
    return rate;
}

// With the carriage held still the belt does no work, so the entire energy change
// must be the dampers' doing. Merely checking that energy falls is far too weak:
// dropping the reaction term leaves a dissipation law that is still negative for
// most trajectories, and the test passes on broken code. Comparing against the
// actual quadratic form is what pins it down.
double max_dissipation_residual(const core::PendulumParams& params, int n, double dt, int steps,
                                std::vector<double>* trace = nullptr, int stride = 1,
                                double* worst_rise = nullptr, double* total_loss = nullptr) {
    const core::Plant plant(params);
    core::State state = initial_state(n);

    const double e0 = plant.energy(state.theta, state.omega, state.xdot);
    double previous_energy = e0;
    double dissipated = 0.0;
    double previous_rate = dissipation_rate(params, state.omega);
    double worst = 0.0;

    for (int i = 0; i <= steps; ++i) {
        const double energy = plant.energy(state.theta, state.omega, state.xdot);
        const double residual = (energy - e0) + dissipated;
        worst = std::max(worst, std::abs(residual));
        if (worst_rise != nullptr) {
            *worst_rise = std::max(*worst_rise, energy - previous_energy);
        }
        previous_energy = energy;
        if (trace != nullptr && i % stride == 0) {
            trace->push_back(residual);
        }
        if (i == steps) {
            if (total_loss != nullptr) {
                *total_loss = e0 - energy;
            }
            break;
        }

        plant.step(state, 0.0, dt);
        const double rate = dissipation_rate(params, state.omega);
        dissipated += 0.5 * dt * (previous_rate + rate);
        previous_rate = rate;
    }
    return worst;
}

// Drives the carriage and tracks how far the energy change strays from the work
// the belt reports doing. Optionally records the trace for plotting.
double max_work_residual(const core::Plant& plant, int n, double dt, int steps,
                         std::vector<double>* trace = nullptr, int stride = 1) {
    core::State state = initial_state(n);
    const double e0 = plant.energy(state.theta, state.omega, state.xdot);
    double work = 0.0;
    double worst = 0.0;

    for (int i = 0; i <= steps; ++i) {
        const double residual = (plant.energy(state.theta, state.omega, state.xdot) - e0) - work;
        worst = std::max(worst, std::abs(residual));
        if (trace != nullptr && i % stride == 0) {
            trace->push_back(residual);
        }
        if (i == steps) {
            break;
        }

        // The command is held across the step, so both ends of the trapezoid are
        // evaluated at the same acceleration -- integrating across the zero-order-
        // hold discontinuity instead would leave an O(dt) error that swamps what
        // this check is looking for.
        const double accel_cmd = kDrive * std::sin(2.0 * kPi * kDriveHz * (i * dt));
        const double power_left =
            plant.accel(state.theta, state.omega, accel_cmd).belt_force * state.xdot;
        plant.step(state, accel_cmd, dt);
        const double power_right =
            plant.accel(state.theta, state.omega, accel_cmd).belt_force * state.xdot;
        work += 0.5 * dt * (power_left + power_right);
    }
    return worst;
}

struct Scenario {
    std::vector<double> free_drift;       // E(t) - E(0), undamped, carriage still
    std::vector<double> damping_residual;  // energy change vs the dissipation law
    std::vector<double> work_residual;     // energy change vs belt work, driven

    double max_free_drift = 0.0;
    double max_work_residual = 0.0;
    double max_damping_residual = 0.0;
    double damping_worst_rise = 0.0;  // largest step-over-step energy increase
    double damped_loss = 0.0;
};

}  // namespace

int main(int argc, char** argv) {
    const std::string out_path = argc > 1 ? argv[1] : "energy_conservation.html";

    std::vector<double> time;
    std::vector<Scenario> scenarios(3);
    std::vector<std::string> labels;

    // ---- check 1: h_ and D_ against geometry -------------------------------
    check::section("energy vs geometry");
    {
        std::mt19937 rng(20260811);
        std::uniform_real_distribution<double> angle(-3.2, 3.2);
        std::uniform_real_distribution<double> rate(-9.0, 9.0);
        std::uniform_real_distribution<double> cart(-2.0, 2.0);

        double worst = 0.0;
        for (int n = 1; n <= 3; ++n) {
            const core::PendulumParams params = non_uniform(n);
            const core::Plant plant(params);
            const double scale = energy_scale(params, core::kGravity);

            for (int trial = 0; trial < 400; ++trial) {
                core::Vec theta(n);
                core::Vec omega(n);
                for (int i = 0; i < n; ++i) {
                    theta[i] = angle(rng);
                    omega[i] = rate(rng);
                }
                const double xdot = cart(rng);

                const double got = plant.energy(theta, omega, xdot);
                const double want = reference_energy(params, core::kGravity, theta, omega, xdot);
                // Scale by the magnitude in play: at 9 rad/s the kinetic term dwarfs
                // the potential swing, and dividing by the latter alone would make a
                // rounding-level difference look like a failure.
                const double ref = std::max(scale, std::abs(want));
                worst = std::max(worst, std::abs(got - want) / ref);
            }
        }
        CHECK(worst < kGeometryTol);
        std::cout << std::format("  worst relative error over 1200 random states: {:.3e}\n", worst);
    }

    // ---- checks 2 and 3, per link count ------------------------------------
    const int stride = std::max(1, (kSteps + 1) / kMaxPlotPoints);

    for (int n = 1; n <= 3; ++n) {
        Scenario& sc = scenarios[n - 1];
        labels.push_back(std::format("N = {}", n));

        const core::PendulumParams params = non_uniform(n);
        const core::Plant plant(params);
        const double scale = energy_scale(params, core::kGravity);

        // -- free: conservative, carriage stationary
        {
            core::State state = initial_state(n);
            const double e0 = plant.energy(state.theta, state.omega, state.xdot);
            for (int i = 0; i <= kSteps; ++i) {
                const double drift = plant.energy(state.theta, state.omega, state.xdot) - e0;
                sc.max_free_drift = std::max(sc.max_free_drift, std::abs(drift));
                if (i % stride == 0) {
                    if (n == 1) {
                        time.push_back(i * kDt);
                    }
                    sc.free_drift.push_back(drift);
                }
                if (i < kSteps) {
                    plant.step(state, 0.0, kDt);
                }
            }
        }

        // -- damped: the energy lost must be exactly what the dampers burn
        {
            core::PendulumParams damped_params = params;
            damped_params.joint_damping.setConstant(n, kDamping);
            sc.max_damping_residual =
                max_dissipation_residual(damped_params, n, kDt, kSteps, &sc.damping_residual,
                                         stride, &sc.damping_worst_rise, &sc.damped_loss);
        }

        // -- driven: belt force must account for the energy change
        sc.max_work_residual =
            max_work_residual(plant, n, kDt, kSteps, &sc.work_residual, stride);

        check::section(std::format("N = {}", n));
        CHECK(sc.max_free_drift / scale < kConservationTol);
        CHECK(sc.max_work_residual / scale < kWorkEnergyTol);
        CHECK(sc.max_damping_residual / scale < kWorkEnergyTol);
        CHECK(sc.damping_worst_rise < kMonotoneSlack);
        CHECK(sc.damped_loss > 0.05 * scale);  // damping actually did something

        std::cout << std::format(
            "  N={}: free drift {:.3e} rel, belt-work residual {:.3e} rel, "
            "damping residual {:.3e} rel, damped loss {:.4f} J\n",
            n, sc.max_free_drift / scale, sc.max_work_residual / scale,
            sc.max_damping_residual / scale, sc.damped_loss);
    }

    // ---- integrator order: the residual must be truncation, not a bug ------
    check::section("RK4 convergence order");
    {
        const core::PendulumParams params = non_uniform(3);
        const core::Plant plant(params);

        const auto drift_at = [&](double dt, int steps) {
            core::State state = initial_state(3);
            const double e0 = plant.energy(state.theta, state.omega, state.xdot);
            double worst = 0.0;
            for (int i = 0; i < steps; ++i) {
                plant.step(state, 0.0, dt);
                worst = std::max(worst,
                                 std::abs(plant.energy(state.theta, state.omega, state.xdot) - e0));
            }
            return worst;
        };

        // Same simulated interval, half the step. A fourth-order integrator should
        // cut the error by about 16x; a wrong right-hand side does not improve with
        // step size at anything like that rate.
        const double coarse = drift_at(1.0 / 500.0, 500);
        const double fine = drift_at(1.0 / 1000.0, 1000);
        const double ratio = coarse / fine;
        CHECK(ratio > 8.0);
        std::cout << std::format("  drift ratio on halving dt: {:.1f}x (expect ~16x)\n", ratio);
    }

    check::section("work-energy residual is quadrature, not a wrong belt force");
    {
        // The absolute residual above is dominated by the trapezoid rule, so its
        // size says little. Its *scaling* says everything: quadrature error falls
        // as dt^2, while a belt force that is wrong by some factor or missing a
        // term leaves a residual that barely moves when the step shrinks.
        const core::Plant plant(non_uniform(3));
        const double coarse = max_work_residual(plant, 3, 1.0 / 500.0, 500);
        const double fine = max_work_residual(plant, 3, 1.0 / 1000.0, 1000);
        const double ratio = coarse / fine;
        CHECK(ratio > 3.0);
        std::cout << std::format("  residual ratio on halving dt: {:.1f}x (expect ~4x)\n", ratio);
    }

    // ---- report -------------------------------------------------------------
    std::vector<plot::Chart> charts(3);
    charts[0].title = "Energy drift, free swing";
    charts[0].subtitle =
        "Zero damping, carriage held still. Must stay flat. This is where a sign error in the "
        "cos(th_i - th_k) cross terms shows up -- and it cannot appear at N = 1, where there "
        "are no cross terms to get wrong.";
    charts[0].y_label = "E(t) - E(0)  (J)";

    charts[1].title = "Dissipation residual, damped";
    charts[1].subtitle =
        "Energy change plus the integral of sum b_i (om_i - om_{i-1})^2. Every joint torque enters "
        "two equations with opposite sign, and dropping the reaction term still yields a mostly "
        "decaying energy curve -- so 'energy went down' is not a test. This is.";
    charts[1].y_label = "residual (J)";

    charts[2].title = "Work-energy residual, driven carriage";
    charts[2].subtitle =
        "Energy change minus the integral of belt force times carriage velocity. Flat means the "
        "reported belt force is the one actually doing the work -- the quantity the step-loss "
        "termination is tested against.";
    charts[2].y_label = "residual (J)";

    for (int n = 0; n < 3; ++n) {
        charts[0].series.push_back({labels[n], scenarios[n].free_drift});
        charts[1].series.push_back({labels[n], scenarios[n].damping_residual});
        charts[2].series.push_back({labels[n], scenarios[n].work_residual});
    }

    std::string intro = std::format(
        "<p class=\"lede\">Chain of up to three deliberately dissimilar links -- distinct masses "
        "and lengths, centres of mass away from mid-link, inertias off the thin-rod value -- on a "
        "{:.0f} g carriage. RK4 at {:.0f} Hz for {:.1f} s. Identical links would make "
        "<code>D_ik = L_min * h_max</code> symmetric in its two indices, and a transposed index "
        "would read exactly the same.</p>",
        non_uniform(1).carriage_mass * 1000.0, 1.0 / kDt, kDuration);

    const bool ok = check::failed == 0;
    intro += std::format("<div class=\"verdict {}\"><span class=\"dot\"></span>{}</div>",
                         ok ? "pass" : "fail", ok ? "PASS" : "FAIL");
    intro +=
        "<p class=\"lede\">Before any of the charts below: <code>Plant::energy</code> was checked "
        "against an independent implementation built from link geometry alone, at 1200 random "
        "configurations. That check is what validates <code>h_</code> and <code>D_</code>. "
        "Conservation cannot -- <code>accel()</code> and <code>energy()</code> share those "
        "terms, so a consistently wrong mass matrix conserves its own energy exactly.</p>";

    std::string table =
        "<h2 style=\"margin-top:26px\">Per-link-count summary</h2>"
        "<table><thead><tr><th>Links</th><th>free drift / scale</th>"
        "<th>dissipation residual / scale</th><th>belt-work residual / scale</th>"
        "<th>energy removed by damping (J)</th></tr></thead><tbody>";
    for (int n = 0; n < 3; ++n) {
        const double scale = energy_scale(non_uniform(n + 1), core::kGravity);
        table += std::format(
            "<tr><td>{}</td><td>{:.3e}</td><td>{:.3e}</td><td>{:.3e}</td><td>{:.4f}</td></tr>",
            n + 1, scenarios[n].max_free_drift / scale, scenarios[n].max_damping_residual / scale,
            scenarios[n].max_work_residual / scale, scenarios[n].damped_loss);
    }
    table += "</tbody></table>";

    std::ofstream file(out_path);
    if (!file) {
        std::cerr << "could not open " << out_path << " for writing\n";
        return 2;
    }
    file << plot::page("Plant validation -- multi-link", intro, time, charts, table);
    file.close();

    std::cout << "report: " << out_path << '\n';
    return check::summary("energy_conservation");
}
