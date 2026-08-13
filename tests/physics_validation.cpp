// Physics validation at N=1: five free-swinging runs from different initial
// conditions, checked two independent ways and plotted.
//
//   1. Energy. With zero joint damping and zero carriage acceleration the plant is
//      conservative, so T + V must hold constant for the whole run. This is the
//      real check on the Lagrangian derivation -- far stronger than eyeballing a
//      trajectory, because a wrong h_i or D_ik still produces a plausible-looking
//      swing but leaks or injects energy.
//
//   2. Closed form. A single uniform rod on a carriage obeys
//          alpha = (3 / 2L) (g sin th - a cos th)
//      which is derivable by hand, so it pins down the absolute scale of h_ and D_
//      that check 1 cannot: energy conservation is invariant to a common factor on
//      the inertia terms, and this is not.
//
// N=1 deliberately: it is the case with a hand-checkable answer. It cannot see a
// sign error in the D_ik cross terms, which is why the multi-link energy test is a
// separate and necessary follow-up.

#include "core/plant.hpp"

#include "plot_html.hpp"

#include <cmath>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr double kDt = 1.0 / 500.0;  // the intended control period
constexpr double kDuration = 3.0;    // s
constexpr int kSteps = static_cast<int>(kDuration / kDt);

constexpr double kMass = 0.1;           // kg per link
constexpr double kLength = 0.3;         // m
constexpr double kCarriageMass = 0.5;   // kg
constexpr double kProbeAccel = 3.0;     // m/s^2, for the closed-form check only

// Relative energy drift over the run, against the natural energy scale g*h.
constexpr double kEnergyTol = 1e-6;
// Relative agreement with the hand derivation. Both sides are a handful of flops,
// so anything above rounding noise is a real discrepancy.
constexpr double kClosedFormTol = 1e-12;

constexpr int kMaxPlotPoints = 900;

struct RunSpec {
    const char* label;
    double theta0;  // rad from upright
    double omega0;  // rad/s
};

// Spread across the regimes the controller will actually meet: barely tipped,
// mid-fall, near-horizontal, near-hanging, and one launched hard enough to rotate
// continuously (which is where an integrator sheds energy first).
constexpr RunSpec kRuns[] = {
    {"th0 = 0.05 rad", 0.05, 0.0},
    {"th0 = 0.30 rad", 0.30, 0.0},
    {"th0 = 1.00 rad", 1.00, 0.0},
    {"th0 = 2.50 rad", 2.50, 0.0},
    {"th0 = 0.20, w0 = 8", 0.20, 8.0},
};
constexpr int kRunCount = static_cast<int>(std::size(kRuns));

struct RunResult {
    std::vector<double> theta;
    std::vector<double> omega;
    std::vector<double> drift;      // E(t) - E(0), J
    double energy0 = 0.0;
    double max_drift = 0.0;         // over every step, not just plotted ones
    double max_closed_form = 0.0;   // relative
};

std::string fmt_sci(double v) {
    return std::format("{:.2e}", v);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string out_path = argc > 1 ? argv[1] : "physics_validation.html";

    const core::PendulumParams params =
        core::PendulumParams::uniform_rods(1, kMass, kLength, 0.0, kCarriageMass);
    const core::Plant plant(params);

    // h and D for a uniform rod, recomputed here from the definitions rather than
    // read back from the plant -- a check that shares the code under test proves
    // nothing.
    const double h = kMass * (kLength / 2.0);
    const double energy_scale = core::kGravity * h;

    // Plot every k-th sample; the tolerances are still evaluated on every step.
    const int stride = std::max(1, (kSteps + 1) / kMaxPlotPoints);

    std::vector<double> time;
    std::vector<RunResult> results(kRunCount);

    for (int r = 0; r < kRunCount; ++r) {
        core::State state;
        state.resize(1);
        state.theta[0] = kRuns[r].theta0;
        state.omega[0] = kRuns[r].omega0;

        RunResult& out = results[r];
        out.energy0 = plant.energy(state.theta, state.omega, state.xdot);

        for (int i = 0; i <= kSteps; ++i) {
            const double energy = plant.energy(state.theta, state.omega, state.xdot);
            const double drift = energy - out.energy0;
            out.max_drift = std::max(out.max_drift, std::abs(drift));

            // Closed form, probed at a non-zero carriage acceleration so the
            // -a h cos(th) term is exercised too. This evaluation does not feed
            // back into the trajectory, which stays at a = 0 so energy holds.
            const core::Derivatives d = plant.accel(state.theta, state.omega, kProbeAccel);
            const double analytic = (3.0 / (2.0 * kLength)) *
                                    (core::kGravity * std::sin(state.theta[0]) -
                                     kProbeAccel * std::cos(state.theta[0]));
            const double scale = std::max(std::abs(analytic), 1.0);
            out.max_closed_form =
                std::max(out.max_closed_form, std::abs(d.alpha[0] - analytic) / scale);

            if (i % stride == 0) {
                if (r == 0) {
                    time.push_back(i * kDt);
                }
                out.theta.push_back(state.theta[0]);
                out.omega.push_back(state.omega[0]);
                out.drift.push_back(drift);
            }

            if (i < kSteps) {
                plant.step(state, 0.0, kDt);
            }
        }
    }

    // --- verdict --------------------------------------------------------------
    double worst_energy_rel = 0.0;
    double worst_closed_form = 0.0;
    for (const RunResult& r : results) {
        worst_energy_rel = std::max(worst_energy_rel, r.max_drift / energy_scale);
        worst_closed_form = std::max(worst_closed_form, r.max_closed_form);
    }
    const bool energy_ok = worst_energy_rel < kEnergyTol;
    const bool closed_form_ok = worst_closed_form < kClosedFormTol;
    const bool ok = energy_ok && closed_form_ok;

    std::cout << std::format("energy drift   : {} relative (tol {})  [{}]\n",
                             fmt_sci(worst_energy_rel), fmt_sci(kEnergyTol),
                             energy_ok ? "PASS" : "FAIL");
    std::cout << std::format("closed form    : {} relative (tol {})  [{}]\n",
                             fmt_sci(worst_closed_form), fmt_sci(kClosedFormTol),
                             closed_form_ok ? "PASS" : "FAIL");

    // --- report ---------------------------------------------------------------
    std::vector<plot::Chart> charts(3);
    charts[0].title = "Link angle";
    charts[0].subtitle =
        "Unwrapped, so a run that tips past horizontal and keeps rotating reads as a "
        "ramp rather than a sawtooth. Zero is upright.";
    charts[0].y_label = "theta (rad)";
    charts[1].title = "Angular rate";
    charts[1].subtitle = "Peaks as each run passes through hanging, where all the potential "
                         "energy has become kinetic.";
    charts[1].y_label = "omega (rad/s)";
    charts[2].title = "Energy drift";
    charts[2].subtitle = "T + V measured against its value at t = 0. With zero damping and a "
                         "stationary carriage this must stay flat; the residual is RK4 "
                         "truncation error, not physics.";
    charts[2].y_label = "E(t) - E(0)  (J)";

    for (int r = 0; r < kRunCount; ++r) {
        charts[0].series.push_back({kRuns[r].label, results[r].theta});
        charts[1].series.push_back({kRuns[r].label, results[r].omega});
        charts[2].series.push_back({kRuns[r].label, results[r].drift});
    }

    std::string intro = std::format(
        "<p class=\"lede\">Single uniform rod, {:.0f} g and {:.0f} cm, on a {:.0f} g carriage "
        "held stationary. Free swing, zero joint damping, RK4 at {:.0f} Hz for {:.1f} s. "
        "Each run starts from a different initial condition and is checked two ways.</p>",
        kMass * 1000.0, kLength * 100.0, kCarriageMass * 1000.0, 1.0 / kDt, kDuration);

    intro += std::format(
        "<div class=\"verdict {}\"><span class=\"dot\"></span>{}</div>"
        "<p class=\"lede\">Energy conservation: worst relative drift <code>{}</code> against a "
        "tolerance of <code>{}</code>. Closed form <code>alpha = (3/2L)(g sin th - a cos th)</code>, "
        "probed at a = {:.1f} m/s^2: worst relative error <code>{}</code> against <code>{}</code>.</p>",
        ok ? "pass" : "fail", ok ? "PASS" : "FAIL", fmt_sci(worst_energy_rel),
        fmt_sci(kEnergyTol), kProbeAccel, fmt_sci(worst_closed_form), fmt_sci(kClosedFormTol));

    std::string table =
        "<h2 style=\"margin-top:26px\">Per-run summary</h2>"
        "<p>Maxima are over every one of the "
        + std::to_string(kSteps + 1) +
        " steps, not only the samples drawn above.</p>"
        "<table><thead><tr><th>Run</th><th>theta_0 (rad)</th><th>omega_0 (rad/s)</th>"
        "<th>E(0) (J)</th><th>max |dE| (J)</th><th>max |dE| / gh</th>"
        "<th>closed-form err</th></tr></thead><tbody>";
    for (int r = 0; r < kRunCount; ++r) {
        table += std::format(
            "<tr><td>{}</td><td>{:.2f}</td><td>{:.2f}</td><td>{:.6f}</td><td>{}</td>"
            "<td>{}</td><td>{}</td></tr>",
            kRuns[r].label, kRuns[r].theta0, kRuns[r].omega0, results[r].energy0,
            fmt_sci(results[r].max_drift), fmt_sci(results[r].max_drift / energy_scale),
            fmt_sci(results[r].max_closed_form));
    }
    table += "</tbody></table>";
    table += std::format(
        "<p style=\"margin-top:18px;font-size:0.84rem\">Energy scale gh = {:.4f} J. "
        "N=1 cannot see a sign error in the D_ik cross terms -- the multi-link energy "
        "test is still required.</p>",
        energy_scale);

    std::ofstream file(out_path);
    if (!file) {
        std::cerr << "could not open " << out_path << " for writing\n";
        return 2;
    }
    file << plot::page("Plant validation -- single link", intro, time, charts, table);
    file.close();

    std::cout << "report: " << out_path << '\n';
    return ok ? 0 : 1;
}
