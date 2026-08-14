// Dumps a core::Plant trajectory for an independent engine to reproduce.
//
//   reference_traj [config.json] [--mode free|driven] [--seconds 3] [--out ref.csv]
//
// This exists to make the MuJoCo model checkable. Playback proves the geometry and
// the angle conventions; it cannot touch mass, inertia or damping, because nothing
// in a render reads them. Running the SAME prescribed carriage acceleration through
// both engines and comparing theta(t) is what actually tests those.
//
// Carriage acceleration is prescribed analytically rather than routed through
// Actuator. Plant::accel takes carriage acceleration directly, so this compares the
// two rigid-body formulations and nothing else; Actuator has its own unit tests and
// putting it in the loop here would only add a second place for a mismatch to hide.
//
// The acceleration actually used is written into the CSV rather than described, so
// the comparison cannot drift through two implementations of the same formula.

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "config/config.hpp"
#include "core/plant.hpp"

namespace {

struct Options {
    std::string config = "config.json";
    std::string mode = "driven";
    std::string out = "reference.csv";
    double seconds = 3.0;
    double amplitude = 8.0;   // m/s^2
    double frequency = 1.2;   // Hz
    double theta0 = 2.5;      // rad from upright; away from both equilibria on purpose
};

// Away from 0 and pi deliberately. Both equilibria are degenerate for this
// comparison: at either one sin or cos vanishes and a whole term of the dynamics
// stops contributing, so a sign error there can hide.
double carriage_accel(const Options& opt, double t) {
    if (opt.mode == "free") {
        return 0.0;  // gravity alone; the pendulum dynamics with no forcing
    }
    // COSINE, not sine. A sin(wt) integrates to (A/w)(1 - cos wt), which is
    // non-negative -- a DC velocity of A/w that walks the cart off the rail at about
    // 1 m/s and turns the test into a study of what happens past the end stop.
    // A cos(wt) integrates to (A/w) sin(wt), zero mean, so the cart oscillates in
    // place and the comparison stays about the dynamics.
    return opt.amplitude * std::cos(2.0 * core::kPi * opt.frequency * t);
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() { return (i + 1 < argc) ? argv[++i] : ""; };
        if (arg == "--mode") { opt.mode = next(); }
        else if (arg == "--out") { opt.out = next(); }
        else if (arg == "--seconds") { opt.seconds = std::stod(next()); }
        else if (arg == "--amplitude") { opt.amplitude = std::stod(next()); }
        else if (arg == "--frequency") { opt.frequency = std::stod(next()); }
        else if (arg == "--theta0") { opt.theta0 = std::stod(next()); }
        else if (arg.rfind("--", 0) != 0) { opt.config = arg; }
        else { std::cerr << "unknown option " << arg << "\n"; return 1; }
    }

    config::Config cfg;
    try {
        cfg = config::load(config::find_file(opt.config, argv[0]));
    } catch (const config::Error& e) {
        std::cerr << "config error: " << e.what() << "\n";
        return 1;
    }

    const core::Plant plant(cfg.env.plant, cfg.env.gravity);
    const int n = plant.n_links();
    const double dt = cfg.env.dt;
    const int steps = static_cast<int>(opt.seconds / dt);

    core::State state;
    state.resize(n);
    state.x = 0.0;
    state.xdot = 0.0;
    for (int i = 0; i < n; ++i) {
        // Fan the links out so no two start parallel -- the D_ik cross terms are
        // what a single-link check cannot see, and they vanish when all the angle
        // differences are zero.
        state.theta[i] = opt.theta0 - 0.35 * i;
        state.omega[i] = 0.0;
    }

    std::ofstream out(opt.out);
    if (!out) {
        std::cerr << "cannot write " << opt.out << "\n";
        return 1;
    }
    out << std::setprecision(17);
    out << "t,accel,x,xdot";
    for (int i = 0; i < n; ++i) { out << ",theta" << i; }
    for (int i = 0; i < n; ++i) { out << ",omega" << i; }
    out << ",energy\n";

    for (int k = 0; k <= steps; ++k) {
        const double t = k * dt;
        const double a = carriage_accel(opt, t);

        out << t << ',' << a << ',' << state.x << ',' << state.xdot;
        for (int i = 0; i < n; ++i) { out << ',' << state.theta[i]; }
        for (int i = 0; i < n; ++i) { out << ',' << state.omega[i]; }
        out << ',' << plant.energy(state.theta, state.omega, state.xdot) << '\n';

        if (k < steps) {
            plant.step(state, a, dt);
        }
    }

    std::cout << "wrote " << opt.out << "  (" << steps << " steps, " << n
              << " link" << (n == 1 ? "" : "s") << ", mode " << opt.mode << ")\n";
    return 0;
}
