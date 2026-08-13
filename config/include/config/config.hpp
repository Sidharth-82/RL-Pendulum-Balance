#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "sim/env.hpp"

// Loads the whole run description from a JSON file.
//
// The point is not convenience, it is that there is exactly ONE place a hardware
// number lives. The first PPO run of this project died in 2 policy steps because
// rl/main.cpp set max_accel but silently inherited StepperLimits::holding_force = 10 N
// from the header default -- a motor a third as strong as the one the hardware sizing
// had specified. Nothing reported it; every episode just ended instantly.
//
// So this loader is STRICT: every key is required, and a missing or mistyped one
// throws naming its full path. There are no defaults here, because a default here
// would be a second source of truth and would re-create exactly that failure.
//
// Desktop only, like sim/. The Pi gets the trained network and a populated struct,
// not a JSON parser.

namespace config {

struct Error : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Deliberately no default member initialisers anywhere below: load() fills every
// field, and a plausible-looking default would hide a key that failed to load.

struct NetworkConfig {
    int hidden;
    double initial_log_std;
};

// Mirrors rl::PpoOptions rather than including it. rl/ links config/, so config/
// including rl/ would be a cycle; the copy at the call site is three lines and
// keeps the dependency arrow pointing one way.
struct PpoConfig {
    double learning_rate;
    double clip;
    double value_coefficient;
    double entropy_coefficient;
    double max_gradient_norm;
    int epochs;
    int minibatch_size;
    double gamma;
    double lambda;
    double target_kl;
};

struct TrainingConfig {
    int iterations;
    int steps_per_iteration;
    int environments;
    std::uint64_t seed;
    int evaluate_every;
    int evaluation_episodes;
    std::string checkpoint_path;
    std::string log_path;
    std::string trajectory_prefix;
};

struct Config {
    // Fully populated, including plant, limits and encoder.
    sim::EnvConfig env;
    NetworkConfig network;
    PpoConfig ppo;
    TrainingConfig training;
};

// Throws config::Error on a missing file, malformed JSON, a missing key, a wrong
// type, or a plant description that fails core::PendulumParams::validate().
Config load(const std::string& path);

// Resolves `path` if it exists, otherwise walks up from the executable's directory
// looking for it. Training runs happen in an output directory, not the source tree,
// and requiring a correct relative path from wherever the binary was launched is a
// worse failure than searching a few parents.
std::string find_file(const std::string& path, const char* argv0);

}  // namespace config
