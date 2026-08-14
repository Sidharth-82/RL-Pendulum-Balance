#include "config/config.hpp"

#include <filesystem>
#include <fstream>

#include <nlohmann/json.hpp>

// nlohmann/json is included here and nowhere else on purpose. It is ~25k lines of
// templates, and putting it in config.hpp would push that cost onto every
// translation unit that only wants to read a struct field.

namespace config {
namespace {

using nlohmann::json;

// A cursor carries the dotted path it reached, so an error names the key rather
// than the line. "config.json: missing key 'actuator.holding_force'" is actionable
// on its own; "parse error at line 24" means opening the file to find out what is
// on line 24.
class Cursor {
public:
    Cursor(const json& node, std::string path) : node_(&node), path_(std::move(path)) {}

    Cursor at(const char* key) const {
        if (!node_->is_object()) {
            throw Error(where() + " is not an object");
        }
        const auto it = node_->find(key);
        if (it == node_->end()) {
            throw Error("missing key '" + join(key) + "'");
        }
        return Cursor(*it, join(key));
    }

    double number() const {
        if (!node_->is_number()) {
            throw Error(where() + " must be a number");
        }
        return node_->get<double>();
    }

    int integer() const {
        if (!node_->is_number_integer()) {
            throw Error(where() + " must be an integer");
        }
        return node_->get<int>();
    }

    std::uint64_t unsigned_integer() const {
        if (!node_->is_number_unsigned()) {
            throw Error(where() + " must be a non-negative integer");
        }
        return node_->get<std::uint64_t>();
    }

    bool boolean() const {
        if (!node_->is_boolean()) {
            throw Error(where() + " must be true or false");
        }
        return node_->get<bool>();
    }

    std::string text() const {
        if (!node_->is_string()) {
            throw Error(where() + " must be a string");
        }
        return node_->get<std::string>();
    }

    std::vector<Cursor> array() const {
        if (!node_->is_array()) {
            throw Error(where() + " must be an array");
        }
        std::vector<Cursor> out;
        out.reserve(node_->size());
        for (std::size_t i = 0; i < node_->size(); ++i) {
            out.emplace_back((*node_)[i], path_ + "[" + std::to_string(i) + "]");
        }
        return out;
    }

    // A fixed-length vector of doubles, checked against the link count so a
    // three-link theta_ref against a one-link plant fails here rather than as a
    // silent size mismatch inside Env.
    core::Vec vector_of(int expected) const {
        const std::vector<Cursor> items = array();
        if (static_cast<int>(items.size()) != expected) {
            throw Error(where() + " must have " + std::to_string(expected) +
                        " entries, one per link, but has " +
                        std::to_string(items.size()));
        }
        core::Vec out(expected);
        for (int i = 0; i < expected; ++i) {
            out[i] = items[static_cast<std::size_t>(i)].number();
        }
        return out;
    }

    const std::string& path() const { return path_; }

private:
    std::string join(const char* key) const {
        return path_.empty() ? std::string(key) : path_ + "." + key;
    }
    std::string where() const { return "'" + path_ + "'"; }

    const json* node_;
    std::string path_;
};

}  // namespace

Config load(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        throw Error("cannot open config file '" + path + "'");
    }

    json root;
    try {
        root = json::parse(file, /*cb=*/nullptr, /*allow_exceptions=*/true,
                           /*ignore_comments=*/true);
    } catch (const json::parse_error& e) {
        throw Error("in '" + path + "': " + e.what());
    }

    try {
        const Cursor top(root, "");
        Config cfg;

        // --- plant ------------------------------------------------------------
        const Cursor plant = top.at("plant");
        const std::vector<Cursor> links = plant.at("links").array();
        if (links.empty() || static_cast<int>(links.size()) > core::kMaxLinks) {
            throw Error("'plant.links' must hold between 1 and " +
                        std::to_string(core::kMaxLinks) + " links, has " +
                        std::to_string(links.size()));
        }
        const int n = static_cast<int>(links.size());

        core::PendulumParams& p = cfg.env.plant;
        p.carriage_mass = plant.at("carriage_mass").number();
        p.masses.resize(n);
        p.lengths.resize(n);
        p.com_dists.resize(n);
        p.inertias.resize(n);
        p.joint_damping.resize(n);
        for (int i = 0; i < n; ++i) {
            const Cursor& link = links[static_cast<std::size_t>(i)];
            p.masses[i] = link.at("mass").number();
            p.lengths[i] = link.at("length").number();
            p.com_dists[i] = link.at("com_dist").number();
            p.inertias[i] = link.at("inertia").number();
            p.joint_damping[i] = link.at("joint_damping").number();
        }
        // Catches a non-positive mass or a length/inertia that cannot belong to a
        // real body, here, rather than as NaNs several seconds into a rollout.
        p.validate();

        cfg.env.gravity = plant.at("gravity").number();

        // --- actuator ---------------------------------------------------------
        const Cursor actuator = top.at("actuator");
        core::StepperLimits& limits = cfg.env.limits;
        limits.max_velocity = actuator.at("max_velocity").number();
        limits.max_accel = actuator.at("max_accel").number();
        limits.steps_per_metre = actuator.at("steps_per_metre").number();
        limits.holding_force = actuator.at("holding_force").number();
        limits.force_knee_speed = actuator.at("force_knee_speed").number();
        limits.rail_length = actuator.at("rail_length").number();

        // --- encoder ----------------------------------------------------------
        const Cursor encoder = top.at("encoder");
        core::EncoderSpec& enc = cfg.env.encoder;
        enc.bits = encoder.at("bits").integer();
        enc.noise_std = encoder.at("noise_std").number();
        enc.latency_steps = encoder.at("latency_steps").integer();
        enc.derivative_cutoff_hz = encoder.at("derivative_cutoff_hz").number();

        // --- episode ----------------------------------------------------------
        const Cursor env = top.at("env");
        cfg.env.dt = env.at("dt").number();
        cfg.env.episode_seconds = env.at("episode_seconds").number();
        cfg.env.control_decimation = env.at("control_decimation").integer();
        cfg.env.theta_ref = env.at("theta_ref").vector_of(n);
        cfg.env.theta_start = env.at("theta_start").vector_of(n);
        cfg.env.theta_start_spread = env.at("theta_start_spread").number();
        cfg.env.omega_start_spread = env.at("omega_start_spread").number();
        cfg.env.x_start_spread = env.at("x_start_spread").number();
        cfg.env.use_sensor_model = env.at("use_sensor_model").boolean();
        cfg.env.terminate_on_step_loss = env.at("terminate_on_step_loss").boolean();
        cfg.env.position_weight = env.at("position_weight").number();
        cfg.env.action_weight = env.at("action_weight").number();
        cfg.env.rail_penalty = env.at("rail_penalty").number();
        cfg.env.omega_scale = env.at("omega_scale").number();

        // --- network ----------------------------------------------------------
        const Cursor net = top.at("network");
        cfg.network.hidden = net.at("hidden").integer();
        cfg.network.initial_log_std = net.at("initial_log_std").number();

        // --- ppo --------------------------------------------------------------
        const Cursor ppo = top.at("ppo");
        cfg.ppo.learning_rate = ppo.at("learning_rate").number();
        cfg.ppo.clip = ppo.at("clip").number();
        cfg.ppo.value_coefficient = ppo.at("value_coefficient").number();
        cfg.ppo.entropy_coefficient = ppo.at("entropy_coefficient").number();
        cfg.ppo.max_gradient_norm = ppo.at("max_gradient_norm").number();
        cfg.ppo.epochs = ppo.at("epochs").integer();
        cfg.ppo.minibatch_size = ppo.at("minibatch_size").integer();
        cfg.ppo.gamma = ppo.at("gamma").number();
        cfg.ppo.lambda = ppo.at("lambda").number();
        cfg.ppo.target_kl = ppo.at("target_kl").number();

        // --- training ---------------------------------------------------------
        const Cursor training = top.at("training");
        cfg.training.iterations = training.at("iterations").integer();
        cfg.training.steps_per_iteration = training.at("steps_per_iteration").integer();
        cfg.training.environments = training.at("environments").integer();
        cfg.training.seed = training.at("seed").unsigned_integer();
        cfg.training.evaluate_every = training.at("evaluate_every").integer();
        cfg.training.evaluation_episodes = training.at("evaluation_episodes").integer();
        cfg.training.checkpoint_path = training.at("checkpoint_path").text();
        cfg.training.log_path = training.at("log_path").text();
        cfg.training.trajectory_prefix = training.at("trajectory_prefix").text();
        cfg.training.weights_path = training.at("weights_path").text();

        // --- cross-field checks ----------------------------------------------
        // Individually valid numbers that cannot be true together. Each of these
        // would otherwise surface as an assert deep inside Env or as a training
        // run that behaves nothing like the config appears to describe.
        if (cfg.env.dt <= 0.0) {
            throw Error("'env.dt' must be positive");
        }
        if (cfg.env.control_decimation < 1) {
            throw Error("'env.control_decimation' must be at least 1");
        }
        if (cfg.env.episode_seconds <= 0.0) {
            throw Error("'env.episode_seconds' must be positive");
        }
        if (cfg.training.steps_per_iteration % cfg.training.environments != 0) {
            throw Error("'training.steps_per_iteration' (" +
                        std::to_string(cfg.training.steps_per_iteration) +
                        ") must divide evenly by 'training.environments' (" +
                        std::to_string(cfg.training.environments) +
                        "), or the rollout buffer trains on unfilled rows");
        }
        // The rollout buffer walks one contiguous episode stream, so row t+1 has to
        // be the state that followed row t. See the note in rl/src/main.cpp.
        if (cfg.training.environments != 1) {
            throw Error("'training.environments' must be 1 until RolloutBuffer "
                        "handles more than one interleaved stream");
        }
        if (limits.force_knee_speed > limits.max_velocity) {
            throw Error("'actuator.force_knee_speed' exceeds 'actuator.max_velocity', "
                        "which leaves no droop region and reports zero available force");
        }
        // The failure that motivated this whole file: a force envelope that cannot
        // produce the acceleration the same file asks for. Reject it at load rather
        // than discovering it as episodes that end after two steps.
        const double moving_mass = p.carriage_mass + p.masses.sum();
        const double needed = moving_mass * limits.max_accel;
        if (limits.holding_force < needed) {
            throw Error("'actuator.holding_force' is " +
                        std::to_string(limits.holding_force) + " N but accelerating " +
                        std::to_string(moving_mass) + " kg at " +
                        std::to_string(limits.max_accel) + " m/s^2 needs at least " +
                        std::to_string(needed) +
                        " N; every episode would trip step loss immediately");
        }
        return cfg;
    } catch (const Error& e) {
        throw Error("in '" + path + "': " + e.what());
    }
}

std::string find_file(const std::string& path, const char* argv0) {
    namespace fs = std::filesystem;
    if (fs::exists(path)) {
        return path;
    }
    std::error_code ec;
    fs::path here = fs::absolute(fs::path(argv0 ? argv0 : ""), ec).parent_path();
    for (int depth = 0; depth < 6 && !here.empty(); ++depth) {
        const fs::path candidate = here / path;
        if (fs::exists(candidate)) {
            return candidate.string();
        }
        if (!here.has_parent_path() || here.parent_path() == here) {
            break;
        }
        here = here.parent_path();
    }
    return path;  // let load() produce the "cannot open" error with the original name
}

}  // namespace config
