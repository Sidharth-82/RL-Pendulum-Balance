#pragma once

#include <iomanip>
#include <memory>
#include <ostream>
#include <sstream>

#include <torch/torch.h>

#include "rl/networks.hpp"
#include "rl/rollout.hpp"

namespace rl {

struct PpoOptions {
    double learning_rate = 3e-4;
    double clip = 0.2;
    double value_coefficient = 0.5;
    double entropy_coefficient = 0.01;
    double max_gradient_norm = 0.5;

    int epochs = 10;
    int minibatch_size = 256;

    // Discount, in POLICY steps. The env decimates to 100 Hz, so 0.995 looks about
    // 200 steps or 2 s ahead -- long enough to represent a swing-up. At the raw 500 Hz
    // plant rate the same number would cover 0.4 s and the policy could not see far
    // enough to pump energy. Change the decimation and this needs changing with it.
    double gamma = 0.995;
    double lambda = 0.95;

    // Stop an update early if the policy has moved too far. A safety net rather than a
    // tuning knob: a KL spike usually means something upstream is wrong (bad advantage
    // normalisation, a reward scale change), and stopping makes it visible instead of
    // letting one update wreck the policy.
    double target_kl = 0.02;
};

struct PpoStats {
    double policy_loss = 0.0;
    double value_loss = 0.0;
    double entropy = 0.0;
    double approx_kl = 0.0;
    double clip_fraction = 0.0;  // healthy is roughly 0.05-0.2
    int epochs_run = 0;
};

// One fixed-width line per update, so a training log stays scannable by column.
//
// Formatting goes through a local stream rather than `os` directly: std::fixed and
// std::setprecision are sticky, and setting them on the caller's stream would silently
// change how everything printed after this looks.
inline std::ostream& operator<<(std::ostream& os, const PpoStats& stats) {
    std::ostringstream line;
    line << std::fixed
         << "pi_loss "  << std::setw(9) << std::setprecision(5) << stats.policy_loss
         << "  v_loss " << std::setw(9) << std::setprecision(5) << stats.value_loss
         << "  ent "    << std::setw(7) << std::setprecision(4) << stats.entropy
         << "  kl "     << std::setw(8) << std::setprecision(5) << stats.approx_kl
         << "  clip "   << std::setw(6) << std::setprecision(3) << stats.clip_fraction
         << "  epochs " << stats.epochs_run;
    return os << line.str();
}

class Ppo {
public:
    Ppo(std::shared_ptr<ActorCritic> network, const PpoOptions& options);

    // One full update over a finished buffer: `epochs` passes, shuffled minibatches.
    //
    //   ratio     = exp(new_log_prob - old_log_prob)
    //   surrogate = min(ratio * A, clip(ratio, 1-e, 1+e) * A)
    //   loss      = -surrogate + value_coefficient * value_loss
    //               - entropy_coefficient * entropy
    //
    // Two things that are easy to get wrong and cost a lot:
    //   - Normalise advantages PER MINIBATCH, not once over the batch. Whitening the
    //     whole batch leaks information across minibatches and changes the effective
    //     step size as the batch composition shifts.
    //   - The loss MAXIMISES the surrogate, so it is negated. A sign error here trains
    //     a policy that is confidently and increasingly wrong, which reads as "PPO
    //     diverges" rather than as a sign error.
    PpoStats update(const RolloutBuffer& buffer);

    ActorCritic& network() { return *network_; }

private:
    std::shared_ptr<ActorCritic> network_;
    PpoOptions options_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
};

}  // namespace rl
