#include "rl/ppo.hpp"

#include <algorithm>
#include <cstdint>

namespace rl {

Ppo::Ppo(std::shared_ptr<ActorCritic> network, const PpoOptions& options)
    : network_(std::move(network)), options_(options) {
    // parameters() returns only what was registered. If ActorCritic had assigned log_std
    // with a bare `=` instead of register_parameter, it would be missing here and the
    // exploration noise would be frozen for the whole run, with no error anywhere.
    optimizer_ = std::make_unique<torch::optim::Adam>(
        network_->parameters(), torch::optim::AdamOptions(options.learning_rate));
}

PpoStats Ppo::update(const RolloutBuffer& buffer) {
    // These are views into the buffer, taken once. Minibatches are gathered from them by
    // index rather than by re-slicing, so a shuffled minibatch is contiguous in memory.
    const torch::Tensor observations = buffer.observations();
    const torch::Tensor actions = buffer.actions();
    const torch::Tensor old_log_probs = buffer.log_probs();
    const torch::Tensor advantages = buffer.advantages();
    const torch::Tensor returns = buffer.returns();

    const std::int64_t batch = buffer.size();
    const std::int64_t minibatch =
        std::min<std::int64_t>(options_.minibatch_size, batch);

    PpoStats stats;
    int minibatches_seen = 0;

    for (int epoch = 0; epoch < options_.epochs; ++epoch) {
        // A fresh permutation each epoch. Without it every epoch sees the same
        // minibatch partition, and the update correlates with an arbitrary grouping of
        // the data rather than averaging over it.
        const torch::Tensor permutation = torch::randperm(batch, torch::kLong);

        double epoch_kl = 0.0;
        int epoch_minibatches = 0;

        for (std::int64_t start = 0; start < batch; start += minibatch) {
            const std::int64_t stop = std::min(start + minibatch, batch);
            // A single-element minibatch has an undefined sample standard deviation, so
            // the normalisation below would emit NaN and poison every parameter.
            if (stop - start < 2) { continue; }

            const torch::Tensor index = permutation.slice(0, start, stop);
            const torch::Tensor observation_mb = observations.index_select(0, index);
            const torch::Tensor action_mb = actions.index_select(0, index);
            const torch::Tensor old_log_prob_mb = old_log_probs.index_select(0, index);
            const torch::Tensor return_mb = returns.index_select(0, index);

            // Whitened PER MINIBATCH, not once over the batch. Normalising the whole
            // batch leaks information across minibatches and makes the effective step
            // size drift as the batch composition changes.
            torch::Tensor advantage_mb = advantages.index_select(0, index);
            advantage_mb = (advantage_mb - advantage_mb.mean()) /
                           (advantage_mb.std() + 1e-8);

            // The only place a policy gradient comes from. Re-scores the stored actions
            // under the CURRENT parameters; that recomputation is what makes the ratio
            // measure how far the policy has moved.
            const ActorCritic::Evaluation eval =
                network_->evaluate(observation_mb, action_mb);

            // log_ratio kept explicitly: it is exp()'d for the ratio and reused for the
            // KL estimate, where ratio.log() would just undo the exp and lose precision.
            const torch::Tensor log_ratio = eval.log_prob - old_log_prob_mb;
            const torch::Tensor ratio = log_ratio.exp();

            // The clipped surrogate. Taking the MINIMUM of the raw and clipped terms is
            // what makes the bound pessimistic: improvement beyond the trust region is
            // ignored, but the penalty for moving too far is not.
            const torch::Tensor unclipped = ratio * advantage_mb;
            const torch::Tensor clipped =
                ratio.clamp(1.0 - options_.clip, 1.0 + options_.clip) * advantage_mb;
            const torch::Tensor surrogate = torch::min(unclipped, clipped).mean();

            const torch::Tensor value_loss = (eval.value - return_mb).pow(2).mean();
            const torch::Tensor entropy = eval.entropy.mean();

            // The surrogate is MAXIMISED, so it enters the loss negated, as does the
            // entropy bonus. A sign error on either trains a policy that is confidently
            // and increasingly wrong -- which reads as "PPO diverges", not as a typo.
            const torch::Tensor loss = -surrogate +
                                       options_.value_coefficient * value_loss -
                                       options_.entropy_coefficient * entropy;

            optimizer_->zero_grad();
            loss.backward();
            // One bad batch can otherwise produce a step large enough to destroy the
            // policy; this bounds the damage without changing the direction.
            torch::nn::utils::clip_grad_norm_(network_->parameters(),
                                              options_.max_gradient_norm);
            optimizer_->step();

            {
                // Diagnostics only. backward() has already freed the graph, but the
                // tensor VALUES survive, and these were all computed before the step.
                torch::NoGradGuard no_grad;

                // Schulman's low-variance KL estimator: E[r - 1 - log r]. It is
                // non-negative by construction, unlike the naive mean(old - new), which
                // is noisy, can go negative, and makes the early-stop useless.
                const double approx_kl =
                    ((ratio - 1.0) - log_ratio).mean().item<double>();
                const double clip_fraction =
                    ((ratio - 1.0).abs() > options_.clip)
                        .to(torch::kFloat32).mean().item<double>();

                stats.policy_loss += -surrogate.item<double>();
                stats.value_loss += value_loss.item<double>();
                stats.entropy += entropy.item<double>();
                stats.approx_kl += approx_kl;
                stats.clip_fraction += clip_fraction;

                epoch_kl += approx_kl;
            }

            ++epoch_minibatches;
            ++minibatches_seen;
        }

        stats.epochs_run = epoch + 1;

        // Checked at the epoch boundary rather than mid-epoch so every sample in the
        // batch gets used the same number of times. A KL spike almost always means
        // something upstream changed -- reward scale, advantage normalisation -- and
        // stopping here makes that visible instead of letting one update wreck the
        // policy. epochs_run < options_.epochs in the log is the signal.
        if (epoch_minibatches > 0 && epoch_kl / epoch_minibatches > options_.target_kl) {
            break;
        }
    }

    if (minibatches_seen > 0) {
        const double n = static_cast<double>(minibatches_seen);
        stats.policy_loss /= n;
        stats.value_loss /= n;
        stats.entropy /= n;
        stats.approx_kl /= n;
        stats.clip_fraction /= n;
    }
    return stats;
}

}  // namespace rl
