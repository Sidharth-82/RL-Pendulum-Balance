#include "rl/networks.hpp"

#include <cmath>
#include <cstddef>
#include <vector>

namespace rl {
namespace {

// 0.5 * log(2 * pi) and 0.5 * log(2 * pi * e). Written out rather than computed so the
// Gaussian constants cannot drift between the density and the entropy.
constexpr double kHalfLog2Pi = 0.91893853320467274178;
constexpr double kHalfLog2PiE = 1.41893853320467274178;

// log density of a diagonal Gaussian, summed over the action dimension.
//
// sample() and evaluate() MUST agree exactly -- PPO's ratio is
// exp(new_log_prob - old_log_prob), so any disagreement between the two is read as a
// policy change that never happened. Sharing one implementation makes that structural
// instead of something to verify.
torch::Tensor gaussian_log_prob(const torch::Tensor& action, const torch::Tensor& mean,
                                const torch::Tensor& log_std) {
    const torch::Tensor z = (action - mean) / log_std.exp();
    return (-0.5 * z * z - log_std - kHalfLog2Pi).sum(-1);
}

// Entropy of the same Gaussian, shaped [B] to match log_prob.
//
// log_std is state-independent, so this is the same number for every row. It still
// carries a gradient with respect to log_std, and that gradient is the entire job of
// the entropy bonus: it is what stops sigma collapsing to zero before the policy has
// found a swing-up.
torch::Tensor gaussian_entropy(const torch::Tensor& log_std, std::int64_t batch) {
    return (log_std + kHalfLog2PiE).sum(-1).expand({batch});
}

}  // namespace

ActorCritic::ActorCritic(const ActorCriticOptions& options) {
    actor_ = torch::nn::Sequential(
        torch::nn::Linear(options.observation_size, options.hidden),
        torch::nn::Tanh(),
        torch::nn::Linear(options.hidden, options.hidden),
        torch::nn::Tanh(),
        torch::nn::Linear(options.hidden, options.action_size));

    critic_ = torch::nn::Sequential(
        torch::nn::Linear(options.observation_size, options.hidden),
        torch::nn::Tanh(),
        torch::nn::Linear(options.hidden, options.hidden),
        torch::nn::Tanh(),
        torch::nn::Linear(options.hidden, 1));

    register_module("actor", actor_);
    register_module("critic", critic_);

    log_std_ = register_parameter(
        "log_std",
        torch::full({options.action_size}, options.initial_log_std, torch::kFloat32));

    orthogonal_init(actor_, 0.01);
    orthogonal_init(critic_, 1.0);
}

void orthogonal_init(torch::nn::Sequential& sequential, double final_gain) {
    // Collect the Linears first. The gain depends on which one is LAST, and that is not
    // knowable until the whole container has been walked -- Tanh modules sit between
    // them, so "last child" and "last Linear" are different positions.
    std::vector<torch::nn::LinearImpl*> linears;
    for (const std::shared_ptr<torch::nn::Module>& child : sequential->children()) {
        if (auto* linear = child->as<torch::nn::Linear>()) {
            linears.push_back(linear);
        }
    }

    for (std::size_t i = 0; i < linears.size(); ++i) {
        // sqrt(2) on hidden layers compensates for the signal a tanh removes; the small
        // final_gain on the output layer is what keeps the initial policy near zero.
        const double gain = (i + 1 == linears.size()) ? final_gain : std::sqrt(2.0);
        torch::nn::init::orthogonal_(linears[i]->weight, gain);
        if (linears[i]->bias.defined()) {
            torch::nn::init::zeros_(linears[i]->bias);
        }
    }
}

ActorCritic::Sample ActorCritic::sample(const torch::Tensor& observation) {
    TORCH_CHECK(observation.dim() == 2, "sample expects [B, obs], got ",
                observation.sizes());

    const torch::Tensor mean = actor_->forward(observation);
    const torch::Tensor std = log_std_.exp();

    Sample result;
    result.action = mean + std * torch::randn_like(mean);
    result.log_prob = gaussian_log_prob(result.action, mean, log_std_);
    result.value = value(observation);
    return result;
}

ActorCritic::Evaluation ActorCritic::evaluate(const torch::Tensor& observation,
                                              const torch::Tensor& action) {
    TORCH_CHECK(observation.dim() == 2, "evaluate expects [B, obs], got ",
                observation.sizes());

    const torch::Tensor mean = actor_->forward(observation);

    Evaluation result;
    result.log_prob = gaussian_log_prob(action, mean, log_std_);
    result.entropy = gaussian_entropy(log_std_, observation.size(0));
    result.value = value(observation);
    return result;
}

torch::Tensor ActorCritic::action_mean(const torch::Tensor& observation) {
    return actor_->forward(observation);
}

torch::Tensor ActorCritic::value(const torch::Tensor& observation) {
    return critic_->forward(observation).squeeze(-1);
}

}  // namespace rl
