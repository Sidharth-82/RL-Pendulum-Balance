#pragma once

#include <torch/torch.h>

namespace rl {

// Actor-critic for continuous control. Two separate MLP trunks -- shared trunks save
// parameters but couple the policy and value gradients, and for a problem this small
// the parameters are free and the coupling is not.
struct ActorCriticOptions {
    int observation_size = 0;  // sim::Env::observation_size()
    int action_size = 1;       // carriage acceleration, normalised to [-1, 1]
    int hidden = 64;
    double initial_log_std = -0.5;  // std ~ 0.6, wide enough to find a swing-up
};

class ActorCritic : public torch::nn::Module {
public:
    explicit ActorCritic(const ActorCriticOptions& options);

    // Rollout path: sample an action and record what is needed to score it later.
    // Runs under torch::NoGradGuard at the call site -- collection builds no graph.
    struct Sample {
        torch::Tensor action;    // [B, action_size], unclamped Gaussian sample
        torch::Tensor log_prob;  // [B]
        torch::Tensor value;     // [B]
    };
    Sample sample(const torch::Tensor& observation);

    // Update path: re-score stored actions under the current parameters. This is what
    // makes the importance ratio meaningful, so it MUST build a graph -- no NoGradGuard.
    struct Evaluation {
        torch::Tensor log_prob;  // [B]
        torch::Tensor entropy;   // [B]
        torch::Tensor value;     // [B]
    };
    Evaluation evaluate(const torch::Tensor& observation, const torch::Tensor& action);

    // Deterministic action for evaluation episodes and for export. Training samples;
    // reporting should not, or the score mixes policy quality with exploration noise.
    torch::Tensor action_mean(const torch::Tensor& observation);
    torch::Tensor value(const torch::Tensor& observation);

private:
    // log_std is a free parameter, not a network output. State-independent noise is the
    // standard choice for continuous control: it is easier to optimise and it anneals
    // naturally as the policy improves. A state-dependent std tends to collapse early
    // and stop exploring.
    torch::Tensor log_std_;
    torch::nn::Sequential actor_{nullptr};
    torch::nn::Sequential critic_{nullptr};
};

// Orthogonal initialisation with a small gain on the policy's final layer.
//
// Not cosmetic. Default init makes the initial policy's actions large and its value
// estimates arbitrary, which produces enormous early advantages and a first update that
// destroys the policy. Standard practice is gain sqrt(2) on hidden layers, 0.01 on the
// policy head, and 1.0 on the value head.
void orthogonal_init(torch::nn::Sequential& sequential, double final_gain);

}  // namespace rl
