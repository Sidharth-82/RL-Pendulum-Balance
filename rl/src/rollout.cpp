#include "rl/rollout.hpp"

#include <cassert>

namespace rl {

RolloutBuffer::RolloutBuffer(int capacity, int observation_size, int action_size)
    : capacity_(capacity) {
    assert(capacity > 0 && observation_size > 0 && action_size > 0);

    // Allocated once at full capacity: collection runs capacity times per iteration and
    // must not allocate in that loop. Float32 throughout to match the network -- a
    // float64 stream fails at the first matmul in the update.
    const auto floats = torch::TensorOptions().dtype(torch::kFloat32);
    observations_ = torch::zeros({capacity, observation_size}, floats);
    actions_ = torch::zeros({capacity, action_size}, floats);
    log_probs_ = torch::zeros({capacity}, floats);
    values_ = torch::zeros({capacity}, floats);
    next_values_ = torch::zeros({capacity}, floats);
    rewards_ = torch::zeros({capacity}, floats);
    advantages_ = torch::zeros({capacity}, floats);
    returns_ = torch::zeros({capacity}, floats);

    // Bool rather than float: these are read in a C++ loop in finish(), never in tensor
    // arithmetic, and bool makes the branch there say what it means.
    const auto flags = torch::TensorOptions().dtype(torch::kBool);
    terminated_ = torch::zeros({capacity}, flags);
    truncated_ = torch::zeros({capacity}, flags);
}

void RolloutBuffer::add(const torch::Tensor& observation, const torch::Tensor& action,
                        const torch::Tensor& log_prob, const torch::Tensor& value,
                        double reward, bool terminated, bool truncated,
                        const torch::Tensor& next_value) {
    assert(!full() && "buffer is full; capacity must match the collection loop");

    // copy_, never `=`. observations_[size_] yields a temporary handle, and whether
    // assigning to it rebinds the handle or copies the data depends on LibTorch's
    // ref-qualified operator=. If it rebinds, the buffer stays all zeros and training
    // silently learns nothing. detach() so a caller who forgets NoGradGuard during
    // collection leaks one graph rather than a whole iteration's worth.
    observations_[size_].copy_(observation.detach());
    actions_[size_].copy_(action.detach());
    log_probs_[size_].copy_(log_prob.detach());
    values_[size_].copy_(value.detach());

    // next_value is stored even when terminated. finish() decides whether to use it;
    // keeping that decision in one place is what stops the two flags being conflated.
    next_values_[size_].copy_(next_value.detach());

    rewards_.accessor<float, 1>()[size_] = static_cast<float>(reward);
    terminated_.accessor<bool, 1>()[size_] = terminated;
    truncated_.accessor<bool, 1>()[size_] = truncated;

    ++size_;
}

void RolloutBuffer::finish(double last_value, double gamma, double lambda) {
    // Accessors, not .item<float>(): the loop touches seven streams per step and a
    // tensor round-trip each would dominate the iteration.
    auto reward = rewards_.accessor<float, 1>();
    auto value = values_.accessor<float, 1>();
    auto next_value = next_values_.accessor<float, 1>();
    auto terminated = terminated_.accessor<bool, 1>();
    auto truncated = truncated_.accessor<bool, 1>();
    auto advantage = advantages_.accessor<float, 1>();
    auto discounted_return = returns_.accessor<float, 1>();

    // Backward recursion, so each step's advantage costs O(1) instead of a fresh sum.
    double gae = 0.0;
    for (int t = size_ - 1; t >= 0; --t) {
        // What the future after this transition is worth. The two flags answer
        // different questions and only one of them touches this line.
        double bootstrap = 0.0;
        if (terminated[t]) {
            bootstrap = 0.0;  // the episode really ended; there is no future
        } else if (truncated[t]) {
            bootstrap = next_value[t];  // clock ran out, but that state is worth something
        } else {
            // Mid-episode: V of the next state. Inside the batch that is the value
            // recorded on the following row; past the end it is the caller's tail value.
            bootstrap = (t + 1 < size_) ? static_cast<double>(value[t + 1]) : last_value;
        }

        const double delta = reward[t] + gamma * bootstrap - value[t];

        // Both flags cut the trace: credit must never flow backwards across an episode
        // boundary, whatever kind it was. Only termination zeroed the bootstrap above.
        const bool episode_ended = terminated[t] || truncated[t];
        gae = delta + gamma * lambda * (episode_ended ? 0.0 : gae);

        advantage[t] = static_cast<float>(gae);
        // The critic's target: its own estimate plus the correction just computed.
        discounted_return[t] = static_cast<float>(gae + value[t]);
    }

    finished_ = true;
}

torch::Tensor RolloutBuffer::observations() const {
    // Slice, never the whole tensor. steps_per_iteration / environments is integer
    // division, so an uneven split would otherwise train on fabricated all-zero rows.
    return observations_.slice(0, 0, size_);
}

torch::Tensor RolloutBuffer::actions() const {
    return actions_.slice(0, 0, size_);
}

torch::Tensor RolloutBuffer::log_probs() const {
    return log_probs_.slice(0, 0, size_);
}

torch::Tensor RolloutBuffer::advantages() const {
    // Before finish() these are all zeros, and a PPO update on zero advantages is a
    // perfect no-op: every gradient vanishes, the loss prints as a clean small number,
    // and it reads exactly like a learning-rate problem.
    assert(finished_ && "call finish() before reading advantages");
    return advantages_.slice(0, 0, size_);
}

torch::Tensor RolloutBuffer::returns() const {
    assert(finished_ && "call finish() before reading returns");
    return returns_.slice(0, 0, size_);
}

int RolloutBuffer::size() const {
    return size_;
}

bool RolloutBuffer::full() const {
    return size_ >= capacity_;
}

void RolloutBuffer::clear() {
    size_ = 0;
    finished_ = false;
}

}  // namespace rl
