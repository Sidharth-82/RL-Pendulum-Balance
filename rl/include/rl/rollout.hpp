#pragma once

#include <vector>

#include <torch/torch.h>

namespace rl {

// Fixed-capacity store for one batch of experience, plus the GAE computation that turns
// it into advantages and value targets.
//
// Kept separate from the PPO update on purpose: everything here is bookkeeping with an
// exactly checkable answer, and it is where the subtle bugs live. GAE on a hand-built
// three-step episode can be verified with a calculator; a wrong advantage inside a
// training loop just looks like "PPO does not learn".
class RolloutBuffer {
public:
    RolloutBuffer(int capacity, int observation_size, int action_size);

    // One environment transition. `value` and `log_prob` are the values recorded at
    // collection time, under the parameters that actually chose the action.
    //
    // terminated and truncated are NOT interchangeable, and this is the single most
    // common silent PPO bug:
    //   terminated -- the episode really ended (off rail, step loss). There is no
    //                 future, so the value target is just the reward.
    //   truncated  -- the clock ran out while the state was still fine. The future
    //                 exists and is worth V(s_next); failing to bootstrap here teaches
    //                 the policy that time running out is a catastrophe.
    void add(const torch::Tensor& observation, const torch::Tensor& action,
             const torch::Tensor& log_prob, const torch::Tensor& value, double reward,
             bool terminated, bool truncated, const torch::Tensor& next_value);

    // Backward GAE recursion:
    //   delta_t = r_t + gamma * V(s_{t+1}) * (1 - terminated_t) - V(s_t)
    //   A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    // where done_t means terminated OR truncated -- the trace must not run across an
    // episode boundary of either kind, even though only termination zeroes the
    // bootstrap.
    //
    // `last_value` is V of the state after the final stored transition, used when the
    // batch is cut mid-episode.
    void finish(double last_value, double gamma, double lambda);

    // Flat tensors over the whole batch, for minibatching. Call after finish().
    torch::Tensor observations() const;
    torch::Tensor actions() const;
    torch::Tensor log_probs() const;
    torch::Tensor advantages() const;
    torch::Tensor returns() const;  // advantages + values, the value-function target

    int size() const;
    bool full() const;
    void clear();

private:
    int capacity_ = 0;
    int size_ = 0;

    torch::Tensor observations_;
    torch::Tensor actions_;
    torch::Tensor log_probs_;
    torch::Tensor values_;
    torch::Tensor next_values_;
    torch::Tensor rewards_;
    torch::Tensor terminated_;
    torch::Tensor truncated_;
    torch::Tensor advantages_;
    torch::Tensor returns_;
    bool finished_ = false;
};

}  // namespace rl
