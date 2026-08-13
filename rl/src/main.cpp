// Training entry point: collect, update, report, repeat.
//
// Build this in the order that lets you check each piece before the next depends on it:
//
//   1. Env plus a RANDOM policy. No network, no PPO. Print the mean episode return and
//      length. On this task a random policy mostly hangs, so expect a return near
//      -1 per step and episodes that either run the clock out or end at the rail. That
//      number is the floor everything else is measured against, and it takes ten
//      minutes to get.
//   2. Add ActorCritic and run it untrained. The return should be similar to random.
//      If it is much worse, the policy head initialisation is wrong -- see the note on
//      the 0.01 final gain in networks.cpp.
//   3. Add RolloutBuffer and verify GAE by hand on a short episode before wiring PPO.
//   4. Add Ppo::update and watch clip_fraction, approx_kl and entropy, not just return.
//      Those move first and tell you which knob is wrong.

#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "rl/networks.hpp"
#include "rl/ppo.hpp"
#include "rl/rollout.hpp"
#include "sim/env.hpp"

namespace {

struct TrainConfig {
    int iterations = 2000;
    int steps_per_iteration = 4096;  // policy steps collected per update

    // Must stay 1 until RolloutBuffer handles more than one stream. Collection
    // interleaves envs round-robin, so with E > 1 row t+1 belongs to a DIFFERENT env:
    // finish() would bootstrap each transition off an unrelated state and run the GAE
    // trace across all E episodes at once. Its scalar last_value cannot describe E
    // tails either, so .item<double>() throws -- which is the intended failure. Raising
    // this means per-env buffers, finished separately and concatenated for the update.
    int environments = 1;
    std::uint64_t seed = 1;

    int evaluate_every = 20;    // iterations
    int evaluation_episodes = 5;
    const char* checkpoint_path = "policy.pt";

    // Diagnostics. train_log.csv gets one row per iteration; the trajectory files get a
    // full deterministic episode so the motion can be plotted, not just the score.
    const char* log_path = "train_log.csv";
    const char* trajectory_prefix = "trajectory_";
};

sim::EnvConfig plant_config() {
    // Matches the sizing in the hardware notes: 0.25 m / 0.12 kg link on a 0.70 kg
    // cart, 2 m rail, 30 m/s^2. Start at N=1 -- get the whole pipeline working before
    // adding links, because the basin shrinks sharply with each one.
    sim::EnvConfig config;
    config.plant = core::PendulumParams::uniform_rods(1, 0.12, 0.25, 0.0, 0.70);
    config.limits.max_accel = 30.0;
    config.limits.max_velocity = 3.0;
    config.limits.rail_length = 2.0;
    config.limits.steps_per_metre = 4000.0;

    // The force envelope of the 400 W servo from the hardware sizing, NOT the
    // StepperLimits defaults. Those default to holding_force = 10 N, and 30 m/s^2 on a
    // 0.82 kg carriage needs about 25 N -- so every episode tripped would_lose_steps()
    // on its first plant step and ended with the -100 penalty after ~2 policy steps.
    // The droop above the knee is kept, so the envelope still binds near top speed and
    // the policy cannot treat the motor as infinitely strong.
    config.limits.holding_force = 80.0;
    config.limits.force_knee_speed = 2.5;
    config.episode_seconds = 8.0;
    return config;
}

}  // namespace

int main() {
    // The number that says the task is solved: mean episode return approaching
    // +1 per plant step, i.e. the pendulum reaching upright early and staying there.
    // Watch the *length* too -- a rising return with short episodes means it is
    // finding a way to end early, not a way to balance.
    TrainConfig train_config; 
    sim::EnvConfig env_config = plant_config();
    int obs_size = sim::Env::observation_size(env_config.plant.n_links());
    int act_size = sim::Env::kActionSize;

    std::vector<sim::Env> envs;
    envs.reserve(train_config.environments);
    for (int e = 0; e < train_config.environments; ++e) {
        envs.emplace_back(env_config, train_config.seed + e);
        envs.back().reset();
    }
    
    rl::ActorCriticOptions actor_critic_options{obs_size, act_size, 64, -0.5};
    std::shared_ptr<rl::ActorCritic> net = std::make_shared<rl::ActorCritic>(actor_critic_options);
    
    rl::PpoOptions ppo_options;
    rl::Ppo ppo = rl::Ppo(net, ppo_options);

    rl::RolloutBuffer buffer(train_config.steps_per_iteration, obs_size, act_size);

    std::vector<double> ep_return(train_config.environments, 0.0);
    std::vector<int> ep_length(train_config.environments, 0);
    std::vector<std::tuple<double, int>> recent;
    recent.reserve(20);

    // One row per iteration, for the learning curves. Written incrementally and flushed
    // so a run that is still going -- or one that dies -- still has a readable log.
    std::ofstream log(train_config.log_path);
    log << "iteration,mean_return,mean_length,policy_loss,value_loss,entropy,"
           "approx_kl,clip_fraction,epochs_run,eval_return,eval_length\n";


    for(int i = 0; i < train_config.iterations; ++i){
        buffer.clear();
        
        torch::Tensor obs = torch::zeros({train_config.environments, obs_size}, torch::kFloat32);
        torch::Tensor next_obs = torch::zeros({train_config.environments, obs_size}, torch::kFloat32);

        for(int tick = 0; tick < train_config.steps_per_iteration/train_config.environments; ++tick){
            
            auto acc = obs.accessor<float, 2>();
            for (int e = 0; e < train_config.environments; ++e) {
                const auto& o = envs[e].observation();
                for (int d = 0; d < obs_size; ++d) { acc[e][d] = static_cast<float>(o[d]); }
            }

            rl::ActorCritic::Sample s;
            { //Create scope region
                torch::NoGradGuard no_grad; // Enable C++ RAII guard to disable gradients globally in this scope
                s = net->sample(obs);
            }

            std::vector<double> a(train_config.environments);
            std::vector<sim::StepResult> res(train_config.environments);
            

            for (int e = 0; e < train_config.environments; ++e) {
                a[e] = s.action[e][0].item<double>();
                res[e] = envs[e].step(a[e]);
                ep_return[e] += res[e].reward;
                ep_length[e] += 1;
            }

            auto next_acc = next_obs.accessor<float, 2>();
            for (int e = 0; e < train_config.environments; ++e) {
                const auto& o = envs[e].observation();
                for (int d = 0; d < obs_size; ++d) { next_acc[e][d] = static_cast<float>(o[d]); }
            }

            torch::Tensor v;
            { //Create scope region
                torch::NoGradGuard no_grad; // Enable C++ RAII guard to disable gradients globally in this scope
                v = net->value(next_obs);
            }

            for (int e = 0; e < train_config.environments; ++e) {
                buffer.add(obs[e], s.action[e], s.log_prob[e], s.value[e],
                   res[e].reward, res[e].done, res[e].truncated, v[e]);

                if (res[e].done || res[e].truncated){
                    if(recent.size() == 20){
                        recent.erase(recent.begin());
                    }
                    recent.emplace_back(ep_return[e], ep_length[e]);
                    ep_return[e] = 0.0; 
                    ep_length[e] = 0;
                    envs[e].reset();
                }
            }
        }

        // next_obs still holds the post-step, PRE-reset state from the final tick, which
        // is what finish() needs. Do not re-read observation() here: any env that ended
        // on that tick has already been reset to a fresh episode, and reading it back
        // would bootstrap the tail off the wrong state.
        torch::Tensor last_v;
        { //Create scope region
            torch::NoGradGuard no_grad; // Enable C++ RAII guard to disable gradients globally in this scope
            last_v = net->value(next_obs);
        }

        buffer.finish(last_v.item<double>(), ppo_options.gamma, ppo_options.lambda);
        auto stats = ppo.update(buffer);

        double sum = 0.0;
        double len_sum = 0.0;
        for (const auto& item : recent) {
            sum += std::get<0>(item);
            len_sum += std::get<1>(item);
        }
        // Empty until the first episode finishes -- 800 policy steps, so this is normal
        // for the first iteration or two. Dividing by zero here would print -nan.
        const double mean_return = recent.empty() ? 0.0 : sum / recent.size();
        const double mean_length = recent.empty() ? 0.0 : len_sum / recent.size();

        std::cout << std::setw(5) << i
                  << "  ret " << std::setw(10) << mean_return
                  << "  len " << std::setw(7) << mean_length
                  << "  " << stats << std::endl;


        double eval_return = 0.0;
        int eval_length = 0;
        const bool evaluated = (i % train_config.evaluate_every == 0);

        if (evaluated) {
        // Fresh env, same seed every time: all evaluations run the same start
        // perturbations, so the numbers are comparable across iterations. The seed is
        // disjoint from the training envs so the score is not measured on states the
        // policy has been training against.
        sim::Env eval_env(env_config, train_config.seed + 10000);
        torch::Tensor eval_obs = torch::zeros({1, obs_size}, torch::kFloat32);
        auto eval_acc = eval_obs.accessor<float, 2>();

        for (int e = 0; e < train_config.evaluation_episodes; ++e) {
            eval_env.reset();

            // Episode 0 is dumped in full so the motion can be plotted, not just scored.
            std::ofstream trajectory;
            if (e == 0) {
                trajectory.open(std::string(train_config.trajectory_prefix) +
                                std::to_string(i) + ".csv");
                trajectory << "t,x,xdot,theta,omega,action\n";
            }

            sim::StepResult r;
            while (!r.done && !r.truncated) {
                const auto& o = eval_env.observation();
                for (int d = 0; d < obs_size; ++d) { eval_acc[0][d] = static_cast<float>(o[d]); }

                torch::Tensor a;
                { //Create scope region
                    torch::NoGradGuard no_grad; // Enable C++ RAII guard to disable gradients globally in this scope
                    // action_mean, not sample -- a sampled score improves on its own as
                    // log_std anneals, whether or not the policy actually got better.
                    a = net->action_mean(eval_obs);
                }
                const double action = a[0][0].item<double>();

                if (trajectory.is_open()) {
                    // Truth, not the observation: the plot should show what the pendulum
                    // did, not what the policy was told about it.
                    const core::State& s = eval_env.truth();
                    trajectory << eval_env.elapsed() << ',' << s.x << ',' << s.xdot << ','
                               << core::wrap_angle(s.theta[0]) << ',' << s.omega[0] << ','
                               << action << '\n';
                }

                r = eval_env.step(action);
                eval_return += r.reward;
                ++eval_length;
            }
        }
        }

        const double eval_mean_return = eval_return / train_config.evaluation_episodes;
        const double eval_mean_length =
            static_cast<double>(eval_length) / train_config.evaluation_episodes;

        if (evaluated) {
            std::cout << "      eval ret " << eval_mean_return
                      << "  len " << eval_mean_length << std::endl;
            torch::save(net, train_config.checkpoint_path);
        }

        // Blank rather than 0 on non-evaluation iterations: a zero would plot as a real
        // measurement and drag the eval curve toward the middle of the chart.
        log << i << ',' << mean_return << ',' << mean_length << ','
            << stats.policy_loss << ',' << stats.value_loss << ',' << stats.entropy << ','
            << stats.approx_kl << ',' << stats.clip_fraction << ',' << stats.epochs_run
            << ',';
        if (evaluated) { log << eval_mean_return << ',' << eval_mean_length; }
        else { log << ','; }
        log << '\n';
        log.flush();
    }

    return 0;
}
