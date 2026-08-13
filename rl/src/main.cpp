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

#include "config/config.hpp"
#include "rl/networks.hpp"
#include "rl/ppo.hpp"
#include "rl/rollout.hpp"
#include "sim/env.hpp"

int main(int argc, char** argv) {
    // Every hardware and run number lives in config.json. Nothing is hard-coded here,
    // and nothing falls back to a struct default -- config::load() requires each key
    // and throws naming the one that is missing. That strictness is the point: the
    // first full run of this project died because plant setup here left
    // StepperLimits::holding_force at its 10 N header default while asking the motor
    // for 30 m/s^2, and no layer was in a position to notice.
    const std::string config_path =
        config::find_file(argc > 1 ? argv[1] : "config.json", argv[0]);

    config::Config cfg;
    try {
        cfg = config::load(config_path);
    } catch (const config::Error& e) {
        std::cerr << "config error: " << e.what() << "\n";
        return 1;
    }
    std::cout << "config: " << config_path << "\n";

    // The number that says the task is solved: mean episode return approaching
    // +1 per plant step, i.e. the pendulum reaching upright early and staying there.
    // Watch the *length* too -- a rising return with short episodes means it is
    // finding a way to end early, not a way to balance.
    const config::TrainingConfig& train_config = cfg.training;
    const sim::EnvConfig& env_config = cfg.env;
    int obs_size = sim::Env::observation_size(env_config.plant.n_links());
    int act_size = sim::Env::kActionSize;

    std::vector<sim::Env> envs;
    envs.reserve(train_config.environments);
    for (int e = 0; e < train_config.environments; ++e) {
        envs.emplace_back(env_config, train_config.seed + e);
        envs.back().reset();
    }
    
    rl::ActorCriticOptions actor_critic_options{obs_size, act_size, cfg.network.hidden,
                                                cfg.network.initial_log_std};
    std::shared_ptr<rl::ActorCritic> net = std::make_shared<rl::ActorCritic>(actor_critic_options);

    // Copied field by field rather than parsed straight into rl::PpoOptions: rl/ links
    // config/, so config/ knowing about rl/ would close a dependency cycle.
    rl::PpoOptions ppo_options;
    ppo_options.learning_rate = cfg.ppo.learning_rate;
    ppo_options.clip = cfg.ppo.clip;
    ppo_options.value_coefficient = cfg.ppo.value_coefficient;
    ppo_options.entropy_coefficient = cfg.ppo.entropy_coefficient;
    ppo_options.max_gradient_norm = cfg.ppo.max_gradient_norm;
    ppo_options.epochs = cfg.ppo.epochs;
    ppo_options.minibatch_size = cfg.ppo.minibatch_size;
    ppo_options.gamma = cfg.ppo.gamma;
    ppo_options.lambda = cfg.ppo.lambda;
    ppo_options.target_kl = cfg.ppo.target_kl;
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
