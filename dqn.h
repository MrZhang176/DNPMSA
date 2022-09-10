#ifndef EXP_DQN_H
#define EXP_DQN_H

#include <random>
#include <iterator>
#include <algorithm>
#include "torch/torch.h"
#include "utils.h"

using Transition = std::tuple<std::vector<state_type>, int64_t, std::vector<state_type>, float, int32_t>;

class Net : public torch::nn::Module
{
public:
    explicit Net(const uint32_t& seq_num);

    torch::Tensor forward(const torch::Tensor& input);

private:
    torch::nn::Linear _input, _l1, _l2, _l3, _l4, _output;
};

class DQN
{
public:
    explicit DQN(const uint32_t& seq_num);

    int64_t select(const std::vector<state_type>& state);
    void update();
    int64_t predict(const std::vector<state_type>& state);
    float predict_q_value(const std::vector<state_type>& state);

    void push(Transition transition);

    void save(const std::string& path);
    void load(const std::string& path);

    void reset();
private:
    void copy_parameters();
    void sample(std::vector<state_type>& state, std::vector<int64_t>& action, std::vector<state_type>& next_state, std::vector<float>& reward, std::vector<int32_t> &done);

    Net _eval_net, _target_net;
    torch::optim::Adam _optimizer;
    torch::Tensor _loss;

    double _cur_epsilon, _delta;
    uint32_t _seq_num, _step_counter, _episode_counter;

    std::default_random_engine _rand;

    std::vector<Transition> _replay_memory;
    uint32_t _replay_memorty_size;

    std::vector<int32_t> _rest_actions, _actions;
};

#endif //EXP_DQN_H