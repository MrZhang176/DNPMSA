#include "dqn.h"
#include <iomanip>

Net::Net(const uint32_t& seq_num):  
        _input(register_module("input", torch::nn::Linear(seq_num, 32))),
        _l1(register_module("l1", torch::nn::Linear(32, 64))),
        _l2(register_module("l2", torch::nn::Linear(64, 128))),
        _l3(register_module("l3", torch::nn::Linear(128, 64))),
        _l4(register_module("l4", torch::nn::Linear(64, 32))),
        _output(register_module("output", torch::nn::Linear(32, seq_num)))
{

}

torch::Tensor Net::forward(const torch::Tensor& input)
{
    auto res = torch::relu(_input(input));
    res = torch::relu(_l1(res));
    // res = torch::relu(_l2(res));
    // res = torch::relu(_l3(res));
    res = torch::relu(_l4(res));
    res = torch::tanh(_output(res));

    return res;
}

DQN::DQN(const uint32_t & seq_num) :
        _eval_net(seq_num),
        _target_net(seq_num),
        _optimizer(_eval_net.parameters(), config::alpha),
        _loss(),
        _cur_epsilon(config::init_epsilon),
        _delta((config::init_epsilon - config::final_epsilon) / ((config::episodes) / config::epsilon_decrement)),
        _seq_num(seq_num),
        _rand((uint32_t)time(nullptr)),
        _step_counter(0),
        _episode_counter(0),
        _replay_memorty_size(0),
        _rest_actions(seq_num),
        _actions(seq_num)
{
    for (int i = 0; i < seq_num; i++)
        _actions[i] = i;
    _rest_actions = _actions;
}

int64_t DQN::select(const std::vector<state_type>& state)
{
    int64_t action;

    if ((double)(_rand() % 100001) / 100000 < _cur_epsilon)
    {
        uint32_t idx = _rand() % _rest_actions.size();
        action = _rest_actions[idx];
        _rest_actions.erase(idx + _rest_actions.begin());
    }
    else
    {
        torch::Tensor action_values = _eval_net.forward(
                torch::from_blob((void*)state.data(),
                                 { 1, _seq_num }, torch::kInt32).clone().to(torch::kFloat));
        while (true)
        {
            action = torch::argmax(action_values, 1).item<int64_t>();
            auto iterator = find(_rest_actions.begin(), _rest_actions.end(), action);
            if (iterator != _rest_actions.end())
            {
                _rest_actions.erase(iterator);
                break;
            }
            else
            {
                auto next_state = state;
                next_state[_seq_num - _rest_actions.size()] = (state_type)action;
                push({state, action, next_state, -1, 1 });
                action_values[0][action] = -2;
            }
        }
    }

    return action;
}

void DQN::update()
{
    _step_counter++;
    if (_replay_memory.size() < config::batch_size)
        return;

    if (0 == _step_counter % config::net_update_iteration)
        copy_parameters();

    std::vector<state_type> b_states, b_next_states;
    std::vector<int64_t> b_actions;
    std::vector<float> b_rewards;
    std::vector<int32_t> b_dones;
    sample(b_states, b_actions, b_next_states, b_rewards, b_dones);

    torch::Tensor rewards = torch::from_blob(b_rewards.data(), { config::batch_size }, torch::kFloat32).unsqueeze_(1).clone();
    torch::Tensor dones = torch::from_blob(b_dones.data(), { config::batch_size }, torch::kInt32).unsqueeze_(1).clone();
    torch::Tensor actions = torch::from_blob(b_actions.data(), {config::batch_size, 1}, torch::kInt64).clone();
    torch::Tensor states = torch::from_blob(b_states.data(), { config::batch_size, _seq_num }, torch::kInt32).clone().to(torch::kFloat).permute({0, 1});
    torch::Tensor next_states = torch::from_blob(b_next_states.data(), { config::batch_size, _seq_num }, torch::kInt32).to(torch::kFloat).clone().permute({ 0, 1 });

    torch::Tensor q_eval = _eval_net.forward(states).gather(1, actions);

    torch::autograd::GradMode::set_enabled(false);
    torch::Tensor q_target = rewards + dones * config::gamma * std::get<0>(_target_net.forward(next_states).max(1)).unsqueeze_(1);
    torch::autograd::GradMode::set_enabled(true);

    _loss = torch::mse_loss(q_eval, q_target);
    _eval_net.zero_grad();
    _loss.backward();
    _optimizer.step();

}

int64_t DQN::predict(const std::vector<state_type>& state)
{
    torch::Tensor res = _eval_net.forward(torch::from_blob(const_cast<std::vector<state_type>&>(state).data(),
                                                           { 1, _seq_num }, torch::kInt32).clone().to(torch::kFloat));
    return torch::argmax(res).item<int64_t>();
}

float DQN::predict_q_value(const std::vector<state_type>& state)
{
    torch::Tensor res = _eval_net.forward(torch::from_blob(const_cast<std::vector<state_type>&>(state).data(),
        { 1, _seq_num }, torch::kInt32).clone().to(torch::kFloat));
    return torch::max(res).item<float>();
}

void DQN::push(Transition transition)
{
    if (_replay_memorty_size < config::replay_memory_size)
    {
        _replay_memorty_size++;
        _replay_memory.emplace_back(std::move(transition));
    }
    else
        _replay_memory[_replay_memorty_size++ % config::replay_memory_size] = std::move(transition);
}

void DQN::save(const std::string& path)
{
    torch::serialize::OutputArchive outputArchive;
    _eval_net.save(outputArchive);
    outputArchive.save_to(path);
}

void DQN::load(const std::string& path)
{
    torch::serialize::InputArchive inputArchive;
    inputArchive.load_from(path);

    _eval_net.load(inputArchive);
    copy_parameters();
}

void DQN::reset()
{
    _episode_counter++;

    if (0 == _episode_counter % config::epsilon_decrement)
        _cur_epsilon -= _delta;
    _rest_actions = _actions;
}

void DQN::sample(std::vector<state_type>& state, std::vector<int64_t>& action,
                  std::vector<state_type>& next_state, std::vector<float>& reward, std::vector<int32_t>& done)
{
    std::vector<Transition> samples;
    std::sample(_replay_memory.begin(), _replay_memory.end(), std::back_inserter(samples),
        config::batch_size, std::mt19937{ _rand() });

    state.resize(config::batch_size * _seq_num);
    next_state.resize(config::batch_size * _seq_num);
    action.resize(config::batch_size);
    reward.resize(config::batch_size);
    done.resize(config::batch_size);
    for (int i = 0; i < config::batch_size; i++)
    {
        const Transition& t = samples[i];
        memcpy(&state[i * _seq_num], std::get<0>(t).data(), sizeof(state_type) * _seq_num);
        action[i] = std::get<1>(t);
        memcpy(&next_state[i * _seq_num], std::get<2>(t).data(), sizeof(state_type) * _seq_num);
        reward[i] = std::get<3>(t);
        done[i] = std::get<4>(t);
    }
}

void DQN::copy_parameters()
{
    torch::autograd::GradMode::set_enabled(false);
    auto eval_params = _eval_net.named_parameters();
    auto target_params = _target_net.named_parameters(true);
    auto buffers = _target_net.named_buffers(true);
    for (auto& val : eval_params)
    {
        auto name = val.key();
        auto* t = target_params.find(name);
        if (t != nullptr)
            t->copy_(val.value());
        else
        {
            std::cout << "Can not find the parameter while coping the net parameters." << std::endl;
            exit(-1);
        }
    }
    torch::autograd::GradMode::set_enabled(true);
}