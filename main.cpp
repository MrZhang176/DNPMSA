#include <iostream>
#include "dqn.h"
#include "environment.h"

const std::vector<std::string> data = {
    "GTGCTGCCTGGTACAT",
    "GTGCTGCCTGGTACAT",
    "GTGCTGCCTGGTACAT"
};

int main()
{
    const auto &dataset = data;
    Environment env(dataset);
    DQN agent(dataset.size());

    std::vector<float> scores;
    for (ProgressBar progress; progress < config::episodes; ++progress)
    {
        auto state = env.reset();
        while (true)
        {
            auto action = agent.select(state);
            auto [next_state, reward, done] = env.step(action);
            agent.push({ state, action, next_state, reward, done });
            agent.update();
            if (!done)
            {
                break;
            }
            state = std::move(next_state);
        }
        float q_eval = agent.predict_q_value(state);
        scores.emplace_back(q_eval * env.max_reward());
        agent.reset();
    }

    auto state = env.reset();

    for (int i = 0; i < dataset.size(); i++)
    {
        auto action = agent.predict(state);
        env.step(action);
        state[i] = action;
    }
    for (const auto& val : state) std::cout << val << " ";
    std::cout << std::endl;

    print_sequences(env.alignment());

    std::cout << env.calc_sum_of_pairs() << std::endl;

    return 0;
}
