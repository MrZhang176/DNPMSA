//
// Created by neronzhang on 2022/5/18.
//

#ifndef EXP_ENVIRONMENT_H
#define EXP_ENVIRONMENT_H

#include <vector>
#include <string>
#include <set>
#include "utils.h"

#define MATCH_REWARD        2
#define MISMATCH_PENALTY    -1
#define GAP_PENALTY         -2

class Environment
{
public:
    Environment() = delete;
    explicit Environment(const std::vector<std::string> &sequences);
    ~Environment() = default;

    std::tuple<std::vector<state_type>, float, int32_t> step(int64_t action);

    int32_t calc_sum_of_pairs();

    std::vector<state_type> reset();

    std::vector<std::string> alignment();

    uint32_t max_reward();

private:
    std::string profile();
    std::string pairwise_alignment(const std::string &profile, const std::string &target);
    float calc_reward();

    std::vector<std::string>    _sequences;
    std::vector<state_type>     _current;
    std::vector<std::string>    _alignment;
    uint32_t                    _max_len, _index, _max_reward;
};

#endif //EXP_ENVIRONMENT_H
