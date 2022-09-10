#include "environment.h"
#include "utils.h"

#include <algorithm>
#include <array>

Environment::Environment(const std::vector<std::string> &sequences) :
        _sequences(sequences),
        _current(sequences.size(), -1),
        _alignment(sequences.size()),
        _max_len(std::max_element(sequences.begin(), sequences.end(),
                                  [](const auto &lhs, const auto &rhs) { return lhs.size() < rhs.size(); })->size()),
        _index(0),
        _max_reward(MATCH_REWARD * sequences.size() * (sequences.size() - 1) * _max_len / 32)
{

}

std::tuple<std::vector<state_type>, float, int32_t> Environment::step(int64_t action)
{
    float reward;

    if (0 == _index)
    {
        _current[_index] = action;
        _alignment[_index] = _sequences[action];
        reward = 0;
    }
    else
    {
        auto pfl = profile();
        auto align = pairwise_alignment(pfl, _sequences[action]);
        _alignment[_index] = move(align);
        _current[_index] = action;
        reward = calc_reward();
    }

    if (++_index == _sequences.size())
    {
        return { _current, reward, 0};
    }
    else
    {
        return { _current, reward, 1};
    }
}

std::string Environment::pairwise_alignment(const std::string &profile, const std::string &target)
{
    static auto match_func = [](const char &a, const char &b)
    {
        if (a == '-' || b == '-') { return GAP_PENALTY; }
        else if (a == b) { return MATCH_REWARD; }
        return MISMATCH_PENALTY;
    };
    auto n = profile.size();
    auto m = target.size();

    std::vector<std::vector<int>> score(m+ 1, std::vector<int>(n + 1));

    for (int i = 0; i < m + 1; i++)
    {
        score[i][0] = GAP_PENALTY * i;
    }
    for (int i = 0; i < n + 1; i++)
    {
        score[0][i] = GAP_PENALTY * i;
    }

    for (auto i = 1; i < m + 1; i++)
    {
        for (auto j = 1; j < n + 1; j++)
        {
            auto match = score[i - 1][j - 1] + match_func(profile[j - 1], target[i - 1]);
            auto del = score[i - 1][j] + GAP_PENALTY;
            auto insert = score[i][j - 1] + GAP_PENALTY;
            score[i][j] = std::max(match, std::max(del, insert));
        }
    }

    std::string res;

    auto i = m, j = n;
    while (i > 0 && j > 0)
    {
        auto cur = score[i][j];
        auto diagonal = score[i - 1][j - 1];
        auto up = score[i][j - 1];
        auto left = score[i - 1][j];

        if (cur == diagonal + match_func(profile[j - 1], target[i - 1]))
        {
            res.push_back(target[i - 1]);
            i--;
            j--;
        }
        else if (cur == up + GAP_PENALTY)
        {
            res.push_back('-');
            j--;
        }
        else if (cur == left + GAP_PENALTY)
        {
            for_each(_alignment.begin(), _alignment.begin() + _index, [&](std::string& seq)
                {
                    seq.insert(j, 1, '-');
                });
            res.push_back(target[--i]);
        }
    }

    while (j > 0)
    {
        res.push_back('-');
        j--;
    }
    while (i > 0)
    {
        res.push_back(target[--i]);
        for_each(_alignment.begin(), _alignment.begin() + _index, [&](std::string& seq)
            {
                seq.insert(j, 1, '-');
            });
    }

    return {res.rbegin(), res.rend()};
}

std::string Environment::profile()
{
    static std::array<char, 4> nucleotide = {'A', 'T', 'C', 'G'};
    int n_count[4]{0};
    std::vector<std::array<int32_t, 4>> table(_alignment[0].size(), {0});

    for (int i = 0; i < _alignment[0].size(); i++)
    {
        for (int j = 0; j < _index; j++)
        {
            if (_alignment[j][i] == 'A')
            {
                n_count[0]++;
                table[i][0]++;
            }
            else if (_alignment[j][i] == 'T')
            {
                n_count[1]++;
                table[i][1]++;
            }
            else if (_alignment[j][i] == 'C')
            {
                n_count[2]++;
                table[i][2]++;
            }
            else if (_alignment[j][i] == 'G')
            {
                n_count[3]++;
                table[i][3]++;
            }
        }
    }

    std::string res;
    for (int i = 0; i < _alignment[0].size(); i++)
    {
        for (int j = 0; j < 4; j++)
        {
            table[i][j] = table[i][j] * table[i][j] * n_count[j];
        }
        res.push_back(nucleotide[argmax(table[i].begin(), table[i].end())]);
    }

    return res;
}

int Environment::calc_sum_of_pairs()
{
    int score = 0;
    for (int i = 0; i < _alignment[0].size(); i++)
    {
        for (int j = 0; j < _alignment.size() - 1; j++)
        {
            for (int k = j + 1; k < _alignment.size(); k++)
            {
                if (_alignment[j][i] == '-' || _alignment[k][i] == '-')
                {
                    score += GAP_PENALTY;
                }
                else if (_alignment[j][i] == _alignment[k][i])
                {
                    score += MATCH_REWARD;
                }
                else
                {
                    score += MISMATCH_PENALTY;
                }
            }
        }
    }

    return score;
}

std::vector<state_type> Environment::reset()
{
    _current.swap(std::vector<state_type>(_sequences.size(), -1));
    _alignment.swap(std::vector<std::string>(_sequences.size()));
    _index = 0;
    return _current;
}

std::vector<std::string> Environment::alignment()
{
    return _alignment;
}

uint32_t Environment::max_reward()
{
    return _max_reward;
}

float Environment::calc_reward()
{
    int reward = 0;
    for (int i = 0; i < _alignment[0].size(); i++)
    {
        for (int j = 0; j < _index; j++)
        {
            if (_alignment[_index][i] == '-' || _alignment[j][i] == '-')
            {
                reward += GAP_PENALTY;
            }
            else if (_alignment[_index][i] == _alignment[j][i])
            {
                reward += MATCH_REWARD;
            }
            else
            {
                reward += MISMATCH_PENALTY;
            }
        }
    }

    return (float)reward / (float)_max_reward;
}