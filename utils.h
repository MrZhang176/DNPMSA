//
// Created by neronzhang on 2022/5/18.
//

#ifndef EXP_UTILS_H
#define EXP_UTILS_H


#include <fstream>
#include <algorithm>
#include <vector>
#include <cmath>
#include <iomanip>
#include <chrono>
#include <map>
#include <iostream>
#include <string>

typedef int32_t    state_type;

namespace config{
    constexpr float init_epsilon = 0.8f;
    constexpr float final_epsilon = 0.f;
    constexpr uint32_t epsilon_decrement = 100;
    constexpr uint32_t net_update_iteration = 256;
    constexpr float gamma = 1.f;
    constexpr float alpha = 0.0001f;
    constexpr uint32_t replay_memory_size = 5000;
    constexpr uint32_t batch_size = 128;
    constexpr uint32_t episodes = 50000;
}


template<typename Iterator>
inline int argmax(Iterator begin, Iterator end)
{
    return std::distance(begin, std::max_element(begin, end));
}

template<typename Iterator, typename Val>
inline int index(Iterator begin, Iterator end, Val val)
{
    return std::distance(begin, std::find(begin, end, val));
}

std::vector<std::string> load_sequence(const std::string& path);

void print_sequences(const std::vector<std::string>& ss);

class ProgressBar {
public:
    explicit ProgressBar(const uint64_t& p = 0);
    ProgressBar(const ProgressBar&) = delete;
    ProgressBar& operator=(const ProgressBar&) = delete;

    uint64_t operator+(const uint64_t& increment) const;
    uint64_t& operator+=(const uint64_t& increment);
    ProgressBar& operator=(const uint64_t& progress);
    bool operator<(const int& iteration);
    uint64_t& operator++();
    uint64_t operator++(int);

    [[nodiscard]] uint64_t iteration() const;
    [[nodiscard]] bool finish() const;

    explicit operator uint64_t() const;
private:
    void print_progress_bar() const;
    void calc_time();

    uint64_t _iteration, _progress;
    std::chrono::steady_clock::time_point _start_point, _pre_point;
    uint64_t _time_count, _iteration_per_sec, _total_time, _left_time;
    double _speed;
};


#endif //EXP_UTILS_H
