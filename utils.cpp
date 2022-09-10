#include "utils.h"
#include <cassert>

std::vector<std::string> load_sequence(const std::string& path)
{
    std::ifstream file(path);
    std::vector<std::string> res;

    assert(file.is_open());

    std::string s, temp;
    bool is_first = true;
    while (!file.eof())
    {
        getline(file, temp);
        if (temp.empty())
            continue;
        else if (temp[0] == '>') {
            if (!is_first)
                res.push_back(s);
            is_first = false;
            s.clear();
            continue;
        }
        else
            s += temp;
        temp.clear();
    }
    res.push_back(s);
    return res;
}

void print_sequences(const std::vector<std::string>& ss)
{
    for (const auto& s : ss)
        std::cout << s << std::endl;
}

ProgressBar::ProgressBar(const uint64_t& p) :
        _progress(p),
        _iteration(1),
        _start_point(),
        _pre_point(),
        _iteration_per_sec(0),
        _time_count(0.0),
        _total_time(0),
        _left_time(0),
        _speed(0)
{
    _start_point = std::chrono::steady_clock::now();
    _pre_point = _start_point;
}

uint64_t& ProgressBar::operator++()
{
    _progress++;
    _iteration_per_sec++;
    auto point = std::chrono::steady_clock::now();
    _time_count += std::chrono::duration_cast<std::chrono::nanoseconds>(point - _pre_point).count();
    _pre_point = point;
    calc_time();
    return _progress;
}

uint64_t ProgressBar::operator++(int)
{
    auto old_val = _progress;
    _progress++;
    _iteration_per_sec++;
    auto point = std::chrono::steady_clock::now();
    _time_count += std::chrono::duration_cast<std::chrono::nanoseconds>(point - _pre_point).count();
    _pre_point = point;
    calc_time();
    return old_val;
}

void ProgressBar::calc_time()
{
    if (_time_count > 1000000000)
    {
        _speed = (double)_iteration_per_sec / ((double)_time_count / 1000000000);
        _left_time = (uint64_t)((double )(_iteration - _progress) / _speed);
        _iteration_per_sec = 0;
        _time_count -= 1e9;
    }

    _total_time = std::chrono::duration_cast<std::chrono::seconds>((std::chrono::steady_clock::now() - _start_point)).count();
}

void ProgressBar::print_progress_bar() const
{
    std::cout.clear();
    std::cout << "\r";
    std::cout.flush();
    std::cout << std::fixed << std::setprecision(0) << _progress << "/" << _iteration
              << " ["
              << std::left << std::setw(50) << std::setfill('*')
              << std::string(static_cast<int>(round((double)_progress / (double)_iteration * 50)), '#')
              << "] ";

    std::cout << "  " << std::right
              << std::setw(2) << std::setfill('0') << _total_time / 3600 << ":"
              << std::setw(2) << std::setfill('0') << (_total_time % 3600) / 60 << ":"
              << std::setw(2) << std::setfill('0') << _total_time % 60 << "/"
              << std::setw(2) << std::setfill('0') << _left_time / 3600 << ":"
              << std::setw(2) << std::setfill('0') << (_left_time % 3600) / 60 << ":"
              << std::setw(2) << std::setfill('0') << _left_time % 60
              << "  " << std::setprecision(2) << _speed << "/s";

    if (_progress == _iteration)
        std::cout << std::endl;
}

uint64_t ProgressBar::iteration() const
{
    return _iteration;
}

bool ProgressBar::finish() const
{
    return _progress >= _iteration;
}

ProgressBar::operator uint64_t() const
{
    return _progress;
}

uint64_t ProgressBar::operator+(const uint64_t& increment) const
{
    return _progress + increment;
}

uint64_t& ProgressBar::operator+=(const uint64_t& increment)
{
    return _progress += increment;
}

ProgressBar& ProgressBar::operator=(const uint64_t& progress)
{
    _progress = progress;
    return *this;
}

bool ProgressBar::operator<(const int& iteration)
{
    _iteration = iteration;
    print_progress_bar();
    return _progress < iteration;
}