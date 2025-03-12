#pragma once
#include <string>
#include <sstream>
#include <chrono>
#include <iomanip>

// https://github.com/HowardHinnant/date.git v3.0.1
#include "date/date.h"


namespace vtime {

    // Set std::chrono::time_point from string
    template <typename T>
    T time_point_from_string(
        const std::string& str,
        const std::string fmt = "%F")
    {
// for some reasone gcc finds warning inside date/date.h
// fixing is not worth the effort
#if defined __GNUC__ && !defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
        T tp;
        std::stringstream ss{str};
        ss >> date::parse(fmt, tp);
        return tp;
#if defined __GNUC__ && !defined(__clang__)
    #pragma GCC diagnostic pop
#endif
    }

    // Convert std::chrono::time_point to string
    template <typename T>
    std::string time_point_to_string(
        const T& time_point,
        const std::string fmt = "%FT%TZ")
    {
        // refs
        // - https://stackoverflow.com/questions/34857119/how-to-convert-stdchronotime-point-to-string
        // - https://howardhinnant.github.io/date/date.html#to_stream_formatting
        std::stringstream ss{};
        ss << date::format(fmt, time_point);
        return ss.str();
    }

    std::string seconds_to_hhmmss_string(const double seconds)
    {
        std::stringstream ss{};
        ss << std::setw(2) << std::setfill('0') << ((int)seconds / 60 / 60) % 60
            << ":"
            << std::setw(2) << std::setfill('0') << ((int)seconds / 60) % 60
            << ":"
            << std::setw(2) << std::setfill('0') << (int)seconds % 60;
        return ss.str();
    }
}
