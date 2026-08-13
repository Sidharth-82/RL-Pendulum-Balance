#pragma once

// Minimal assertion harness. Not a framework -- a framework here would mean a
// second dependency for the desktop build and a fight over how it cross-compiles,
// to buy test discovery this project does not need. Every test is a plain main()
// that returns non-zero on failure, which is all ctest reads.
//
// Checks do not abort: a run reports every failure it finds, because when a
// dynamics change breaks four invariants at once, seeing all four is what tells
// you which one is upstream.

#include <cmath>
#include <format>
#include <iostream>
#include <string>

namespace check {

inline int total = 0;
inline int failed = 0;
inline int pending = 0;
inline std::string section_name;

inline void section(const std::string& name) {
    section_name = name;
}

// Marks the current section as not-yet-implemented rather than failed, so a test
// written ahead of the code still runs end to end and reports how far the
// implementation has got. Pending sections do not fail the run -- an unimplemented
// function is a known state, not a regression.
inline void skip(const std::string& reason) {
    ++pending;
    std::cout << std::format("PEND  [{}]\n      {}\n", section_name, reason);
}

inline void fail_line(const char* file, int line, const std::string& expr,
                      const std::string& detail) {
    ++failed;
    std::cout << std::format("FAIL  [{}]  {}:{}\n      {}\n", section_name, file, line, expr);
    if (!detail.empty()) {
        std::cout << std::format("      {}\n", detail);
    }
}

inline void boolean(bool ok, const char* expr, const char* file, int line) {
    ++total;
    if (!ok) {
        fail_line(file, line, expr, {});
    }
}

inline void near(double actual, double expected, double tol, const char* expr, const char* file,
                 int line) {
    ++total;
    const double diff = std::abs(actual - expected);
    if (!(diff <= tol)) {
        fail_line(file, line, expr,
                  std::format("actual {:.17g}, expected {:.17g}, |diff| {:.4e} > tol {:.4e}",
                              actual, expected, diff, tol));
    }
}

inline int summary(const std::string& name) {
    std::cout << std::format("{}: {} checks, {} failed", name, total, failed);
    if (pending > 0) {
        std::cout << std::format(", {} sections pending", pending);
    }
    std::cout << '\n';
    return failed == 0 ? 0 : 1;
}

}  // namespace check

#define CHECK(expr) ::check::boolean((expr), #expr, __FILE__, __LINE__)
#define CHECK_NEAR(actual, expected, tol) \
    ::check::near((actual), (expected), (tol), #actual " ~= " #expected, __FILE__, __LINE__)
