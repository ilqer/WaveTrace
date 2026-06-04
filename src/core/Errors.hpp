#pragma once
#include <stdexcept>
#include <string>

namespace wavetrace {

// Base error for the whole pipeline. Each module narrows it (FrameError in Phase 2, etc.).
// Bound to a Python exception in Bindings.cpp so Python callers catch one stable type.
class WaveTraceError : public std::runtime_error {
public:
  explicit WaveTraceError(const std::string& msg) : std::runtime_error(msg) {}
};

// Invalid CsiFrame geometry/access — the only failure the core types can raise in Phase 1.
class FrameError : public WaveTraceError {
public:
  explicit FrameError(const std::string& msg) : WaveTraceError(msg) {}
};

}  // namespace wavetrace
