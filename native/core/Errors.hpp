#pragma once
#include <stdexcept>
#include <string>

namespace wavetrace {

// Base error for the whole pipeline; bound to one stable Python exception in Bindings.cpp.
class WaveTraceError : public std::runtime_error {
public:
  explicit WaveTraceError(const std::string& message) : std::runtime_error(message) {}
};

// Invalid CsiFrame geometry/access — the only failure the core types can raise in Phase 1.
class FrameError : public WaveTraceError {
public:
  explicit FrameError(const std::string& message) : WaveTraceError(message) {}
};

}  // namespace wavetrace
