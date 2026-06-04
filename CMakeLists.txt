cmake_minimum_required(VERSION 3.18)
project(wavetrace LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Release by default: the hot-path DSP (Phase 3+) needs -O3.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(pybind11 CONFIG REQUIRED)

# Single native extension `_wavetrace`; Phase 1 ships only core types + bindings.
pybind11_add_module(_wavetrace
  src/Bindings.cpp
)
target_include_directories(_wavetrace PRIVATE src)

# Land the .so inside the importable Python package.
install(TARGETS _wavetrace DESTINATION wavetrace)
