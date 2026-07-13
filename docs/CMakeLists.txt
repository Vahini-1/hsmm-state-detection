cmake_minimum_required(VERSION 3.20)
project(hsmm_regime_engine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# pybind11: prefer an installed copy (pip install pybind11), fall back to
# FetchContent if not found.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
  )
  FetchContent_MakeAvailable(pybind11)
endif()

find_package(Eigen3 CONFIG QUIET)
if(NOT Eigen3_FOUND)
  message(STATUS "Eigen3 not found via find_package; header-only fallback "
                  "expected at cpp/third_party/eigen if not installed system-wide.")
endif()

# ---------------------------------------------------------------------------
# Core C++ engine (static library, testable independently of Python)
# ---------------------------------------------------------------------------
add_library(hsmm_core STATIC
    cpp/src/forward_backward.cpp
    cpp/src/duration_distributions.cpp
    cpp/src/em_engine.cpp
    cpp/src/particle_filter.cpp
    cpp/src/emissions.cpp
)
target_include_directories(hsmm_core PUBLIC cpp/include)
if(Eigen3_FOUND)
  target_link_libraries(hsmm_core PUBLIC Eigen3::Eigen)
endif()

# ---------------------------------------------------------------------------
# pybind11 module: hsmm_regime._core
# ---------------------------------------------------------------------------
pybind11_add_module(_core cpp/src/bindings.cpp)
target_link_libraries(_core PRIVATE hsmm_core)
target_include_directories(_core PRIVATE cpp/include)

# Place the compiled extension directly inside the python package so
# `import hsmm_regime._core` works from an editable install.
set_target_properties(_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/python/hsmm_regime
)

# ---------------------------------------------------------------------------
# Tests (optional, requires Catch2)
# ---------------------------------------------------------------------------
option(BUILD_TESTS "Build C++ unit tests" OFF)
if(BUILD_TESTS)
  find_package(Catch2 3 QUIET)
  if(Catch2_FOUND)
    add_executable(hsmm_tests
        cpp/tests/test_forward_backward.cpp
        cpp/tests/test_particle_filter.cpp
    )
    target_link_libraries(hsmm_tests PRIVATE hsmm_core Catch2::Catch2WithMain)
    include(CTest)
    add_test(NAME hsmm_tests COMMAND hsmm_tests)
  else()
    message(STATUS "Catch2 not found; skipping C++ test target.")
  endif()
endif()
