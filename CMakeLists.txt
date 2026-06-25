cmake_minimum_required(VERSION 3.20)
project(qec_fast_loss LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# GCC >= 15 enforces transitive header inclusion; several stim headers omit
# <cstdint> and rely on it being pulled in indirectly. Force-include it
# everywhere so the upstream stim source compiles without modification.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-include cstdint -fno-strict-aliasing -O3 -g0)
endif()

find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# Prefer a system-installed pybind11, but fall back to fetching it at configure
# time when it's not available. This makes configuring in clean environments
# (e.g., CI or fresh developer machines) work without requiring a separate
# pybind11 installation step.
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
    )
    FetchContent_MakeAvailable(pybind11)
endif()

set(STIM_ROOT "${CMAKE_SOURCE_DIR}/extern/stim")
if(NOT EXISTS "${STIM_ROOT}/src/stim/circuit/circuit.h")
    message(FATAL_ERROR
        "extern/stim is empty. Initialise submodules with:\n"
        "  git submodule update --init --recursive")
endif()

# Glob all of stim's .cc files, then drop everything we don't need:
#   - tests and perf benchmarks,
#   - stim's own pybind11 bindings (we provide our own module entry),
#   - executable-only sources (main*.cc, stim.cc),
#   - subsystems unused by TableauSimulator: cmd/, diagram/, gen/, search/, py/.
file(GLOB_RECURSE STIM_SOURCES CONFIGURE_DEPENDS
    "${STIM_ROOT}/src/stim/*.cc"
)
list(FILTER STIM_SOURCES EXCLUDE REGEX "\\.test\\.cc$")
list(FILTER STIM_SOURCES EXCLUDE REGEX "\\.perf\\.cc$")
list(FILTER STIM_SOURCES EXCLUDE REGEX "\\.pybind\\.cc$")
list(FILTER STIM_SOURCES EXCLUDE REGEX "/stim/cmd/")
list(FILTER STIM_SOURCES EXCLUDE REGEX "/stim/diagram/")
list(FILTER STIM_SOURCES EXCLUDE REGEX "/stim/gen/")
list(FILTER STIM_SOURCES EXCLUDE REGEX "/stim/search/")
list(FILTER STIM_SOURCES EXCLUDE REGEX "/stim/py/")
list(FILTER STIM_SOURCES EXCLUDE REGEX "/stim/main_namespaced\\.cc$")
list(FILTER STIM_SOURCES EXCLUDE REGEX "/stim/main\\.cc$")

file(GLOB_RECURSE FAST_QEC_LOSS_SOURCES CONFIGURE_DEPENDS
    "src/qec_loss/_native/*.cc"
)

set(PYMATCHING_ROOT "${CMAKE_SOURCE_DIR}/extern/pymatching")
if(NOT EXISTS "${PYMATCHING_ROOT}/src/pymatching/sparse_blossom/driver/mwpm_decoding.h")
    message(FATAL_ERROR
        "extern/pymatching is empty. Initialise submodules with:\n"
        "  git submodule update --init --recursive")
endif()

file(GLOB_RECURSE PYMATCHING_SOURCES CONFIGURE_DEPENDS
    "${PYMATCHING_ROOT}/src/pymatching/*.cc"
)
list(FILTER PYMATCHING_SOURCES EXCLUDE REGEX "\\.test\\.cc$")
list(FILTER PYMATCHING_SOURCES EXCLUDE REGEX "\\.perf\\.cc$")
list(FILTER PYMATCHING_SOURCES EXCLUDE REGEX "\\.pybind\\.cc$")

pybind11_add_module(_native MODULE
    ${FAST_QEC_LOSS_SOURCES}
    ${STIM_SOURCES}
    ${PYMATCHING_SOURCES}
)

target_include_directories(_native PRIVATE
    "${STIM_ROOT}/src"
    "${PYMATCHING_ROOT}/src"
    "${CMAKE_SOURCE_DIR}/extern/xxhash"
)

target_compile_definitions(_native PRIVATE
    STIM_PYBIND11_MODULE_NAME=_native
)

install(TARGETS _native LIBRARY DESTINATION .)
