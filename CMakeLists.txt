cmake_minimum_required(VERSION 3.16)
project(OLIVE VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ARM SoC: enable NEON via Eigen; on x86 dev machines fall back to SSE
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm")
    add_compile_options(-O3 -march=native -ffast-math)
    message(STATUS "Target: ARM SoC (Vigx Ant-H1 Pro) — enabling NEON")
else()
    add_compile_options(-O3 -march=native -ffast-math)
    message(STATUS "Target: x86 dev machine")
endif()

# ── Eigen3 (header-only, required for matrix math) ─────────────────────────
find_package(Eigen3 3.4 QUIET)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found via find_package, fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG        3.4.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(eigen)
    set(EIGEN3_INCLUDE_DIR ${eigen_SOURCE_DIR})
endif()

# ── Library sources ─────────────────────────────────────────────────────────
set(OLIVE_SOURCES
    src/model.cpp
    src/sensor.cpp
    src/reward.cpp
    src/trainer.cpp
    src/intent.cpp
)

add_library(olive_lib STATIC ${OLIVE_SOURCES})
target_include_directories(olive_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# ── Main executable ─────────────────────────────────────────────────────────
add_executable(olive_deploy src/main.cpp)
target_link_libraries(olive_deploy PRIVATE olive_lib)

# ── Evaluation / ablation executable ────────────────────────────────────────
add_executable(olive_eval scripts/eval.cpp)
target_link_libraries(olive_eval PRIVATE olive_lib)

# ── Unit tests ───────────────────────────────────────────────────────────────
enable_testing()
add_executable(olive_tests tests/test_olive.cpp)
target_link_libraries(olive_tests PRIVATE olive_lib)
add_test(NAME OliveUnitTests COMMAND olive_tests)

message(STATUS "OLIVE configured. Build: mkdir build && cd build && cmake .. && make -j$(nproc)")
