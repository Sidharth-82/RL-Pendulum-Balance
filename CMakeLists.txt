cmake_minimum_required(VERSION 3.21)
project(PID_RL_Tuneing LANGUAGES CXX)

# MSYS2 sits on the system PATH on this machine, so a configure that runs before the
# MSVC environment is ready will auto-detect MinGW g++ and cache the GNU linker.
# CMAKE_LINKER is never re-derived afterwards, so pinning the compiler on a later
# configure does not undo it -- you get cl.exe compiling and ld.exe linking, which
# fails with pages of "cannot find /nologo". Fail loudly and point at the real fix.
if(NOT MSVC)
  message(FATAL_ERROR
    "Expected MSVC, got '${CMAKE_CXX_COMPILER_ID}' at ${CMAKE_CXX_COMPILER}.\n"
    "A configure almost certainly ran before the Visual Studio environment was set up "
    "and cached a MinGW toolchain. Delete the build directory and reconfigure:\n"
    "  Remove-Item -Recurse -Force \"$ENV{LOCALAPPDATA}\\pidrl-build\"")
endif()

# The same cached-compiler trap has a second form, and it looks nothing like a
# toolchain problem. This machine carries two VS 2022 installs -- Community with
# toolsets 14.36/14.39, BuildTools with 14.44 -- and CMAKE_CXX_COMPILER is cached on
# the first configure. Build later under an environment whose INCLUDE points at a
# NEWER STL than the cached cl.exe and the STL's own version guard fires:
#
#   yvals_core.h(912): error C2338: static_assert failed:
#     'error STL1001: Unexpected compiler version, expected MSVC 19.44 or newer.'
#
# followed by hundreds of cascade errors inside <type_traits>, <concepts>, <compare>
# and <xutility>, none of which name a file in this project. CMakePresets.json pins
# "toolset" to 14.36 so CMake Tools selects a matching vcvars every time; this check
# catches the case where something else supplied the environment anyway.
#
# Comparing CMAKE_CXX_COMPILER_VERSION against a floor would not find this: the
# compiler really is 19.36 in both the working and the broken case. What differs is
# INCLUDE, so compare the toolset directory in the compiler path against the one the
# environment is actually handing to the preprocessor.
if(MSVC AND DEFINED ENV{INCLUDE})
  string(REGEX MATCH "MSVC/([0-9]+\\.[0-9]+\\.[0-9]+)/" _pidrl_ignored "${CMAKE_CXX_COMPILER}")
  if(CMAKE_MATCH_1)
    set(PIDRL_COMPILER_TOOLSET "${CMAKE_MATCH_1}")
    file(TO_CMAKE_PATH "$ENV{INCLUDE}" PIDRL_INCLUDE_DIRS)
    if(NOT PIDRL_INCLUDE_DIRS MATCHES "MSVC/${PIDRL_COMPILER_TOOLSET}/")
      message(FATAL_ERROR
        "MSVC toolset mismatch. The cached compiler is toolset "
        "${PIDRL_COMPILER_TOOLSET}:\n"
        "  ${CMAKE_CXX_COMPILER}\n"
        "but INCLUDE points at a different toolset's STL:\n"
        "  $ENV{INCLUDE}\n"
        "Building this way produces STL1001 plus hundreds of cascade errors in the "
        "standard headers. Either use a shell whose vcvars matches the cached "
        "compiler, or wipe the cache and reconfigure under the toolset you want:\n"
        "  Remove-Item -Recurse -Force \"$ENV{LOCALAPPDATA}\\pidrl-build\"")
    endif()
  endif()
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Export compile_commands.json so clangd / the VSCode C++ extension see the real flags.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# LibTorch is ~1 GB and drags in a Debug/Release ABI constraint, and nothing before
# the PPO step needs it. Flip this on when rl/ actually has content.
option(PIDRL_BUILD_RL "Build the LibTorch PPO trainer" ON)
option(PIDRL_BUILD_TESTS "Build validation tests" ON)

# core is compiled rather than header-only, so Plant::accel cannot inline into a
# caller's rollout loop across the object boundary. LTO restores that for the
# configs where throughput matters, and is left off in Debug where it only slows
# builds and confuses the debugger.
include(CheckIPOSupported)
check_ipo_supported(RESULT PIDRL_IPO_SUPPORTED OUTPUT PIDRL_IPO_MESSAGE)
if(PIDRL_IPO_SUPPORTED)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELEASE ON)
  set(CMAKE_INTERPROCEDURAL_OPTIMIZATION_RELWITHDEBINFO ON)
else()
  message(STATUS "LTO unavailable: ${PIDRL_IPO_MESSAGE}")
endif()

# --- dependencies ------------------------------------------------------------
# Eigen is header-only, so fetching it costs nothing at link time and keeps the
# repo self-contained without adding vcpkg as another tool in the chain.
include(FetchContent)

# SOURCE_SUBDIR names a directory that has no CMakeLists.txt, so FetchContent
# downloads Eigen but never runs its build system. Eigen is header-only, so its
# CMake package buys us nothing, and skipping it avoids both its test targets and
# the policy error CMake 4.x raises against 3.4.0's cmake_minimum_required.
FetchContent_Declare(eigen
  GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
  GIT_TAG        3.4.0
  GIT_SHALLOW    TRUE
  SOURCE_SUBDIR  do-not-configure)
FetchContent_MakeAvailable(eigen)

# SYSTEM keeps Eigen's headers from spraying warnings once we turn /W4 on.
add_library(eigen_headers INTERFACE)
target_include_directories(eigen_headers SYSTEM INTERFACE ${eigen_SOURCE_DIR})
add_library(Eigen3::Eigen ALIAS eigen_headers)

# nlohmann/json, single header. Fetched as one file rather than through
# FetchContent because the upstream repository is ~100 MB of tests and CI we
# would clone and then ignore; the amalgamated header is 900 KB and is the whole
# library. Same treatment as Eigen after that: INTERFACE target, SYSTEM include.
set(PIDRL_JSON_DIR "${CMAKE_BINARY_DIR}/_deps/nlohmann-single")
if(NOT EXISTS "${PIDRL_JSON_DIR}/nlohmann/json.hpp")
  message(STATUS "Downloading nlohmann/json single header")
  file(DOWNLOAD
    "https://raw.githubusercontent.com/nlohmann/json/v3.11.3/single_include/nlohmann/json.hpp"
    "${PIDRL_JSON_DIR}/nlohmann/json.hpp"
    EXPECTED_HASH SHA256=9bea4c8066ef4a1c206b2be5a36302f8926f7fdc6087af5d20b417d0cf103ea6
    TLS_VERIFY ON
    STATUS PIDRL_JSON_STATUS)
  list(GET PIDRL_JSON_STATUS 0 PIDRL_JSON_CODE)
  if(NOT PIDRL_JSON_CODE EQUAL 0)
    list(GET PIDRL_JSON_STATUS 1 PIDRL_JSON_MESSAGE)
    message(FATAL_ERROR "Failed to download nlohmann/json: ${PIDRL_JSON_MESSAGE}")
  endif()
endif()
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json SYSTEM INTERFACE "${PIDRL_JSON_DIR}")

# --- targets -----------------------------------------------------------------
add_subdirectory(core)      # shared with the Pi -- Eigen only, no other deps
add_subdirectory(baseline)  # LQR: linearize + Riccati
add_subdirectory(sim)       # desktop only: env, rollouts, domain randomization
add_subdirectory(config)    # config.json -> populated structs; desktop only

if(PIDRL_BUILD_RL)
  add_subdirectory(rl)
endif()

if(PIDRL_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# pi/ is intentionally not added here -- it builds under a cross-compile preset
# against an ARM toolchain file, not as part of the desktop build.
