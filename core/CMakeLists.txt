# The only target that also cross-compiles to the Raspberry Pi.
#
# Hard rule: core links Eigen and nothing else. No LibTorch, no sim-only helpers,
# no logging frameworks. Everything else in the project may depend on core; core
# depends on nobody. This is what makes the controller running on real hardware
# literally the same code that RL trained against, and it is far easier to hold
# from the start than to untangle later.
#
# Compiled rather than header-only: separate translation units keep incremental
# builds fast and error messages readable. The inlining that header-only would
# have bought -- Plant::accel folding into the RK4 and rollout loops -- comes back
# via link-time optimisation, enabled for the release configs in the top-level
# CMakeLists.

add_library(core STATIC
  src/types.cpp
  src/plant.cpp
  src/controller.cpp
  src/delay_line.cpp
  src/actuator.cpp
  src/sensors.cpp
)

target_include_directories(core PUBLIC include)
target_link_libraries(core PUBLIC Eigen3::Eigen)
