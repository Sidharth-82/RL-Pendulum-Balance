# LQR baseline: linearize the plant about upright, solve the continuous-time
# algebraic Riccati equation, emit gains in the same 2N+3 layout the RL policy
# will output.
#
# Worth building before PPO. It answers "is this plant stabilizable at all under
# the actuator envelope" from a Riccati solve rather than from a training run
# that quietly fails to converge, and it gives RL both an initialization and a
# baseline to beat.
#
add_library(baseline STATIC src/lqr.cpp)
target_include_directories(baseline PUBLIC include)
target_link_libraries(baseline PUBLIC core)
