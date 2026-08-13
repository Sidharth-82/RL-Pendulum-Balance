# Raspberry Pi real-time control loop. Built under a cross-compile preset with an
# ARM toolchain file, not as part of the desktop build -- hence no add_subdirectory(pi)
# in the top-level CMakeLists.
#
# Links core/ only. The trained policy arrives as exported weights, not as a
# LibTorch dependency: if the policy emits fixed gains this is 2N+3 floats, and
# even for state-dependent gains a small MLP forward pass is a few dozen lines of
# Eigen. LibTorch on ARM is a heavy runtime with allocation behaviour you do not
# want inside a loop that has to hold 200-500 Hz with tight jitter.
#
# add_executable(pid_rl_pi src/main.cpp src/gpio.cpp src/encoders.cpp)
# target_link_libraries(pid_rl_pi PRIVATE core)
