# LibTorch PPO trainer. Only added to the build when PIDRL_BUILD_RL=ON.
#
# Setup when you get here:
#   1. Download LibTorch for Windows, *Release* build, CXX11 ABI, from
#      https://pytorch.org/get-started/locally/  (select LibTorch / C++ / CPU)
#   2. Point CMake at it:  -DCMAKE_PREFIX_PATH=C:/path/to/libtorch
#
# The trap: LibTorch ships separate Debug and Release binaries and the config you
# build must match the one you downloaded. Mismatching produces link errors, or
# worse, heap corruption at runtime that looks like a bug in your own code. Use
# the msvc-release / msvc-relwithdebinfo presets with a Release LibTorch.

find_package(Torch REQUIRED)

add_executable(train src/main.cpp src/ppo.cpp src/rollout.cpp src/networks.cpp)
target_include_directories(train PRIVATE include)
target_link_libraries(train PRIVATE sim "${TORCH_LIBRARIES}")

# LibTorch needs the same C++ standard the rest of the project uses, and it will not
# tell you politely if it does not get it.
set_property(TARGET train PROPERTY CXX_STANDARD 20)

# LibTorch DLLs must sit beside the executable on Windows:
if(MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET train POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different ${TORCH_DLLS} $<TARGET_FILE_DIR:train>)
endif()
