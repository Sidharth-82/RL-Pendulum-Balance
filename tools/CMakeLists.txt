# Desktop-only diagnostics. Nothing here ships to the Pi.
#
# reference_traj dumps a core::Plant trajectory under a prescribed carriage
# acceleration so tools/compare_mujoco.py can drive MuJoCo with the same input and
# compare. That comparison is the only thing that tests the mass, inertia and
# damping in the generated MJCF -- replaying a recorded trajectory renders
# identically no matter what those are set to.
add_executable(reference_traj reference_traj.cpp)
target_link_libraries(reference_traj PRIVATE config core)
