# Validation, not unit tests in the usual sense. They are layered, and the layering
# is the point -- each one closes a hole the one below it cannot see:
#
#   types / controller / delay_line / actuator / sensors
#     Component contracts. Cheap, exact, and mostly written against bugs that were
#     actually found: a delay line that passed values straight through, a quantiser
#     that returned counts instead of metres, a set_spec that updated copies.
#
#   physics_validation -- N=1 only, where alpha = (3/2L)(g sin th - a cos th) is
#     derivable by hand. Pins the absolute scale of h_ and D_, which energy
#     conservation cannot: conservation is invariant to a common factor on them.
#
#   energy_conservation -- N=1..3 with deliberately dissimilar links. Checks
#     Plant::energy against link geometry (the only check on build_inertia_terms),
#     then conservation under RK4, then the belt force via the work-energy theorem.
#
# The two physics tests write an HTML report next to their binary. Reading them is
# how a regression gets diagnosed; the exit code only says that one happened.

add_executable(header_compile_check header_compile_check.cpp)
target_link_libraries(header_compile_check PRIVATE core)
add_test(NAME header_compile_check COMMAND header_compile_check)

# --- component contracts -----------------------------------------------------
foreach(unit types controller delay_line actuator sensors)
  add_executable(${unit}_test ${unit}_test.cpp)
  target_link_libraries(${unit}_test PRIVATE core)
  add_test(NAME ${unit}_test COMMAND ${unit}_test)
endforeach()

# --- physics -----------------------------------------------------------------
# Reports go next to the binary rather than into the source tree, so a build
# directory stays disposable and repeated runs do not dirty the checkout.
add_executable(physics_validation physics_validation.cpp)
target_link_libraries(physics_validation PRIVATE core)
add_test(NAME physics_validation
         COMMAND physics_validation "$<TARGET_FILE_DIR:physics_validation>/physics_validation.html")

add_executable(energy_conservation energy_conservation.cpp)
target_link_libraries(energy_conservation PRIVATE core)
add_test(NAME energy_conservation
         COMMAND energy_conservation "$<TARGET_FILE_DIR:energy_conservation>/energy_conservation.html")

# --- LQR baseline ------------------------------------------------------------
# Written ahead of the implementation. Sections whose functions still throw
# logic_error report PEND and do not fail the run, so this is runnable from the
# first commit and the pending count tracks progress.
add_executable(lqr_test lqr_test.cpp)
target_link_libraries(lqr_test PRIVATE baseline)
add_test(NAME lqr_test COMMAND lqr_test)

# --- RL environment ----------------------------------------------------------
add_executable(env_test env_test.cpp)
target_link_libraries(env_test PRIVATE sim)
add_test(NAME env_test COMMAND env_test)
