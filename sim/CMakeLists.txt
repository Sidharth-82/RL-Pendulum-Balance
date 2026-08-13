# Desktop-only simulation layer: episode/env wrapper around core, domain
# randomization sampling, rollout batching, trajectory logging.
#
# Deliberately separate from core/ -- none of this ships to the Pi.
#
add_library(sim STATIC src/env.cpp)
target_include_directories(sim PUBLIC include)
target_link_libraries(sim PUBLIC core)
