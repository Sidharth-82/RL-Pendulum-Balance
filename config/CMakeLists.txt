# Reads config.json into populated core/sim structs.
#
# Separate from sim/ so the JSON dependency stays off sim's include path, and well
# away from core/ -- core cross-compiles to the Pi against Eigen and nothing else,
# and a parser has no business in that build.
#
# nlohmann_json is linked PRIVATE: it appears in config.cpp only, never in
# config.hpp, so nothing downstream pays the ~25k-line template cost.
add_library(config STATIC src/config.cpp)
target_include_directories(config PUBLIC include)
target_link_libraries(config PUBLIC sim PRIVATE nlohmann_json)
