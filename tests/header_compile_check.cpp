// Compiles every core header and instantiates nothing. Catches type and syntax
// errors in the declarations while the bodies are still unwritten, so the
// scaffold stays honest as it fills in.
//
// Deliberately calls nothing: the functions are declared but not defined, so any
// call would fail at link rather than telling you something useful.

#include "core/actuator.hpp"
#include "core/controller.hpp"
#include "core/delay_line.hpp"
#include "core/plant.hpp"
#include "core/sensors.hpp"
#include "core/types.hpp"

static_assert(core::kMaxLinks >= 3, "triple pendulum must fit");

// The whole point of the fixed-capacity Eigen types: storage is inline, so no
// heap traffic in the hot loop and nothing to trip the Pi's real-time deadline.
static_assert(sizeof(core::Vec) >= sizeof(double) * core::kMaxLinks,
              "Vec should carry inline storage, not a heap pointer");
static_assert(sizeof(core::Mat) >= sizeof(double) * core::kMaxLinks * core::kMaxLinks,
              "Mat should carry inline storage, not a heap pointer");

int main() { return 0; }
