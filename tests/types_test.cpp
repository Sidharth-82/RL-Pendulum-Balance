// State's flat form is the boundary with LQR and RL. A layout mistake here does
// not crash -- it silently routes every gain onto the wrong signal, and the result
// still trains, just against the wrong problem.

#include "core/types.hpp"

#include "check.hpp"

int main() {
    check::section("resize sizes and zeroes everything");
    {
        core::State s;
        s.x = 7.0;
        s.xdot = -3.0;
        s.resize(3);

        CHECK(s.n_links() == 3);
        CHECK(s.theta.size() == 3);
        CHECK(s.omega.size() == 3);
        CHECK_NEAR(s.x, 0.0, 0.0);
        CHECK_NEAR(s.xdot, 0.0, 0.0);
        for (int i = 0; i < 3; ++i) {
            CHECK_NEAR(s.theta[i], 0.0, 0.0);
            CHECK_NEAR(s.omega[i], 0.0, 0.0);
        }
    }

    check::section("flat size is 2n + 2");
    {
        CHECK(core::State::flat_size(1) == 4);
        CHECK(core::State::flat_size(3) == 8);
        CHECK(core::State::flat_size(core::kMaxLinks) == 2 * core::kMaxLinks + 2);
    }

    check::section("flatten packs [x, xdot, theta, omega]");
    {
        core::State s;
        s.resize(3);
        s.x = 0.11;
        s.xdot = 0.22;
        s.theta << 1.0, 2.0, 3.0;
        s.omega << 4.0, 5.0, 6.0;

        const Eigen::VectorXd v = s.flatten();
        CHECK(v.size() == 8);
        // Spelled out index by index rather than compared against a round trip:
        // a self-consistent but transposed layout survives a round trip intact.
        CHECK_NEAR(v(0), 0.11, 0.0);
        CHECK_NEAR(v(1), 0.22, 0.0);
        CHECK_NEAR(v(2), 1.0, 0.0);
        CHECK_NEAR(v(3), 2.0, 0.0);
        CHECK_NEAR(v(4), 3.0, 0.0);
        CHECK_NEAR(v(5), 4.0, 0.0);
        CHECK_NEAR(v(6), 5.0, 0.0);
        CHECK_NEAR(v(7), 6.0, 0.0);
    }

    check::section("unflatten inverts flatten");
    {
        for (int n = 1; n <= core::kMaxLinks; ++n) {
            core::State original;
            original.resize(n);
            original.x = 0.3 * n;
            original.xdot = -0.7 * n;
            for (int i = 0; i < n; ++i) {
                original.theta[i] = 0.1 * (i + 1);
                original.omega[i] = -0.2 * (i + 1);
            }

            core::State restored;
            restored.unflatten(original.flatten(), n);

            CHECK(restored.n_links() == n);
            CHECK_NEAR(restored.x, original.x, 0.0);
            CHECK_NEAR(restored.xdot, original.xdot, 0.0);
            for (int i = 0; i < n; ++i) {
                CHECK_NEAR(restored.theta[i], original.theta[i], 0.0);
                CHECK_NEAR(restored.omega[i], original.omega[i], 0.0);
            }
        }
    }

    check::section("unflatten resizes a state that was the wrong size");
    {
        core::State s;
        s.resize(1);

        Eigen::VectorXd v(8);
        v << 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0;
        s.unflatten(v, 3);

        CHECK(s.n_links() == 3);
        CHECK_NEAR(s.theta[2], 5.0, 0.0);
        CHECK_NEAR(s.omega[2], 8.0, 0.0);
    }

    return check::summary("types");
}
