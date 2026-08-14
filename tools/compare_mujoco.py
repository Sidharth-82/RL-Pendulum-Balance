"""Runs the MuJoCo model against a core::Plant reference and reports the difference.

    tools\\reference_traj.exe --mode driven --out ref.csv
    python tools/compare_mujoco.py ref.csv

This is the test that validates the MJCF as a *dynamics* model. Replay does not:
mass, inertia, damping and gravity are all unread when a trajectory is played back,
so a rod ten times too heavy renders identically. Only stepping both engines from
the same state under the same input can see them.

The two engines share nothing but config.json. core::Plant assembles A(theta) and
C(theta) from a hand-derived Lagrangian; MuJoCo builds its mass matrix and bias
force with composite-rigid-body and recursive Newton-Euler off the MJCF. Agreement
is therefore evidence the derivation is right, which the energy tests -- written
from the same derivation -- cannot be.

Driving the cart
----------------
core::Plant treats the carriage as a PRESCRIBED-ACCELERATION boundary condition; a
belt drive is a position source, so its motion is commanded, not solved for. MuJoCo
has no such joint, so the force that produces the commanded acceleration is solved
for each step from MuJoCo's own dynamics:

    M qddot + c = tau,   tau = [F, 0...0],   qddot[0] = a_commanded

Partitioning on the cart DOF (0) and the link DOFs (1:):

    a_links = -M11^-1 (M10 a_cart + c_links)
    F       =  M00 a_cart + M01 a_links + c_cart

A stiff position servo would have been simpler and is what the MJCF carries for
interactive use, but it tracks with a lag that shows up as exactly the kind of small
phase error this comparison exists to detect. This is exact to integrator error.
"""
import argparse
import csv
import sys

import mujoco
import numpy as np


def read_reference(path):
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    names = rows[0].keys()
    thetas = sorted(k for k in names if k.startswith("theta"))
    omegas = sorted(k for k in names if k.startswith("omega"))
    col = lambda k: np.array([float(r[k]) for r in rows])
    return {
        "t": col("t"),
        "accel": col("accel"),
        "x": col("x"),
        "xdot": col("xdot"),
        "theta": np.column_stack([col(k) for k in thetas]),
        "omega": np.column_stack([col(k) for k in omegas]),
        "energy": col("energy") if "energy" in names else None,
    }


def cart_force(model, data, a_cart):
    """Force on the slide joint that yields exactly a_cart, given the current state."""
    nv = model.nv
    M = np.zeros((nv, nv))
    # MuJoCo 3.x: mj_fullM(model, data, dst). Older bindings took the packed matrix
    # (data.qM) as the third argument instead.
    mujoco.mj_fullM(model, data, M)
    c = data.qfrc_bias.copy()          # C(q,v)v + gravity, MuJoCo's own RNE

    if nv == 1:                        # cart with no links; degenerate but valid
        return M[0, 0] * a_cart + c[0]

    a_links = np.linalg.solve(M[1:, 1:], -(M[1:, 0] * a_cart + c[1:]))
    return M[0, 0] * a_cart + M[0, 1:] @ a_links + c[0]


def run(model_path, ref, substeps=1, check_tracking=True):
    model = mujoco.MjModel.from_xml_path(model_path)

    # The MJCF carries a stiff <position> actuator for interactive use, and ctrl
    # defaults to zero -- so leaving actuation enabled means an 8000 N/m spring
    # dragging the cart back to x=0 and fighting every force applied here. It shows
    # up as a cart acceleration error equal to the full commanded amplitude.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION

    # The slide joint carries range="-rail/2 rail/2" so a live sim cannot throw the
    # cart into space. core::Plant has no such term: the rail is a TERMINATION
    # decided in code, never a force in the dynamics. Leaving the limit enabled makes
    # MuJoCo's cart stop dead at the rail end while the reference sails past, and the
    # comparison then reports a modelling disagreement that is really a constraint
    # only one of the two engines has.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT

    data = mujoco.MjData(model)

    n_links = model.nv - 1
    if ref["theta"].shape[1] != n_links:
        raise SystemExit(f"reference has {ref['theta'].shape[1]} link(s), model has "
                         f"{n_links}; regenerate the model from the same config.json")

    dt = float(ref["t"][1] - ref["t"][0])
    if abs(dt - model.opt.timestep) > 1e-12:
        raise SystemExit(f"timestep mismatch: reference {dt}, model "
                         f"{model.opt.timestep}; both come from config.json, so one "
                         f"of them is stale")
    model.opt.timestep = dt / substeps

    # Same initial condition, exactly.
    data.qpos[0] = ref["x"][0]
    data.qpos[1:] = ref["theta"][0]
    data.qvel[0] = ref["xdot"][0]
    data.qvel[1:] = ref["omega"][0]
    mujoco.mj_forward(model, data)

    steps = len(ref["t"])
    theta = np.zeros((steps, n_links))
    x = np.zeros(steps)
    accel_error = np.zeros(steps)

    for k in range(steps):
        # Refresh M and qfrc_bias at the CURRENT state before solving for the force.
        # After mj_step under RK4 they hold values from an intermediate substage, and
        # solving the constraint from a stale mass matrix produces a force that is
        # simply wrong -- visibly so once the configuration is moving quickly.
        mujoco.mj_forward(model, data)
        theta[k] = data.qpos[1:]
        x[k] = data.qpos[0]
        if k == steps - 1:
            break

        # Substepping holds the prescribed ACCELERATION fixed across the reference
        # step, which is what core::Plant does. Applying a single constant force for
        # the whole step is not the same thing: the force that produces a given cart
        # acceleration depends on configuration, and the configuration moves.
        a_cmd = ref["accel"][k]
        for s in range(substeps):
            if s:
                mujoco.mj_forward(model, data)
            data.qfrc_applied[0] = cart_force(model, data, a_cmd)
            if check_tracking:
                # Check BEFORE integrating. After mj_step, data.qacc is the last RK4
                # substage at the new state, not the value that was commanded.
                mujoco.mj_forward(model, data)
                accel_error[k] = max(accel_error[k],
                                     abs(data.qacc[0] - a_cmd))
            mujoco.mj_step(model, data)

    return {"theta": theta, "x": x, "accel_error": accel_error}


def error_at(ref, got, horizon):
    """Worst link-angle error up to `horizon` seconds."""
    upto = ref["t"] <= horizon
    return float(np.abs(got["theta"][upto] - ref["theta"][upto]).max())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("reference", help="CSV from tools/reference_traj")
    ap.add_argument("--model", default="tools/pendulum.xml")
    ap.add_argument("--horizon", type=float, default=0.2,
                    help="seconds over which agreement is judged")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="max |dtheta| in radians within the horizon")
    args = ap.parse_args()

    ref = read_reference(args.reference)
    span = ref["t"][-1]

    # Judged on a SHORT horizon and on convergence, not on the max over the whole
    # run. The two engines hold different quantities constant across a step -- MuJoCo
    # a force, core::Plant an acceleration -- which is a first-order difference that
    # vanishes as the step shrinks. A driven pendulum spinning through revolutions
    # then amplifies whatever seed that leaves, so the max error at 3 s measures
    # Lyapunov divergence, not modelling agreement. Refining the step is what
    # separates the two: discretisation shrinks, a wrong model does not.
    runs = {n: run(args.model, ref, substeps=n) for n in (1, 4, 16)}

    print(f"{args.reference}  ->  {args.model}")
    print(f"  {len(ref['t'])} steps, {span:.2f} s, {ref['theta'].shape[1]} link(s), "
          f"|a| up to {np.abs(ref['accel']).max():.1f} m/s^2, "
          f"{np.abs(np.diff(ref['theta'][:, 0])).sum():.1f} rad travelled")
    print()
    print(f"  cart accel constraint     {runs[1]['accel_error'].max():.2e} m/s^2 "
          f"(force solve; should be ~1e-15)")
    print()

    horizons = [h for h in (0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0) if h <= span]
    print("  max |dtheta| (rad) by horizon and MuJoCo substeps per reference step")
    print("      " + "".join(f"{h:>10.2f}s" for h in horizons))
    for n, got in runs.items():
        print(f"  x{n:<3d}" + "".join(f"{error_at(ref, got, h):>11.2e}" for h in horizons))
    print()

    e1 = error_at(ref, runs[1], args.horizon)
    e16 = error_at(ref, runs[16], args.horizon)
    ratio = e1 / e16 if e16 > 0 else float("inf")

    print(f"  at the {args.horizon:g}s horizon: {e1:.2e} -> {e16:.2e} for 16x finer "
          f"steps, a factor of {ratio:.1f}")

    converging = ratio > 4.0          # first order would give 16; allow slack
    accurate = e16 < args.tol
    exact_drive = runs[1]["accel_error"].max() < 1e-8
    ok = converging and accurate and exact_drive

    print()
    print(f"  -> {'PASS' if ok else 'FAIL'}")
    if ok:
        print("     The two engines agree; the residual is the constant-force vs")
        print("     constant-acceleration hold, and it converges away.")
    else:
        if not exact_drive:
            print("     The prescribed-acceleration force solve is not holding. Check")
            print("     that actuation, limits and contacts are disabled, and that M")
            print("     and qfrc_bias are refreshed at the current state first.")
        if not converging:
            print("     Error does not shrink with the step, so this is a MODEL")
            print("     difference, not discretisation. Likely a convention:")
            print("       ~pi offset      reference pose built hanging, not upright")
            print("       mirrored sign   hinge axis backwards")
            print("       wrong magnitude inertia about the joint vs about the COM,")
            print("                       or com_dist measured from the wrong end")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
