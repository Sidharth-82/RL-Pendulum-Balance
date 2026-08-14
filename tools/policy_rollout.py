"""Runs the trained policy against MuJoCo physics, closed loop.

    python tools/policy_rollout.py <run>/policy_weights.txt --video swingup.mp4

Unlike view.py, nothing here is replayed. MuJoCo integrates the dynamics, the
network chooses the command from MuJoCo's own state, and the loop closes. If the
policy only worked because of some quirk of core::Plant's integration, this is
where it shows.

Three things have to match sim::Env exactly or the policy sees a different world
than the one it trained in, and behaves accordingly:

  the observation layout and its normalisation constants,
  the actuator (slew clamp, velocity clamp, achieved-vs-commanded acceleration),
  the decimation -- one policy decision held across control_decimation plant steps.

All of them come from the same config.json the trainer read.
"""
import argparse
import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def load_config(path):
    text = re.sub(r"^\s*//.*$", "", open(path, encoding="utf-8").read(), flags=re.M)
    return json.loads(text)


def load_weights(path):
    """Reads the plain-text dump written by export_weights() in rl/src/main.cpp."""
    out, lines = {}, open(path, encoding="utf-8").read().split("\n")
    i = 0
    while i < len(lines):
        head = lines[i].strip()
        i += 1
        if not head:
            continue
        parts = head.split()
        name, shape = parts[0], [int(v) for v in parts[1:]]
        values = np.fromstring(lines[i], sep=" ")
        i += 1
        out[name] = values.reshape(shape)
    return out


class Policy:
    """The actor's forward pass. Linear -> tanh -> Linear -> tanh -> Linear.

    Deterministic: action_mean, never a sample. log_std exists in the dump and is
    ignored on purpose -- sampling here would measure exploration noise rather than
    what the policy learned.
    """

    def __init__(self, weights):
        def layer(idx):
            return (weights[f"actor.{idx}.weight"], weights[f"actor.{idx}.bias"])
        self.layers = [layer(0), layer(2), layer(4)]   # 1 and 3 are the tanhs

    def __call__(self, obs):
        h = np.asarray(obs, dtype=np.float64)
        for k, (W, b) in enumerate(self.layers):
            h = W @ h + b
            if k < len(self.layers) - 1:
                h = np.tanh(h)
        return h


def cart_force(model, data, a_cart):
    """Force yielding exactly a_cart on the slide joint. See compare_mujoco.py."""
    nv = model.nv
    M = np.zeros((nv, nv))
    mujoco.mj_fullM(model, data, M)
    c = data.qfrc_bias.copy()
    if nv == 1:
        return M[0, 0] * a_cart + c[0]
    a_links = np.linalg.solve(M[1:, 1:], -(M[1:, 0] * a_cart + c[1:]))
    return M[0, 0] * a_cart + M[0, 1:] @ a_links + c[0]


def rollout(cfg, model_path, policy, seconds=None):
    act, env = cfg["actuator"], cfg["env"]
    dt = env["dt"]
    decim = env["control_decimation"]
    half_rail = 0.5 * act["rail_length"]
    max_accel, max_velocity = act["max_accel"], act["max_velocity"]
    omega_scale = env["omega_scale"]
    seconds = seconds if seconds is not None else env["episode_seconds"]

    model = mujoco.MjModel.from_xml_path(model_path)
    # Same three as the comparison: the <position> actuator would fight the applied
    # force, and the joint limit is a termination decided in code, not a force.
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_ACTUATION
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_LIMIT
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONTACT
    data = mujoco.MjData(model)

    n = model.nv - 1
    data.qpos[0] = 0.0
    data.qpos[1:] = env["theta_start"][:n]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    rows = []
    steps = int(seconds / dt)
    off_rail = False
    for k in range(steps):
        if k % decim == 0:
            x, v = data.qpos[0], data.qvel[0]
            obs = [x / half_rail, v / max_velocity]
            for i in range(n):
                th = data.qpos[1 + i]
                obs += [math.sin(th), math.cos(th), data.qvel[1 + i] / omega_scale]
            raw = float(policy(obs)[0])
            # Env clamps to [-1, 1] before scaling; the network output is unsquashed.
            command = np.clip(raw, -1.0, 1.0) * max_accel

        # Actuator::apply -- the velocity clamp means the achieved acceleration is
        # not the commanded one at top speed, and Plant::accel must see what the
        # carriage actually did.
        v_before = data.qvel[0]
        v_after = float(np.clip(v_before + command * dt, -max_velocity, max_velocity))
        achieved = (v_after - v_before) / dt

        rows.append((k * dt, data.qpos[0], data.qvel[0],
                     *data.qpos[1:], *data.qvel[1:], raw))

        mujoco.mj_forward(model, data)
        data.qfrc_applied[0] = cart_force(model, data, achieved)
        mujoco.mj_step(model, data)

        if abs(data.qpos[0]) > half_rail:
            off_rail = True
            break

    return rows, n, off_rail


def write_csv(rows, n, path):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "x", "xdot"] + [f"theta{i}" for i in range(n)]
                   + [f"omega{i}" for i in range(n)] + ["action"])
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", help="policy_weights.txt from a training run")
    ap.add_argument("--config", default=os.path.join(ROOT, "config.json"))
    ap.add_argument("--model", default=os.path.join(HERE, "pendulum.xml"))
    ap.add_argument("--csv", help="write the rollout here")
    ap.add_argument("--seconds", type=float)
    args = ap.parse_args()

    cfg = load_config(args.config)
    policy = Policy(load_weights(args.weights))
    rows, n, off_rail = rollout(cfg, args.model, policy, args.seconds)

    t = np.array([r[0] for r in rows])
    theta = np.array([r[3:3 + n] for r in rows])
    upright = np.degrees(np.abs(np.arctan2(np.sin(theta[:, 0]), np.cos(theta[:, 0]))))
    settle = None
    for i in range(len(upright)):
        if np.all(upright[i:] < 15.0):
            settle = t[i]
            break

    print(f"MuJoCo closed-loop rollout, {t[-1]:.2f} s, {len(rows)} plant steps")
    print(f"  off rail          {'YES' if off_rail else 'no'}")
    print(f"  settle to <15 deg {f'{settle:.2f} s' if settle is not None else 'never'}")
    print(f"  final |angle|     {upright[-1]:.2f} deg")
    print(f"  max |x|           {max(abs(r[1]) for r in rows):.3f} m")

    if args.csv:
        write_csv(rows, n, args.csv)
        print(f"  wrote {args.csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
