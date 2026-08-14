"""Watch the pendulum.

    python tools/view.py                              # interactive, hanging at rest
    python tools/view.py --traj run/trajectory_560.csv # replay a logged episode
    python tools/view.py --traj ... --video out.mp4    # render to a file instead

Replay, not re-simulation. The trajectory files already hold the exact state
core::Plant produced, so this writes qpos straight from the CSV and calls
mj_forward. That means what you watch is what actually happened during training --
not MuJoCo's opinion of what would have happened, which would drift from it.

Re-simulating in MuJoCo is a separate job (validating the dynamics against core/,
or driving the model live from a policy). Keeping playback free of MuJoCo's
integrator means a disagreement between the two engines can never be mistaken for
a rendering artifact.
"""
import argparse
import csv
import math
import os
import shutil
import subprocess
import sys

import mujoco
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "pendulum.xml")


def read_trajectory(path):
    """t, x, theta_1..N from a trainer trajectory CSV."""
    rows = list(csv.DictReader(open(path, newline="")))
    if not rows:
        raise SystemExit(f"{path}: no rows")
    thetas = sorted(k for k in rows[0] if k == "theta" or k.startswith("theta"))
    return {
        "t": np.array([float(r["t"]) for r in rows]),
        "x": np.array([float(r["x"]) for r in rows]),
        "theta": np.array([[float(r[k]) for k in thetas] for r in rows]),
        "action": np.array([float(r.get("action", 0.0)) for r in rows]),
    }


def set_state(model, data, x, theta):
    data.qpos[0] = x
    data.qpos[1:1 + len(theta)] = theta
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def make_camera(model, distance):
    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.lookat[:] = [0.0, 0.0, 0.02]
    cam.distance = distance
    cam.azimuth = 90.0    # looking down -y, so the swing plane faces the viewer
    cam.elevation = -6.0
    return cam


def clip_time(traj, start, end):
    """Restrict a trajectory to [start, end] seconds.

    The default view is misleading on this task: swing-up takes about 0.3 s of an
    8 s episode, so evenly spaced stills over the whole run are seven pictures of a
    pendulum standing still. Use --end 1.5 to actually see it move.
    """
    t = traj["t"]
    keep = np.ones(len(t), dtype=bool)
    if start is not None:
        keep &= t >= start
    if end is not None:
        keep &= t <= end
    if not keep.any():
        raise SystemExit(f"no samples in [{start}, {end}] s; "
                         f"trajectory spans {t[0]:g}-{t[-1]:g} s")
    return {k: v[keep] for k, v in traj.items()}


def run_video(model, traj, out_path, width, height, fps, speed, distance):
    """Renders offscreen and pipes raw frames to ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not on PATH; use --frames to write PNGs instead")

    dt = float(traj["t"][1] - traj["t"][0])
    stride = max(1, int(round((1.0 / fps) * speed / dt)))
    data = mujoco.MjData(model)
    cam = make_camera(model, distance)

    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-an", "-vcodec", "libx264", "-pix_fmt", "yuv420p", "-crf", "20",
         out_path],
        stdin=subprocess.PIPE)

    with mujoco.Renderer(model, height, width) as renderer:
        for i in range(0, len(traj["t"]), stride):
            set_state(model, data, traj["x"][i], traj["theta"][i])
            renderer.update_scene(data, cam)
            proc.stdin.write(renderer.render().tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    print(f"wrote {out_path}  ({len(traj['t'])//stride} frames at {fps} fps, "
          f"{speed:g}x speed)")


def run_gif(model, traj, out_path, width, height, fps, speed, distance):
    """Animated GIF, for embedding in the README.

    GitHub renders a GIF inline; a relative-path .mp4 only becomes a link. Goes
    through ffmpeg's palettegen rather than PIL because this scene is a few flat
    greys plus one saturated blue, and a naive 256-colour quantisation bands the
    floor gradient badly.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise SystemExit("ffmpeg not on PATH; needed for --gif")

    dt = float(traj["t"][1] - traj["t"][0])
    stride = max(1, int(round((1.0 / fps) * speed / dt)))
    data = mujoco.MjData(model)
    cam = make_camera(model, distance)

    proc = subprocess.Popen(
        [ffmpeg, "-y", "-loglevel", "error",
         "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
         "-vf", "split[a][b];[a]palettegen=stats_mode=diff[p];"
                "[b][p]paletteuse=dither=bayer:bayer_scale=3",
         "-loop", "0", out_path],
        stdin=subprocess.PIPE)

    frames = 0
    with mujoco.Renderer(model, height, width) as renderer:
        for i in range(0, len(traj["t"]), stride):
            set_state(model, data, traj["x"][i], traj["theta"][i])
            renderer.update_scene(data, cam)
            proc.stdin.write(renderer.render().tobytes())
            frames += 1
    proc.stdin.close()
    if proc.wait() != 0:
        raise SystemExit("ffmpeg failed")
    size_kb = os.path.getsize(out_path) / 1024
    print(f"wrote {out_path}  ({frames} frames, {size_kb:.0f} KB)")


def run_frames(model, traj, out_dir, width, height, count, distance):
    """Evenly spaced stills -- for a report, or when there is no ffmpeg."""
    from PIL import Image
    os.makedirs(out_dir, exist_ok=True)
    data = mujoco.MjData(model)
    cam = make_camera(model, distance)
    picks = np.linspace(0, len(traj["t"]) - 1, count).astype(int)
    with mujoco.Renderer(model, height, width) as renderer:
        for n, i in enumerate(picks):
            set_state(model, data, traj["x"][i], traj["theta"][i])
            renderer.update_scene(data, cam)
            Image.fromarray(renderer.render()).save(
                os.path.join(out_dir, f"frame_{n:03d}.png"))
    print(f"wrote {count} frames to {out_dir}")


def run_interactive(model, traj, speed, distance):
    import time
    import mujoco.viewer

    data = mujoco.MjData(model)
    if traj is None:
        # Hanging at rest -- the pose every episode starts from.
        mujoco.mj_resetDataKeyframe(model, data, 0)
        mujoco.mj_forward(model, data)
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = distance
            viewer.cam.azimuth = 90.0
            viewer.cam.elevation = -6.0
            print("showing the start pose; close the window to exit")
            while viewer.is_running():
                mujoco.mj_forward(model, data)
                viewer.sync()
                time.sleep(0.02)
        return

    dt = float(traj["t"][1] - traj["t"][0])
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = distance
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -6.0
        print(f"replaying {len(traj['t'])} steps at {speed:g}x; "
              f"close the window to exit")
        while viewer.is_running():
            start = time.time()
            for i in range(len(traj["t"])):
                if not viewer.is_running():
                    break
                set_state(model, data, traj["x"][i], traj["theta"][i])
                viewer.sync()
                target = start + traj["t"][i] / speed
                remaining = target - time.time()
                if remaining > 0:
                    time.sleep(remaining)
            time.sleep(0.6)  # beat between loops
            start = time.time()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--traj", help="trajectory CSV from the trainer")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="playback rate; 0.25 is useful for the swing-up itself")
    ap.add_argument("--video", help="render to this mp4 instead of opening a window")
    ap.add_argument("--gif", help="render to this animated gif (for the README)")
    ap.add_argument("--frames", help="write evenly spaced PNG stills to this directory")
    ap.add_argument("--count", type=int, default=8, help="stills for --frames")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--start", type=float, help="clip playback to this second")
    ap.add_argument("--end", type=float,
                    help="clip playback here; --end 1.5 frames the swing-up")
    ap.add_argument("--distance", type=float, default=1.25,
                    help="camera distance; raise it to see the whole rail")
    args = ap.parse_args()

    if not os.path.exists(args.model):
        raise SystemExit(f"{args.model} not found -- run tools/make_mjcf.py first")
    model = mujoco.MjModel.from_xml_path(args.model)

    traj = read_trajectory(args.traj) if args.traj else None
    if traj is not None and (args.start is not None or args.end is not None):
        traj = clip_time(traj, args.start, args.end)
    if traj is not None:
        n_links = model.nq - 1
        if traj["theta"].shape[1] != n_links:
            raise SystemExit(
                f"{args.traj} has {traj['theta'].shape[1]} link angle(s) but "
                f"{args.model} has {n_links}; regenerate the model from the same "
                f"config.json the run used")

    if args.video or args.frames or args.gif:
        if traj is None:
            raise SystemExit("--video, --gif and --frames need --traj")
        if args.video:
            run_video(model, traj, args.video, args.width, args.height,
                      args.fps, args.speed, args.distance)
        if args.gif:
            run_gif(model, traj, args.gif, args.width, args.height,
                    args.fps, args.speed, args.distance)
        if args.frames:
            run_frames(model, traj, args.frames, args.width, args.height,
                       args.count, args.distance)
        return

    run_interactive(model, traj, args.speed, args.distance)


if __name__ == "__main__":
    sys.exit(main())
