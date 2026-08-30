"""Self-contained HTML training report from a run directory.

Reads train_log.csv (one row per iteration) and every trajectory_<i>.csv the
trainer dumped at evaluation, derives the headline numbers, and writes one HTML
file with inline SVG charts.

Deliberately dependency-free -- Python standard library only, no plotting
package, and no script or stylesheet fetched at view time. Same rule as
tests/plot_html.hpp: the report has to open from disk on any machine, years from
now, with no network and no environment. The only <script> is inlined below and
does nothing but the hover crosshair; every chart is fully readable with
JavaScript disabled.

    python tools/training_report.py <run-dir> [-o report.html]

Every number quoted in README.md under "Results" is regenerated here from the
CSVs, so the two cannot drift again without this file disagreeing.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# --- palette -----------------------------------------------------------------
# Categorical slots 1-3 of the reference palette, validated all-pairs in both
# modes (worst CVD dE 9.2 light / 9.4 dark; worst normal-vision dE 24.0 / 20.9).
# Aqua sits at 2.74:1 on the light surface, below the 3:1 bar, so the relief rule
# applies: every series carries a direct label and the table view is always
# present. Do not add a fourth slot without re-running the validator -- slot 4 is
# yellow and puts yellow beside orange, which fails all-pairs.
LIGHT = {
    "surface": "#fcfcfb", "plane": "#f9f9f7",
    "ink": "#0b0b0b", "ink2": "#52514e", "muted": "#898781",
    "grid": "#e1e0d9", "axis": "#c3c2b7", "border": "rgba(11,11,11,0.10)",
    "s1": "#2a78d6", "s2": "#eb6834", "s3": "#1baf7a",
    "critical": "#d03b3b", "good": "#0ca30c",
}
DARK = {
    "surface": "#1a1a19", "plane": "#0d0d0d",
    "ink": "#ffffff", "ink2": "#c3c2b7", "muted": "#898781",
    "grid": "#2c2c2a", "axis": "#383835", "border": "rgba(255,255,255,0.10)",
    "s1": "#3987e5", "s2": "#d95926", "s3": "#199e70",
    "critical": "#d03b3b", "good": "#0ca30c",
}

TAU = 2.0 * math.pi


def wrap(a: float) -> float:
    """Angle folded into (-pi, pi]. Upright is 0; hanging is +-pi."""
    a = math.fmod(a + math.pi, TAU)
    if a <= 0.0:
        a += TAU
    return a - math.pi


# --- axes (Heckbert nice numbers, mirroring tests/plot_html.hpp) --------------

def nice_num(x: float, round_to_nearest: bool) -> float:
    if not (x > 0.0):
        return 1.0
    exp = math.floor(math.log10(x))
    frac = x / (10.0 ** exp)
    if round_to_nearest:
        nice = 1.0 if frac < 1.5 else 2.0 if frac < 3.0 else 5.0 if frac < 7.0 else 10.0
    else:
        nice = 1.0 if frac <= 1.0 else 2.0 if frac <= 2.0 else 5.0 if frac <= 5.0 else 10.0
    return nice * (10.0 ** exp)


def nice_axis(lo: float, hi: float, max_ticks: int = 6):
    if not (hi > lo):
        pad = max(abs(hi), 1.0) * 0.5
        lo, hi = hi - pad, hi + pad
    rng = nice_num(hi - lo, False)
    step = nice_num(rng / max(1, max_ticks - 1), True)
    return math.floor(lo / step) * step, math.ceil(hi / step) * step, step


def ticks(lo: float, hi: float, step: float):
    out, v, guard = [], lo, 0
    while v <= hi + step * 1e-6 and guard < 200:
        out.append(0.0 if abs(v) < step * 1e-9 else v)
        v += step
        guard += 1
    return out


def fmt(v: float) -> str:
    a = abs(v)
    if v == 0:
        return "0"
    if a >= 10000 or a < 0.001:
        return f"{v:.1e}".replace("e-0", "e-").replace("e+0", "e")
    if a >= 100:
        return f"{v:,.0f}"
    if a >= 10:
        return f"{v:.1f}".rstrip("0").rstrip(".")
    if a >= 1:
        return f"{v:.2f}".rstrip("0").rstrip(".")
    return f"{v:.3f}".rstrip("0").rstrip(".")


# --- chart model -------------------------------------------------------------

@dataclass
class Series:
    label: str
    pts: list                      # [(x, y) | (x, None)]
    slot: str = "s1"
    dashed: bool = False


@dataclass
class Rule:
    y: float
    label: str


@dataclass
class Chart:
    key: str
    title: str
    subtitle: str
    x_label: str
    y_label: str
    series: list
    rules: list = field(default_factory=list)
    note: str = ""
    # A phase portrait is parametric -- x is not monotone, so "nearest sample by x"
    # is ambiguous and the crosshair would lie. Those charts opt out.
    hover: bool = True


W, H = 760, 300
PAD_L, PAD_R, PAD_T, PAD_B = 66, 108, 16, 40


def _draw(c: Chart, dy: float = 0.0):
    """The plot itself: every SVG element between <svg> and </svg>, minus the
    hover layer. Shared by the interactive HTML report and the standalone SVGs
    the Markdown report references, so the two can never drift apart.

    dy shifts the whole plot down, which is how the standalone variant makes room
    for a title it has to draw itself.
    """
    xs = [p[0] for s in c.series for p in s.pts]
    ys = [p[1] for s in c.series for p in s.pts if p[1] is not None]
    for r in c.rules:
        ys.append(r.y)
    if not xs or not ys:
        return None

    x0, x1, xstep = nice_axis(min(xs), max(xs))
    y0, y1, ystep = nice_axis(min(ys), max(ys))
    px0, px1 = PAD_L, W - PAD_R
    py0, py1 = H - PAD_B + dy, PAD_T + dy

    def sx(v):
        return px0 + (v - x0) / (x1 - x0) * (px1 - px0)

    def sy(v):
        return py0 + (v - y0) / (y1 - y0) * (py1 - py0)

    out = []

    # gridlines + y ticks
    for t in ticks(y0, y1, ystep):
        y = sy(t)
        out.append(f'<line class="grid" x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}"/>')
        out.append(f'<text class="tick ty" x="{px0 - 10}" y="{y + 4:.1f}">{fmt(t)}</text>')
    # x ticks
    for t in ticks(x0, x1, xstep):
        x = sx(t)
        out.append(f'<text class="tick tx" x="{x:.1f}" y="{py0 + 22}">{fmt(t)}</text>')
    out.append(f'<line class="axis" x1="{px0}" y1="{py0}" x2="{px1}" y2="{py0}"/>')

    # reference rules
    for r in c.rules:
        y = sy(r.y)
        out.append(f'<line class="rule" x1="{px0}" y1="{y:.1f}" x2="{px1}" y2="{y:.1f}"/>')
        # Left-aligned: the right-hand gutter belongs to the direct series labels,
        # and a rule that lands near a converged series was overprinting them.
        out.append(f'<text class="rule-label" x="{px0 + 6}" y="{y - 6:.1f}">{html.escape(r.label)}</text>')

    # series
    labels = []   # (y, x, text, slot) -- de-overlapped after the loop
    for s in c.series:
        segs, cur = [], []
        for (xv, yv) in s.pts:
            if yv is None:
                if len(cur) > 1:
                    segs.append(cur)
                cur = []
            else:
                cur.append(f"{sx(xv):.1f},{sy(yv):.1f}")
        if len(cur) > 1:
            segs.append(cur)
        for seg in segs:
            out.append(
                f'<polyline class="line {s.slot}{" dashed" if s.dashed else ""}" '
                f'points="{" ".join(seg)}"/>'
            )
        # sparse series (evaluations) also get markers -- >=8px hit area.
        # A full 2000-iteration run evaluates 100 times, so the cap has to clear
        # that or the evaluation series silently loses its markers at the end.
        real = [(x, y) for (x, y) in s.pts if y is not None]
        if 0 < len(real) <= 110:
            for (xv, yv) in real:
                out.append(f'<circle class="dot {s.slot}" cx="{sx(xv):.1f}" cy="{sy(yv):.1f}" r="3.2"/>')
        # direct label at the last point -- required relief for light-mode aqua
        if real:
            lx, ly = real[-1]
            labels.append([sy(ly) + 4.0, min(sx(lx) + 8, px1 + 6), s.label, s.slot])

    # Series that converge (every policy ends upright, every action ends at zero)
    # put their labels on top of each other and the text becomes unreadable. Push
    # them apart vertically, keeping order, so the label still points at the right
    # line without overprinting its neighbour.
    GAP = 13.0
    labels.sort(key=lambda L: L[0])
    for i in range(1, len(labels)):
        if labels[i][0] - labels[i - 1][0] < GAP:
            labels[i][0] = labels[i - 1][0] + GAP
    overflow = labels[-1][0] - (py0 + 4) if labels else 0.0
    if overflow > 0:
        for L in labels:
            L[0] -= overflow
    for ly, lx, text, slot in labels:
        out.append(
            f'<text class="direct {slot}" x="{lx:.1f}" y="{ly:.1f}">{html.escape(text)}</text>'
        )

    out.append(f'<text class="axis-title ax-y" x="14" y="{(py0 + py1) / 2:.0f}" '
               f'transform="rotate(-90 14 {(py0 + py1) / 2:.0f})">{html.escape(c.y_label)}</text>')
    out.append(f'<text class="axis-title ax-x" x="{(px0 + px1) / 2:.0f}" y="{py0 + 36:.0f}">'
               f'{html.escape(c.x_label)}</text>')

    geom = {
        "px0": px0, "px1": px1, "py0": py0, "py1": py1,
        "x0": x0, "x1": x1, "y0": y0, "y1": y1,
        "xLabel": c.x_label, "yLabel": c.y_label,
        "series": [{"label": s.label, "slot": s.slot,
                    "pts": [[p[0], p[1]] for p in s.pts if p[1] is not None]}
                   for s in c.series],
    }
    return out, geom, (px0, px1, py0, py1)


def render_chart(c: Chart) -> str:
    """Interactive figure for the HTML report."""
    drawn = _draw(c)
    if drawn is None:
        return ""
    body, geom, (px0, px1, py0, py1) = drawn

    out = [
        f'<figure class="chart" data-key="{c.key}">',
        f'<figcaption><h3>{html.escape(c.title)}</h3>'
        f'<p>{html.escape(c.subtitle)}</p></figcaption>',
        f'<div class="plot"><svg viewBox="0 0 {W} {H}" role="img" '
        f'aria-label="{html.escape(c.title)}. {html.escape(c.subtitle)}" '
        f'preserveAspectRatio="xMidYMid meet">',
    ]
    out += body
    if c.hover:
        out.append(f'<line class="crosshair" x1="0" y1="{py1}" x2="0" y2="{py0}" style="display:none"/>')
        out.append('<g class="hover-dots"></g>')
        out.append(
            f'<rect class="hit" x="{px0}" y="{py1}" width="{px1 - px0}" height="{py0 - py1}" '
            f'fill="transparent"/>'
        )
    out.append('</svg><div class="tooltip" hidden></div></div>')

    if len(c.series) > 1:
        out.append('<ul class="legend">')
        for s in c.series:
            out.append(f'<li><span class="swatch {s.slot}"></span>{html.escape(s.label)}</li>')
        out.append("</ul>")
    if c.note:
        out.append(f'<p class="note">{c.note}</p>')
    out.append(f'<script type="application/json" class="geom">{json.dumps(geom)}</script>')
    out.append("</figure>")
    return "\n".join(out)


# Height added above the plot for the title the standalone SVG has to draw
# itself, and below it for the legend, since a referenced image carries no
# surrounding HTML.
SA_TOP, SA_BOT = 52, 30


def render_standalone_svg(c: Chart) -> str:
    """One chart as a complete SVG document, for the Markdown report.

    Markdown references it with an <img>, so the title, the legend and both
    colour schemes have to live inside the file. Media queries inside a
    referenced SVG are honoured by the browser, so this still follows the
    viewer's light/dark setting rather than being baked to one mode.
    """
    drawn = _draw(c, dy=SA_TOP)
    if drawn is None:
        return ""
    body, _, (px0, px1, py0, py1) = drawn
    total = H + SA_TOP + SA_BOT

    def vars_for(d):
        keep = ("surface", "ink", "ink2", "muted", "grid", "axis", "s1", "s2", "s3")
        return " ".join(f"--{k}: {d[k]};" for k in keep)

    style = (
        f".r {{ {vars_for(LIGHT)} }}"
        f"@media (prefers-color-scheme: dark) {{ .r {{ {vars_for(DARK)} }} }}"
        ".bg { fill: var(--surface); }"
        ".grid { stroke: var(--grid); stroke-width: 1; }"
        ".axis { stroke: var(--axis); stroke-width: 1; }"
        ".rule { stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 4; }"
        ".rule-label { fill: var(--muted); font-size: 10.5px; text-anchor: start; }"
        ".tick { fill: var(--muted); font-size: 11px; }"
        ".ty { text-anchor: end; } .tx { text-anchor: middle; }"
        ".axis-title { fill: var(--muted); font-size: 11.5px; text-anchor: middle; }"
        ".line { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }"
        ".line.dashed { stroke-dasharray: 5 4; }"
        ".dot { stroke: var(--surface); stroke-width: 2; }"
        ".direct { font-size: 11.5px; font-weight: 600; }"
        ".s1 { stroke: var(--s1); } .s2 { stroke: var(--s2); } .s3 { stroke: var(--s3); }"
        "circle.s1 { fill: var(--s1); } circle.s2 { fill: var(--s2); } circle.s3 { fill: var(--s3); }"
        "text.s1 { fill: var(--s1); stroke: none; }"
        "text.s2 { fill: var(--s2); stroke: none; }"
        "text.s3 { fill: var(--s3); stroke: none; }"
        "rect.s1 { fill: var(--s1); } rect.s2 { fill: var(--s2); } rect.s3 { fill: var(--s3); }"
        ".t { fill: var(--ink); font-size: 15px; font-weight: 620; }"
        ".st { fill: var(--ink2); font-size: 11.5px; }"
        ".lg { fill: var(--ink2); font-size: 11.5px; }"
        "svg { font-family: system-ui, -apple-system, 'Segoe UI', sans-serif; }"
    )

    sub = c.subtitle if len(c.subtitle) <= 118 else c.subtitle[:115].rstrip() + "..."
    out = [
        f'<svg xmlns="http:{"//"}www.w3.org/2000/svg" viewBox="0 0 {W} {total}" '
        f'width="{W}" height="{total}" role="img" '
        f'aria-label="{html.escape(c.title)}. {html.escape(c.subtitle)}">',
        f"<style>{style}</style>",
        f'<g class="r"><rect class="bg" x="0" y="0" width="{W}" height="{total}" rx="10"/>',
        f'<text class="t" x="{PAD_L - 40}" y="26">{html.escape(c.title)}</text>',
        f'<text class="st" x="{PAD_L - 40}" y="44">{html.escape(sub)}</text>',
    ]
    out += body
    if len(c.series) > 1:
        lx = PAD_L - 40
        ly = total - 10
        for s in c.series:
            out.append(f'<rect class="{s.slot}" x="{lx}" y="{ly - 9}" width="10" height="10" rx="2.5"/>')
            out.append(f'<text class="lg" x="{lx + 15}" y="{ly}">{html.escape(s.label)}</text>')
            lx += 15 + 7.0 * len(s.label) + 22
    out.append("</g></svg>")
    return "\n".join(out)


def render_table(c: Chart) -> str:
    xsall = sorted({p[0] for s in c.series for p in s.pts if p[1] is not None})
    if len(xsall) > 400:
        keep = set(xsall[:: max(1, len(xsall) // 200)])
        xsall = [x for x in xsall if x in keep]
    lut = [{p[0]: p[1] for p in s.pts if p[1] is not None} for s in c.series]
    rows = []
    for x in xsall:
        cells = "".join(
            f"<td>{fmt(l[x]) if x in l else ''}</td>" for l in lut
        )
        rows.append(f"<tr><th scope=\"row\">{fmt(x)}</th>{cells}</tr>")
    heads = "".join(f"<th scope=\"col\">{html.escape(s.label)}</th>" for s in c.series)
    return (
        f'<details class="tableview"><summary>Table view &mdash; {html.escape(c.title)}</summary>'
        f'<div class="tablewrap"><table><caption>{html.escape(c.title)}</caption><thead><tr>'
        f'<th scope="col">{html.escape(c.x_label)}</th>{heads}</tr></thead>'
        f"<tbody>{''.join(rows)}</tbody></table></div></details>"
    )


# --- data --------------------------------------------------------------------

def load_log(run: Path):
    rows = []
    with (run / "train_log.csv").open(newline="") as fh:
        for r in csv.DictReader(fh):
            def num(k):
                v = (r.get(k) or "").strip()
                if v == "":
                    return None
                try:
                    return float(v)
                except ValueError:
                    return None
            rows.append({k: num(k) for k in r})
    return rows


def load_trajectories(run: Path):
    out = {}
    for f in run.glob("trajectory_*.csv"):
        m = re.search(r"trajectory_(\d+)\.csv$", f.name)
        if not m:
            continue
        with f.open(newline="") as fh:
            rd = list(csv.DictReader(fh))
        out[int(m.group(1))] = [
            {k: float(v) for k, v in r.items() if v not in (None, "")} for r in rd
        ]
    return dict(sorted(out.items()))


UPRIGHT_TOL = math.radians(15.0)


def traj_metrics(rows, hold_seconds=2.0):
    """Everything README quotes per-evaluation, recomputed from the dump."""
    if not rows:
        return {}
    t = [r["t"] for r in rows]
    tilt = [abs(wrap(r["theta"])) for r in rows]
    x = [r["x"] for r in rows]
    act = [r["action"] for r in rows]

    # swing-up: first sample after which the tilt never leaves the 15 deg cone
    swing = None
    for i in range(len(tilt)):
        if all(v < UPRIGHT_TOL for v in tilt[i:]):
            swing = t[i]
            break

    t_end = t[-1]
    hold = [i for i in range(len(t)) if t[i] >= t_end - hold_seconds]
    hold_act = [act[i] for i in hold]
    reversals = sum(1 for a, b in zip(hold_act, hold_act[1:])
                    if (a > 0 > b) or (a < 0 < b))
    return {
        "swing_up_s": swing,
        "final_tilt_deg": math.degrees(tilt[-1]),
        "peak_abs_x": max(abs(v) for v in x),
        "hold_mean_abs_action": (sum(abs(a) for a in hold_act) / len(hold_act)) if hold_act else None,
        "hold_reversal_pct": (100.0 * reversals / max(1, len(hold_act) - 1)) if hold_act else None,
        "duration_s": t_end,
        "samples": len(rows),
    }


# --- page --------------------------------------------------------------------

def css() -> str:
    def block(d):
        return "\n".join(f"    --{k}: {v};" for k, v in d.items())
    return f"""
:root {{ color-scheme: light dark; }}
.viz-root {{
    color-scheme: light;
{block(LIGHT)}
}}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) .viz-root {{
    color-scheme: dark;
{block(DARK)}
  }}
}}
:root[data-theme="dark"] .viz-root {{
    color-scheme: dark;
{block(DARK)}
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--plane); color: var(--ink);
  font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif; }}
.wrap {{ max-width: 1040px; margin: 0 auto; padding: 40px 24px 80px; }}
header h1 {{ font-size: 30px; letter-spacing: -0.02em; margin: 0 0 6px; }}
header .sub {{ color: var(--ink2); margin: 0 0 4px; }}
header .meta {{ color: var(--muted); font-size: 13px; margin: 0; }}
.themebtn {{ float: right; font: inherit; font-size: 13px; padding: 6px 12px;
  border-radius: 999px; border: 1px solid var(--border); background: var(--surface);
  color: var(--ink2); cursor: pointer; }}
h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: .09em;
  color: var(--muted); font-weight: 600; margin: 44px 0 14px;
  padding-bottom: 8px; border-bottom: 1px solid var(--border); }}
.tiles {{ display: grid; gap: 12px; grid-template-columns: repeat(auto-fit, minmax(184px, 1fr)); }}
.tile {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 16px 18px; }}
.tile .k {{ font-size: 12px; color: var(--muted); text-transform: uppercase;
  letter-spacing: .06em; margin-bottom: 8px; }}
.tile .v {{ font-size: 27px; font-weight: 650; letter-spacing: -0.02em; }}
.tile .v small {{ font-size: 14px; font-weight: 500; color: var(--ink2); margin-left: 3px; }}
.tile .d {{ font-size: 12.5px; color: var(--ink2); margin-top: 7px; }}
.chart {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 12px; padding: 18px 18px 10px; margin: 0 0 18px; }}
figcaption h3 {{ margin: 0 0 3px; font-size: 16px; font-weight: 620; }}
figcaption p {{ margin: 0 0 6px; color: var(--ink2); font-size: 13px; }}
.plot {{ position: relative; }}
svg {{ display: block; width: 100%; height: auto; overflow: visible; }}
.grid {{ stroke: var(--grid); stroke-width: 1; }}
.axis {{ stroke: var(--axis); stroke-width: 1; }}
.rule {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 3 4; }}
.rule-label {{ fill: var(--muted); font-size: 10.5px; text-anchor: start; }}
.tick {{ fill: var(--muted); font-size: 11px; font-variant-numeric: tabular-nums; }}
.ty {{ text-anchor: end; }} .tx {{ text-anchor: middle; }}
.axis-title {{ fill: var(--muted); font-size: 11.5px; text-anchor: middle; }}
.line {{ fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }}
.line.dashed {{ stroke-dasharray: 5 4; }}
.dot {{ stroke: var(--surface); stroke-width: 2; }}
.direct {{ font-size: 11.5px; font-weight: 600; }}
.s1 {{ stroke: var(--s1); }} .s2 {{ stroke: var(--s2); }} .s3 {{ stroke: var(--s3); }}
circle.s1 {{ fill: var(--s1); }} circle.s2 {{ fill: var(--s2); }} circle.s3 {{ fill: var(--s3); }}
text.s1 {{ fill: var(--s1); stroke: none; }}
text.s2 {{ fill: var(--s2); stroke: none; }}
text.s3 {{ fill: var(--s3); stroke: none; }}
.crosshair {{ stroke: var(--muted); stroke-width: 1; stroke-dasharray: 2 3; }}
.legend {{ list-style: none; display: flex; flex-wrap: wrap; gap: 16px;
  margin: 6px 0 0; padding: 0; font-size: 12.5px; color: var(--ink2); }}
.legend li {{ display: flex; align-items: center; gap: 7px; }}
.swatch {{ width: 11px; height: 11px; border-radius: 3px; display: inline-block; }}
.swatch.s1 {{ background: var(--s1); }} .swatch.s2 {{ background: var(--s2); }}
.swatch.s3 {{ background: var(--s3); }}
.note {{ font-size: 12.5px; color: var(--ink2); margin: 10px 0 4px;
  padding-top: 10px; border-top: 1px solid var(--border); }}
.tooltip {{ position: absolute; pointer-events: none; background: var(--surface);
  border: 1px solid var(--border); border-radius: 8px; padding: 8px 10px;
  font-size: 12px; color: var(--ink); box-shadow: 0 4px 14px rgba(0,0,0,.13);
  white-space: nowrap; z-index: 5; }}
.tooltip .row {{ display: flex; align-items: center; gap: 7px; margin-top: 3px; }}
.tooltip b {{ font-variant-numeric: tabular-nums; }}
.tableview {{ margin: 0 0 26px; }}
.tableview summary {{ cursor: pointer; font-size: 12.5px; color: var(--ink2);
  padding: 7px 2px; }}
.tablewrap {{ max-height: 320px; overflow: auto; border: 1px solid var(--border);
  border-radius: 8px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12.5px;
  font-variant-numeric: tabular-nums; background: var(--surface); }}
caption {{ text-align: left; padding: 8px 10px; color: var(--muted); font-size: 12px; }}
th, td {{ padding: 5px 10px; text-align: right; border-bottom: 1px solid var(--border); }}
thead th {{ position: sticky; top: 0; background: var(--surface); color: var(--ink2);
  text-align: right; }}
tbody th {{ text-align: left; color: var(--ink2); font-weight: 500; }}
.prose {{ max-width: 74ch; color: var(--ink2); font-size: 14px; }}
.prose strong {{ color: var(--ink); }}
@media print {{ .themebtn, .tableview {{ display: none; }} }}
"""


SVG_NS = "http:" + "//www.w3.org/2000/svg"

HOVER_JS = r"""
document.querySelectorAll('.chart').forEach(function (fig) {
  var svg = fig.querySelector('svg');
  var geomEl = fig.querySelector('script.geom');
  var tip = fig.querySelector('.tooltip');
  var cross = fig.querySelector('.crosshair');
  var dots = fig.querySelector('.hover-dots');
  var hit = fig.querySelector('.hit');
  if (!svg || !geomEl || !hit) return;
  var g = JSON.parse(geomEl.textContent);
  var sx = function (v) { return g.px0 + (v - g.x0) / (g.x1 - g.x0) * (g.px1 - g.px0); };
  var sy = function (v) { return g.py0 + (v - g.y0) / (g.y1 - g.y0) * (g.py1 - g.py0); };

  function hide() { tip.hidden = true; cross.style.display = 'none'; dots.innerHTML = ''; }

  function move(ev) {
    var r = svg.getBoundingClientRect();
    var ux = (ev.clientX - r.left) / r.width * svg.viewBox.baseVal.width;
    var dataX = g.x0 + (ux - g.px0) / (g.px1 - g.px0) * (g.x1 - g.x0);
    var rows = [], anchor = null;
    dots.innerHTML = '';
    g.series.forEach(function (s) {
      if (!s.pts.length) return;
      var best = null, bd = Infinity;
      for (var i = 0; i < s.pts.length; i++) {
        var d = Math.abs(s.pts[i][0] - dataX);
        if (d < bd) { bd = d; best = s.pts[i]; }
      }
      if (!best || bd > (g.x1 - g.x0) * 0.06) return;
      if (anchor === null) anchor = best[0];
      rows.push({ label: s.label, slot: s.slot, x: best[0], y: best[1] });
      var c = document.createElementNS(SVG_NS, 'circle');
      c.setAttribute('cx', sx(best[0]));
      c.setAttribute('cy', sy(best[1]));
      c.setAttribute('r', 4.5);
      c.setAttribute('class', 'dot ' + s.slot);
      dots.appendChild(c);
    });
    if (!rows.length) { hide(); return; }
    var cx = sx(anchor);
    cross.setAttribute('x1', cx); cross.setAttribute('x2', cx);
    cross.style.display = '';
    tip.innerHTML = '<div><b>' + g.xLabel + ' ' + rows[0].x.toLocaleString() + '</b></div>' +
      rows.map(function (r) {
        var val = Math.abs(r.y) >= 1000
          ? r.y.toLocaleString(undefined, { maximumFractionDigits: 0 })
          : parseFloat(r.y.toPrecision(4)).toString();
        return '<div class="row"><span class="swatch ' + r.slot + '"></span>' +
               r.label + ' <b>' + val + '</b></div>';
      }).join('');
    tip.hidden = false;
    var pr = fig.querySelector('.plot').getBoundingClientRect();
    var left = ev.clientX - pr.left + 14;
    if (left + tip.offsetWidth > pr.width) left = ev.clientX - pr.left - tip.offsetWidth - 14;
    tip.style.left = Math.max(0, left) + 'px';
    tip.style.top = Math.max(0, ev.clientY - pr.top - tip.offsetHeight - 10) + 'px';
  }
  hit.addEventListener('mousemove', move);
  hit.addEventListener('mouseleave', hide);
});
var btn = document.querySelector('.themebtn');
if (btn) btn.addEventListener('click', function () {
  var next = document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark';
  document.documentElement.setAttribute('data-theme', next);
  btn.textContent = next === 'dark' ? 'Light mode' : 'Dark mode';
});
"""


# --- assembly ----------------------------------------------------------------

def tile(k, v, unit="", d=""):
    u = f"<small>{html.escape(unit)}</small>" if unit else ""
    dd = f'<div class="d">{d}</div>' if d else ""
    return (f'<div class="tile"><div class="k">{html.escape(k)}</div>'
            f'<div class="v">{v}{u}</div>{dd}</div>')


def md_cell(v) -> str:
    """A bare pipe inside a cell ends the column early and silently mangles the
    whole table -- and 'mean |action|' is a phrase this report genuinely uses."""
    return str(v).replace("|", "\\|")


def md_table(rows, headers) -> str:
    out = ["| " + " | ".join(md_cell(h) for h in headers) + " |",
           "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(md_cell(v) for v in r) + " |")
    return "\n".join(out)


def build(run: Path, out: Path, md: Path = None, svg_dir: Path = None) -> None:
    cfg = json.loads((run / "config.json").read_text())
    log = load_log(run)
    trajs = load_trajectories(run)

    dt = cfg["env"]["dt"]
    ep_s = cfg["env"]["episode_seconds"]
    dec = cfg["env"]["control_decimation"]
    ceiling = ep_s / dt                     # +1 reward per plant step, upright
    steps_per_iter = cfg["training"]["steps_per_iteration"]
    n_iter = len(log)

    ev = [r for r in log if r.get("eval_return") is not None]
    best = max(ev, key=lambda r: r["eval_return"]) if ev else None
    solved = next((r for r in ev if r["eval_return"] >= 0.9 * ceiling), None)

    target_kl = cfg["ppo"]["target_kl"]
    full_len = ep_s / (dt * dec)

    # A KL spike is not a collapse -- the run can and does recover from one. The
    # README's collapse is specifically "episodes went from 800 steps to 10 and
    # never recovered", so require that it never recovers: the first iteration
    # after which mean_length stays under a quarter of a full episode for the
    # entire remainder of the run. Reported as None on a run that recovered,
    # which is the honest answer and stops the tile from crying wolf.
    collapse = None
    lens = [(r["iteration"], r.get("mean_length")) for r in log]
    for idx, (it, ln) in enumerate(lens):
        if it <= 50 or ln is None:
            continue
        rest = [v for _, v in lens[idx:] if v is not None]
        if rest and max(rest) < 0.25 * full_len:
            collapse = log[idx]
            break

    # Reported separately, because it is the diagnostic that moves first.
    kl_rows = [r for r in log if r.get("approx_kl") is not None]
    worst_kl = max(kl_rows, key=lambda r: r["approx_kl"]) if kl_rows else None
    early_stops = sum(1 for r in log
                      if r.get("epochs_run") is not None
                      and r["epochs_run"] < cfg["ppo"]["epochs"])

    met = {i: traj_metrics(rows) for i, rows in trajs.items()}
    best_i = int(best["iteration"]) if best else (max(trajs) if trajs else 0)
    bm = met.get(best_i, {})

    charts, ax = [], "iteration"

    def col(name):
        return [(r["iteration"], r.get(name)) for r in log]

    charts.append(Chart(
        "return", "Learning curve",
        "Sampled training return against the deterministic evaluation score. "
        "Evaluation uses the action mean, not a sample.",
        ax, "return per episode",
        [Series("training (sampled)", col("mean_return"), "s1"),
         Series("evaluation (greedy)", col("eval_return"), "s2")],
        [Rule(ceiling, f"ceiling {ceiling:,.0f}")],
        note="The gap between the two is the whole story of the collapse: a greedy policy "
             "can sit at ceiling while the stochastic policy it is actually training on dies early."))

    charts.append(Chart(
        "length", "Episode length",
        "Steps survived. Read alongside return -- a rising return with short episodes "
        "means the agent found a way to end early, not a way to balance.",
        ax, "policy steps",
        [Series("training (sampled)", col("mean_length"), "s1"),
         Series("evaluation (greedy)", col("eval_length"), "s2")],
        [Rule(ep_s / (dt * dec), "full episode")]))

    charts.append(Chart(
        "entropy", "Policy entropy",
        "Exploration left in the Gaussian head. Decay toward zero is what removes the "
        "policy's ability to absorb a large update.",
        ax, "nats", [Series("entropy", col("entropy"), "s1")]))

    charts.append(Chart(
        "kl", "Approximate KL per update",
        f"Trust region. The trainer early-stops the epoch loop when this crosses "
        f"{target_kl}.",
        ax, "approx KL", [Series("approx_kl", col("approx_kl"), "s1")],
        [Rule(target_kl, f"target {target_kl}")]))

    charts.append(Chart(
        "clip", "Clip fraction",
        "Share of sampled actions whose probability ratio hit the PPO clip. "
        "Sustained high values mean the step size is too large for the data.",
        ax, "fraction", [Series("clip_fraction", col("clip_fraction"), "s1")],
        [Rule(cfg["ppo"]["clip"], f"clip {cfg['ppo']['clip']}")]))

    charts.append(Chart(
        "epochs", "Epochs actually run",
        "Ten unless the KL early-stop fired. Every dip below ten is an update the "
        "trainer judged too aggressive to finish.",
        ax, "epochs", [Series("epochs_run", col("epochs_run"), "s1")],
        [Rule(cfg["ppo"]["epochs"], f"configured {cfg['ppo']['epochs']}")]))

    charts.append(Chart(
        "ploss", "Policy loss",
        "Negated surrogate, so descending is the policy improving. Near zero is "
        "expected -- it is a clipped ratio objective, not a fit error.",
        ax, "loss", [Series("policy_loss", col("policy_loss"), "s1")]))

    charts.append(Chart(
        "vloss", "Value loss",
        "Critic fit against the GAE returns. Falls as the value function learns the "
        "episode is worth roughly the ceiling.",
        ax, "loss", [Series("value_loss", col("value_loss"), "s1")]))

    # --- trajectory comparison ------------------------------------------------
    picks, seen = [], set()
    for cand in (min(trajs) if trajs else None,
                 int(solved["iteration"]) if solved else None,
                 best_i):
        if cand is not None and cand in trajs and cand not in seen:
            seen.add(cand)
            picks.append(cand)
    slots = ["s1", "s2", "s3"]

    def traj_series(field_fn, label_fmt="iteration {i}"):
        return [Series(label_fmt.format(i=i), [(r["t"], field_fn(r)) for r in trajs[i]],
                       slots[k % 3])
                for k, i in enumerate(picks)]

    if picks:
        charts.append(Chart(
            "tilt", "Tilt from upright",
            "Deterministic evaluation rollouts. The link starts hanging at 180 degrees; "
            "the task is to reach and hold 0.",
            "time (s)", "degrees from upright",
            traj_series(lambda r: math.degrees(abs(wrap(r["theta"])))),
            [Rule(15.0, "15 deg cone")]))

        half_rail = 0.5 * cfg["actuator"]["rail_length"]
        charts.append(Chart(
            "cart", "Cart position",
            "How much rail each policy spends. Leaving the rail ends the episode.",
            "time (s)", "metres from centre",
            traj_series(lambda r: r["x"]),
            [Rule(half_rail, "rail end"), Rule(-half_rail, "rail end")]))

        charts.append(Chart(
            "action", "Commanded acceleration",
            "Raw network output, logged before the environment clamps it. Rapid sign "
            "flips during the hold are a limit cycle at the control rate -- the "
            "behaviour that chews through a real belt.",
            "time (s)", "network output (pre-clamp)",
            traj_series(lambda r: r["action"]),
            [Rule(1.0, "clamp +1"), Rule(-1.0, "clamp -1")],
            note="The trainer logs <code>action_mean</code> straight off the network, so "
                 "excursions past +/-1 are real: an untrained policy asks for several times "
                 "the available acceleration and the env saturates it. Everything the plant "
                 "actually saw is inside the clamp lines."))

        pr = trajs[best_i]
        charts.append(Chart(
            "phase", "Phase portrait, best policy",
            f"Iteration {best_i}. Angle against angular rate, the full 8 s rollout: "
            "one sweep from hanging into the origin, then nothing.",
            "degrees from upright", "rad/s",
            [Series(f"iteration {best_i}",
                    [(math.degrees(wrap(r["theta"])), r["omega"]) for r in pr], "s1")],
            hover=False))

    # --- derived per-evaluation series ---------------------------------------
    def derived(key):
        return [(i, met[i].get(key)) for i in sorted(met)]

    charts.append(Chart(
        "swing", "Swing-up time",
        "First moment the tilt enters the 15 degree cone and never leaves it again. "
        "Blank where the policy never got there and stayed.",
        ax, "seconds", [Series("swing-up time", derived("swing_up_s"), "s1")]))

    charts.append(Chart(
        "finaltilt", "Tilt at the end of the episode",
        "Steady-state error after 8 s. This is the number that separates 'gets upright' "
        "from 'holds still'.",
        ax, "degrees", [Series("final tilt", derived("final_tilt_deg"), "s1")]))

    charts.append(Chart(
        "reversal", "Command reversals during the hold",
        "Share of consecutive control steps that flip the command sign, over the last "
        "2 s. High values are the limit cycle; it is invisible in the return.",
        ax, "percent of steps", [Series("reversal rate", derived("hold_reversal_pct"), "s1")]))

    charts.append(Chart(
        "effort", "Mean command magnitude during the hold",
        "Average |action| over the last 2 s. Driven to zero by the effort penalty, "
        "which is a hundred times smaller than the alignment term.",
        ax, "mean |action|", [Series("hold effort", derived("hold_mean_abs_action"), "s1")]))

    # --- page ----------------------------------------------------------------
    tiles = [
        tile("Best evaluation return",
             f"{best['eval_return']:,.0f}" if best else "&mdash;",
             f" / {ceiling:,.0f}",
             f"iteration {best_i}" if best else ""),
        tile("First past 90% of ceiling",
             f"{int(solved['iteration'])}" if solved else "never",
             "",
             f"scored {solved['eval_return']:,.0f}" if solved else "no evaluation cleared it"),
        tile("Swing-up time",
             fmt(bm["swing_up_s"]) if bm.get("swing_up_s") is not None else "&mdash;",
             " s", "hanging to inside 15 deg, and stays"),
        tile("Steady-state tilt",
             fmt(bm["final_tilt_deg"]) if bm.get("final_tilt_deg") is not None else "&mdash;",
             " deg", "at the end of the 8 s episode"),
        tile("Rail used",
             fmt(bm["peak_abs_x"]) if bm.get("peak_abs_x") is not None else "&mdash;",
             " m", f"of +/-{0.5 * cfg['actuator']['rail_length']:g} m available"),
        tile("Hold effort",
             fmt(bm["hold_mean_abs_action"]) if bm.get("hold_mean_abs_action") is not None else "&mdash;",
             "", "mean |action| over the last 2 s"),
        tile("Iterations", f"{n_iter:,}", "",
             f"{n_iter * steps_per_iter / 1e6:.2f} M policy steps, CPU only"),
        tile("Evaluations at ceiling",
             f"{sum(1 for r in ev if r['eval_return'] >= 0.9 * ceiling)} / {len(ev)}",
             "", "greedy score past 90% of the ceiling"),
    ]
    tiles.append(tile(
        "Unrecovered collapse",
        f"{int(collapse['iteration'])}" if collapse else "none",
        "",
        "episode length never recovers after this iteration" if collapse
        else "every dip in this run recovered"))
    if worst_kl:
        tiles.append(tile(
            "Worst update", f"{worst_kl['approx_kl']:.3f}", " KL",
            f"iteration {int(worst_kl['iteration'])}, against a {target_kl} target &middot; "
            f"{early_stops} early-stopped updates"))

    body = [
        '<div class="viz-root"><div class="wrap">',
        '<header><button class="themebtn" type="button">Dark mode</button>',
        "<h1>PPO training report</h1>",
        f'<p class="sub">N=1 swing-up and balance, {cfg["plant"]["links"][0]["length"]} m link on a '
        f'{cfg["actuator"]["rail_length"]} m rail. Policy at {1 / (dt * dec):.0f} Hz over a '
        f'{1 / dt:.0f} Hz plant.</p>',
        f'<p class="meta">Run directory <code>{html.escape(run.name)}</code> &middot; '
        f'{n_iter:,} iterations &middot; {len(ev)} evaluations &middot; '
        f'{len(trajs)} trajectory dumps &middot; seed {cfg["training"]["seed"]}</p></header>',
        "<h2>Headline</h2>",
        f'<div class="tiles">{"".join(tiles)}</div>',
        '<p class="prose" style="margin-top:18px">Every figure on this page is computed from '
        '<code>train_log.csv</code> and the <code>trajectory_*.csv</code> dumps in the run '
        'directory named above. Nothing is transcribed by hand, which is the point: the '
        'README quotes these numbers, and regenerating this file is how they are kept honest.</p>',
        "<h2>Training</h2>",
    ]
    for c in charts[:8]:
        body.append(render_chart(c))
        body.append(render_table(c))
    body.append("<h2>Behaviour</h2>")
    for c in charts[8:12]:
        body.append(render_chart(c))
        body.append(render_table(c))
    body.append("<h2>Derived per evaluation</h2>")
    for c in charts[12:]:
        body.append(render_chart(c))
        body.append(render_table(c))
    body.append("</div></div>")

    page = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>PPO training report</title>\n<style>" + css() + "</style>\n</head>\n<body>\n"
        + "\n".join(body)
        + "\n<script>\nconst SVG_NS = \"" + SVG_NS + "\";\n" + HOVER_JS + "\n</script>\n"
        "</body>\n</html>\n"
    )
    # Same rule as tests/plot_html.hpp: emitted text is ASCII only. A stray em-dash
    # that survives one round-trip through a non-UTF-8 tool renders as mojibake in
    # the report and is invisible in review. Fail loudly at generation instead.
    if not page.isascii():
        bad = sorted({c for c in page if ord(c) > 127})
        raise SystemExit(
            "non-ASCII in generated report: "
            + ", ".join(f"U+{ord(c):04X} ({c!r})" for c in bad[:12])
        )
    out.write_text(page, encoding="ascii", newline="\n")

    # --- Markdown twin -------------------------------------------------------
    if md:
        svg_dir = svg_dir or (md.parent / "media" / "training-report")
        svg_dir.mkdir(parents=True, exist_ok=True)
        rel = Path(os.path.relpath(svg_dir, md.parent)).as_posix()

        m = [
            "# PPO training report",
            "",
            f"N=1 swing-up and balance, {cfg['plant']['links'][0]['length']} m link on a "
            f"{cfg['actuator']['rail_length']} m rail. Policy at {1 / (dt * dec):.0f} Hz "
            f"over a {1 / dt:.0f} Hz plant.",
            "",
            f"> Generated by `tools/training_report.py` from run `{run.name}` "
            f"({n_iter:,} iterations, {len(ev)} evaluations, seed "
            f"{cfg['training']['seed']}). Every figure is computed from "
            f"`train_log.csv` and the `trajectory_*.csv` dumps -- nothing is "
            f"transcribed by hand. The interactive version, with hover readouts and "
            f"a table view under every chart, is "
            f"[`training-report.html`](training-report.html).",
            "",
            "## Headline",
            "",
        ]

        head = [
            ("Best evaluation return", f"**{best['eval_return']:,.1f}** of {ceiling:,.0f}"
                if best else "--", f"iteration {best_i}" if best else ""),
            ("First past 90% of ceiling", f"**{int(solved['iteration'])}**" if solved else "never",
                f"scored {solved['eval_return']:,.1f}" if solved else ""),
            ("Swing-up time", f"**{bm['swing_up_s']:.2f} s**" if bm.get("swing_up_s") is not None else "--",
                "hanging to inside 15 deg, and stays"),
            ("Steady-state tilt", f"**{bm['final_tilt_deg']:.4f} deg**" if bm.get("final_tilt_deg") is not None else "--",
                "at the end of the 8 s episode"),
            ("Rail used", f"**{bm['peak_abs_x']:.3f} m**" if bm.get("peak_abs_x") is not None else "--",
                f"of +/-{0.5 * cfg['actuator']['rail_length']:g} m available"),
            ("Hold effort", f"**{bm['hold_mean_abs_action']:.2e}**" if bm.get("hold_mean_abs_action") is not None else "--",
                "mean |action| over the last 2 s"),
            ("Cost", f"**{n_iter:,} iterations**",
                f"{n_iter * steps_per_iter / 1e6:.2f} M policy steps, CPU only"),
            ("Evaluations at ceiling",
                f"**{sum(1 for r in ev if r['eval_return'] >= 0.9 * ceiling)} of {len(ev)}**",
                "greedy score past 90% of the ceiling"),
            ("Unrecovered collapse",
                f"**iteration {int(collapse['iteration'])}**" if collapse else "**none**",
                "episode length never recovers after this point" if collapse
                else "every dip in this run recovered"),
            ("Worst update", f"**{worst_kl['approx_kl']:.3f} KL**" if worst_kl else "--",
                f"iteration {int(worst_kl['iteration'])} against a {target_kl} target; "
                f"{early_stops} early-stopped updates" if worst_kl else ""),
        ]
        m += [md_table(head, ["", "value", "note"]), ""]

        sections = [("Training", charts[:8]), ("Behaviour", charts[8:12]),
                    ("Derived per evaluation", charts[12:])]
        for name, group in sections:
            m += [f"## {name}", ""]
            for c in group:
                svg = render_standalone_svg(c)
                if not svg:
                    continue
                (svg_dir / f"{c.key}.svg").write_text(svg, encoding="ascii", newline="\n")
                m += [f"### {c.title}", "", c.subtitle, "",
                      f"![{c.title}]({rel}/{c.key}.svg)", ""]
                if c.note:
                    m += ["> " + re.sub(r"<[^>]+>", "`", c.note).replace("``", "`"), ""]

        m += ["## Every evaluation", "",
              "The deterministic evaluation score, run every "
              f"{cfg['training']['evaluate_every']} iterations with `action_mean` "
              "rather than a sample.", ""]
        erows = []
        for r in ev:
            i = int(r["iteration"])
            mm = met.get(i, {})
            erows.append((
                i, f"{r['eval_return']:,.1f}", f"{r['eval_length']:.0f}",
                f"{mm['swing_up_s']:.2f}" if mm.get("swing_up_s") is not None else "--",
                f"{mm['final_tilt_deg']:.3f}" if mm.get("final_tilt_deg") is not None else "--",
                f"{mm['peak_abs_x']:.3f}" if mm.get("peak_abs_x") is not None else "--",
            ))
        m += [md_table(erows, ["iteration", "return", "length", "swing-up (s)",
                               "final tilt (deg)", "peak |x| (m)"]), ""]

        text = "\n".join(m) + "\n"
        if not text.isascii():
            bad = sorted({ch for ch in text if ord(ch) > 127})
            raise SystemExit("non-ASCII in Markdown report: "
                             + ", ".join(f"U+{ord(ch):04X}" for ch in bad[:12]))
        md.write_text(text, encoding="ascii", newline="\n")
        print(f"wrote {md}  (+{len(list(svg_dir.glob('*.svg')))} SVGs in {svg_dir})")

    print(f"wrote {out}  ({out.stat().st_size / 1024:.0f} KB)")
    print(f"  iterations       {n_iter}")
    print(f"  evaluations      {len(ev)}")
    if best:
        print(f"  best eval        {best['eval_return']:.2f} of {ceiling:.0f} at iteration {best_i}")
    if solved:
        print(f"  first >=90%      iteration {int(solved['iteration'])} ({solved['eval_return']:.2f})")
    else:
        print("  first >=90%      never")
    print(f"  collapse         "
          + (f"iteration {int(collapse['iteration'])} (unrecovered)" if collapse
             else "none -- every dip recovered"))
    if worst_kl:
        print(f"  worst kl         {worst_kl['approx_kl']:.4f} at iteration "
              f"{int(worst_kl['iteration'])} (target {target_kl}), "
              f"{early_stops} early-stopped updates")
    for k in ("swing_up_s", "final_tilt_deg", "peak_abs_x",
              "hold_mean_abs_action", "hold_reversal_pct"):
        if bm.get(k) is not None:
            print(f"  {k:<16} {bm[k]:.4g}")


def clip_table(run: Path, rail_half: float, episode_s: float) -> None:
    """Caption numbers for the README clips, read off the MuJoCo rollouts.

    The trajectory_*.csv dumps come from core::Plant; the GIFs come from
    policy_rollout.py driving MuJoCo closed loop, and the two do not agree
    exactly -- that disagreement is the transfer test and is the point. So the
    captions have to be measured on the clips themselves, not copied from the
    training log.
    """
    files = sorted(run.glob("closed_*.csv"),
                   key=lambda f: int(re.search(r"closed_(\d+)", f.name).group(1)))
    if not files:
        raise SystemExit(f"no closed_*.csv in {run} -- run tools/policy_rollout.py first")

    rows = []
    for f in files:
        it = int(re.search(r"closed_(\d+)", f.name).group(1))
        with f.open(newline="") as fh:
            raw = list(csv.DictReader(fh))
        key = "theta" if "theta" in raw[0] else "theta0"
        rec = [{"t": float(r["t"]), "x": float(r["x"]),
                "theta": float(r[key]),
                "omega": 0.0, "action": 0.0} for r in raw]
        m = traj_metrics(rec)
        # The rollout stops on the step that crosses the rail, so the last logged
        # |x| is always a hair *under* half_rail and comparing against it misses.
        # Ending early is the reliable signal -- but a full-length clip also ends
        # one sample short of episode_seconds, so the tolerance has to be the
        # sample spacing rather than an epsilon.
        spacing = (rec[1]["t"] - rec[0]["t"]) if len(rec) > 1 else 0.0
        off = m["duration_s"] < episode_s - 2.0 * spacing
        rows.append((
            it,
            f"{m['swing_up_s']:.2f} s" if m["swing_up_s"] is not None else "never",
            f"{m['final_tilt_deg']:.3g} deg",
            f"{m['peak_abs_x']:.2f} m",
            f"{m['duration_s']:.2f} s" + (" (off rail)" if off else ""),
        ))

    print(md_table(rows, ["iteration", "upright at", "final tilt",
                          "peak |x|", "survived"]))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path, help="run directory holding train_log.csv")
    ap.add_argument("--clips", action="store_true",
                    help="instead of a report, print caption numbers measured off the "
                         "closed_<iter>.csv MuJoCo rollouts in the run directory")
    ap.add_argument("-o", "--out", type=Path, default=None,
                    help="output HTML (default: <run>/training_report.html)")
    ap.add_argument("--md", type=Path, default=None,
                    help="also write a Markdown twin here, with one standalone SVG "
                         "per chart beside it")
    ap.add_argument("--svg-dir", type=Path, default=None,
                    help="where the Markdown report's SVGs go "
                         "(default: <md-dir>/media/training-report)")
    a = ap.parse_args()
    if a.clips:
        cfg = json.loads((a.run / "config.json").read_text())
        clip_table(a.run, 0.5 * cfg["actuator"]["rail_length"], cfg["env"]["episode_seconds"])
        return
    build(a.run, a.out or (a.run / "training_report.html"), a.md, a.svg_dir)


if __name__ == "__main__":
    main()
