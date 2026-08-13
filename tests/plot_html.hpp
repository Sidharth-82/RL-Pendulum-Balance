#pragma once

// Test-only plotting: emits a self-contained HTML fragment with inline SVG line
// charts, no scripts fetched and no plotting library.
//
// Lives in tests/ rather than core/ on purpose -- core links Eigen and nothing
// else, and a validation report has no business on the Pi. Kept dependency-free
// (no Python, no gnuplot) so `ctest` produces the picture on any machine that can
// build the project at all.
//
// ASCII only in all emitted text: the fragment carries no <meta charset>, so a
// stray Unicode theta would render as mojibake when the file is opened directly
// from disk.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <format>
#include <limits>
#include <string>
#include <vector>

namespace plot {

struct Series {
    std::string label;
    std::vector<double> y;
};

// One chart. All series share the page-level x samples.
struct Chart {
    std::string title;
    std::string subtitle;
    std::string y_label;
    std::vector<Series> series;
};

namespace detail {

// Heckbert's "nice numbers" -- axis bounds land on 1/2/5 x 10^k so tick labels
// read as round numbers rather than as artefacts of the data range.
inline double nice_num(double x, bool round_to_nearest) {
    if (!(x > 0.0)) {
        return 1.0;
    }
    const double exponent = std::floor(std::log10(x));
    const double fraction = x / std::pow(10.0, exponent);

    double nice = 10.0;
    if (round_to_nearest) {
        if (fraction < 1.5) nice = 1.0;
        else if (fraction < 3.0) nice = 2.0;
        else if (fraction < 7.0) nice = 5.0;
    } else {
        if (fraction <= 1.0) nice = 1.0;
        else if (fraction <= 2.0) nice = 2.0;
        else if (fraction <= 5.0) nice = 5.0;
    }
    return nice * std::pow(10.0, exponent);
}

struct Axis {
    double lo = 0.0;
    double hi = 1.0;
    double step = 1.0;
};

inline Axis nice_axis(double lo, double hi, int max_ticks = 6) {
    if (!(hi > lo)) {
        const double pad = std::max(std::abs(hi), 1.0) * 0.5;
        lo = hi - pad;
        hi = hi + pad;
    }
    const double range = nice_num(hi - lo, false);
    const double step = nice_num(range / (max_ticks - 1), true);
    return {std::floor(lo / step) * step, std::ceil(hi / step) * step, step};
}

// Tick labels are sized by the tick *step*, not by the value, so an axis reading
// 0 / 2e-9 / 4e-9 does not print fifteen zeros.
inline std::string fmt_tick(double v, double step) {
    if (std::abs(v) < step * 1e-9) {
        return "0";
    }
    const double magnitude = std::max(std::abs(v), step);
    if (magnitude >= 1e-2 && magnitude < 1e5) {
        const int decimals = std::clamp(static_cast<int>(std::ceil(-std::log10(step))), 0, 6);
        return std::format("{:.{}f}", v, decimals);
    }
    return std::format("{:.1e}", v);
}

inline std::string escape(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (const char c : s) {
        switch (c) {
            case '&': out += "&amp;"; break;
            case '<': out += "&lt;"; break;
            case '>': out += "&gt;"; break;
            case '"': out += "&quot;"; break;
            default: out += c;
        }
    }
    return out;
}

inline std::string render_chart(int index, const Chart& chart, const std::vector<double>& x) {
    constexpr double kWidth = 900.0;
    constexpr double kHeight = 280.0;
    constexpr double kLeft = 78.0;
    constexpr double kRight = 16.0;
    constexpr double kTop = 12.0;
    constexpr double kBottom = 34.0;

    const double plot_w = kWidth - kLeft - kRight;
    const double plot_h = kHeight - kTop - kBottom;

    double y_lo = std::numeric_limits<double>::infinity();
    double y_hi = -std::numeric_limits<double>::infinity();
    for (const Series& s : chart.series) {
        for (const double v : s.y) {
            if (std::isfinite(v)) {
                y_lo = std::min(y_lo, v);
                y_hi = std::max(y_hi, v);
            }
        }
    }
    const Axis ay = nice_axis(y_lo, y_hi);
    const Axis ax = nice_axis(x.front(), x.back(), 7);

    const auto px = [&](double v) { return kLeft + (v - ax.lo) / (ax.hi - ax.lo) * plot_w; };
    const auto py = [&](double v) { return kTop + (ay.hi - v) / (ay.hi - ay.lo) * plot_h; };

    std::string svg = std::format(
        "<svg class=\"chart\" viewBox=\"0 0 {:.0f} {:.0f}\" role=\"img\" "
        "aria-label=\"{}\" data-chart=\"{}\">",
        kWidth, kHeight, escape(chart.title), index);

    // Horizontal gridlines only -- vertical ones add ink without adding meaning
    // on a time axis that already has tick labels.
    for (double v = ay.lo; v <= ay.hi + ay.step * 0.5; v += ay.step) {
        svg += std::format(
            "<line class=\"grid\" x1=\"{:.1f}\" y1=\"{:.1f}\" x2=\"{:.1f}\" y2=\"{:.1f}\"/>"
            "<text class=\"tick tick-y\" x=\"{:.1f}\" y=\"{:.1f}\">{}</text>",
            kLeft, py(v), kWidth - kRight, py(v), kLeft - 8.0, py(v) + 4.0,
            escape(fmt_tick(v, ay.step)));
    }

    for (double v = ax.lo; v <= ax.hi + ax.step * 0.5; v += ax.step) {
        if (v < x.front() - ax.step * 0.01 || v > x.back() + ax.step * 0.01) {
            continue;
        }
        svg += std::format("<text class=\"tick tick-x\" x=\"{:.1f}\" y=\"{:.1f}\">{}</text>",
                           px(v), kHeight - 12.0, escape(fmt_tick(v, ax.step)));
    }

    // Zero line, when the range straddles it: the energy-drift charts are read
    // against zero, so it gets a stronger rule than the grid.
    if (ay.lo < 0.0 && ay.hi > 0.0) {
        svg += std::format("<line class=\"axis\" x1=\"{:.1f}\" y1=\"{:.1f}\" x2=\"{:.1f}\" "
                           "y2=\"{:.1f}\"/>",
                           kLeft, py(0.0), kWidth - kRight, py(0.0));
    }
    svg += std::format("<line class=\"axis\" x1=\"{:.1f}\" y1=\"{:.1f}\" x2=\"{:.1f}\" "
                       "y2=\"{:.1f}\"/>",
                       kLeft, kTop, kLeft, kTop + plot_h);

    for (std::size_t si = 0; si < chart.series.size(); ++si) {
        std::string points;
        points.reserve(chart.series[si].y.size() * 14);
        for (std::size_t i = 0; i < x.size() && i < chart.series[si].y.size(); ++i) {
            points += std::format("{:.2f},{:.2f} ", px(x[i]), py(chart.series[si].y[i]));
        }
        svg += std::format("<polyline class=\"line s{}\" points=\"{}\"/>", si + 1, points);
    }

    svg += std::format(
        "<line class=\"crosshair\" x1=\"0\" y1=\"{:.1f}\" x2=\"0\" y2=\"{:.1f}\" "
        "style=\"display:none\"/>"
        "<rect class=\"hit\" x=\"{:.1f}\" y=\"{:.1f}\" width=\"{:.1f}\" height=\"{:.1f}\"/>",
        kTop, kTop + plot_h, kLeft, kTop, plot_w, plot_h);
    svg += "</svg>";

    std::string html = "<figure class=\"figure\">";
    html += std::format("<figcaption><h2>{}</h2><p>{}</p></figcaption>",
                        escape(chart.title), escape(chart.subtitle));
    html += std::format("<div class=\"y-label\">{}</div>", escape(chart.y_label));
    html += std::format("<div class=\"plot\" data-plot=\"{}\">{}<div class=\"tip\" "
                        "style=\"display:none\"></div></div>",
                        index, svg);
    html += "<div class=\"x-label\">time (s)</div>";
    html += "</figure>";
    return html;
}

}  // namespace detail

// Series labels are shared across charts, so the legend is emitted once for the
// whole page rather than repeated under each figure.
inline std::string page(const std::string& title, const std::string& intro_html,
                        const std::vector<double>& x, const std::vector<Chart>& charts,
                        const std::string& tail_html) {
    const std::vector<std::string> light = {"#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"};
    const std::vector<std::string> dark = {"#3987e5", "#d95926", "#199e70", "#c98500", "#d55181"};

    std::string series_light;
    std::string series_dark;
    for (std::size_t i = 0; i < light.size(); ++i) {
        series_light += std::format("--series-{}:{};", i + 1, light[i]);
        series_dark += std::format("--series-{}:{};", i + 1, dark[i]);
    }

    std::string out = std::format("<title>{}</title>\n", detail::escape(title));

    out += std::format(
        "<style>\n"
        ":root{{color-scheme:light;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;"
        "--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;"
        "--border:rgba(11,11,11,0.10);--good:#0ca30c;--bad:#d03b3b;{}}}\n"
        "@media (prefers-color-scheme:dark){{:root:where(:not([data-theme=\"light\"])){{"
        "color-scheme:dark;--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;--ink-2:#c3c2b7;"
        "--muted:#898781;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,0.10);"
        "--good:#0ca30c;--bad:#d03b3b;{}}}}}\n"
        ":root[data-theme=\"dark\"]{{color-scheme:dark;--surface:#1a1a19;--plane:#0d0d0d;"
        "--ink:#fff;--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;"
        "--border:rgba(255,255,255,0.10);--good:#0ca30c;--bad:#d03b3b;{}}}\n",
        series_light, series_dark, series_dark);

    out +=
        "body{background:var(--plane);color:var(--ink);margin:0;padding:32px 20px 64px;"
        "font-family:system-ui,-apple-system,\"Segoe UI\",sans-serif;line-height:1.55;}\n"
        ".wrap{max-width:1000px;margin:0 auto;}\n"
        "h1{font-size:1.6rem;margin:0 0 4px;letter-spacing:-0.01em;}\n"
        "h2{font-size:1.02rem;margin:0;letter-spacing:-0.005em;}\n"
        "p{color:var(--ink-2);margin:6px 0;}\n"
        ".lede{max-width:70ch;}\n"
        ".verdict{display:inline-flex;align-items:center;gap:8px;font-weight:600;"
        "border:1px solid var(--border);border-radius:8px;padding:8px 14px;margin:14px 0 8px;"
        "background:var(--surface);}\n"
        ".dot{width:10px;height:10px;border-radius:50%;display:inline-block;}\n"
        ".pass{color:var(--good);} .pass .dot{background:var(--good);}\n"
        ".fail{color:var(--bad);} .fail .dot{background:var(--bad);}\n"
        ".legend{display:flex;flex-wrap:wrap;gap:8px 18px;margin:18px 0 6px;"
        "color:var(--ink-2);font-size:0.86rem;}\n"
        ".legend span{display:inline-flex;align-items:center;gap:7px;}\n"
        ".swatch{width:18px;height:3px;border-radius:2px;display:inline-block;}\n"
        ".figure{background:var(--surface);border:1px solid var(--border);border-radius:12px;"
        "margin:14px 0;padding:14px 16px 8px;display:grid;"
        "grid-template-columns:auto 1fr;grid-template-areas:'cap cap' 'ylab plot' '. xlab';"
        "align-items:center;}\n"
        "figcaption{grid-area:cap;margin-bottom:6px;}\n"
        "figcaption p{font-size:0.84rem;margin:2px 0 0;}\n"
        ".y-label{grid-area:ylab;writing-mode:vertical-rl;transform:rotate(180deg);"
        "color:var(--muted);font-size:0.76rem;justify-self:center;padding-right:2px;}\n"
        ".x-label{grid-area:xlab;color:var(--muted);font-size:0.76rem;text-align:center;"
        "padding-bottom:6px;}\n"
        ".plot{grid-area:plot;position:relative;overflow-x:auto;}\n"
        ".chart{width:100%;min-width:520px;height:auto;display:block;overflow:visible;}\n"
        ".grid{stroke:var(--grid);stroke-width:1;}\n"
        ".axis{stroke:var(--axis);stroke-width:1;}\n"
        ".crosshair{stroke:var(--axis);stroke-width:1;stroke-dasharray:3 3;}\n"
        ".tick{fill:var(--muted);font-size:11px;font-variant-numeric:tabular-nums;}\n"
        ".tick-y{text-anchor:end;} .tick-x{text-anchor:middle;}\n"
        ".line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round;}\n"
        ".hit{fill:transparent;}\n"
        ".s1{stroke:var(--series-1);} .s2{stroke:var(--series-2);} .s3{stroke:var(--series-3);}\n"
        ".s4{stroke:var(--series-4);} .s5{stroke:var(--series-5);}\n"
        ".tip{position:absolute;pointer-events:none;background:var(--surface);color:var(--ink);"
        "border:1px solid var(--border);border-radius:8px;padding:8px 10px;font-size:0.78rem;"
        "box-shadow:0 4px 14px rgba(0,0,0,0.14);white-space:nowrap;z-index:2;"
        "font-variant-numeric:tabular-nums;}\n"
        ".tip b{display:block;margin-bottom:4px;font-weight:600;}\n"
        ".tip i{width:9px;height:3px;border-radius:2px;display:inline-block;"
        "margin-right:6px;vertical-align:middle;}\n"
        "table{border-collapse:collapse;width:100%;font-size:0.84rem;margin-top:10px;"
        "font-variant-numeric:tabular-nums;background:var(--surface);}\n"
        "th,td{text-align:right;padding:7px 10px;border-bottom:1px solid var(--border);}\n"
        "th:first-child,td:first-child{text-align:left;}\n"
        "th{color:var(--ink-2);font-weight:600;}\n"
        "code{background:var(--surface);border:1px solid var(--border);border-radius:4px;"
        "padding:1px 5px;font-size:0.86em;}\n"
        "</style>\n";

    out += "<div class=\"wrap\">";
    out += std::format("<h1>{}</h1>", detail::escape(title));
    out += intro_html;

    out += "<div class=\"legend\">";
    if (!charts.empty()) {
        for (std::size_t i = 0; i < charts.front().series.size(); ++i) {
            out += std::format("<span><i class=\"swatch\" style=\"background:var(--series-{})\">"
                               "</i>{}</span>",
                               i + 1, detail::escape(charts.front().series[i].label));
        }
    }
    out += "</div>";

    for (std::size_t i = 0; i < charts.size(); ++i) {
        out += detail::render_chart(static_cast<int>(i), charts[i], x);
    }

    out += tail_html;
    out += "</div>\n";

    // Hover data, emitted once, in the same order the charts were rendered.
    out += "<script type=\"application/json\" id=\"plotdata\">{\"x\":[";
    for (std::size_t i = 0; i < x.size(); ++i) {
        out += std::format("{}{:.4f}", i ? "," : "", x[i]);
    }
    out += "],\"charts\":[";
    for (std::size_t ci = 0; ci < charts.size(); ++ci) {
        out += std::format("{}{{\"unit\":\"{}\",\"series\":[", ci ? "," : "",
                           detail::escape(charts[ci].y_label));
        for (std::size_t si = 0; si < charts[ci].series.size(); ++si) {
            out += std::format("{}{{\"label\":\"{}\",\"y\":[", si ? "," : "",
                               detail::escape(charts[ci].series[si].label));
            const std::vector<double>& ys = charts[ci].series[si].y;
            for (std::size_t i = 0; i < ys.size(); ++i) {
                out += std::format("{}{:.6g}", i ? "," : "", ys[i]);
            }
            out += "]}";
        }
        out += "]}";
    }
    out += "]}</script>\n";

    out +=
        "<script>\n"
        "(function(){\n"
        "  var data = JSON.parse(document.getElementById('plotdata').textContent);\n"
        "  document.querySelectorAll('.plot').forEach(function(box){\n"
        "    var ci = +box.dataset.plot, svg = box.querySelector('svg');\n"
        "    var hair = svg.querySelector('.crosshair'), hit = svg.querySelector('.hit');\n"
        "    var tip = box.querySelector('.tip');\n"
        "    var hx = +hit.getAttribute('x'), hw = +hit.getAttribute('width');\n"
        "    function hide(){ hair.style.display='none'; tip.style.display='none'; }\n"
        "    box.addEventListener('mouseleave', hide);\n"
        "    box.addEventListener('mousemove', function(ev){\n"
        "      var r = svg.getBoundingClientRect();\n"
        "      var vb = svg.viewBox.baseVal;\n"
        "      var ux = (ev.clientX - r.left) / r.width * vb.width;\n"
        "      var t = (ux - hx) / hw;\n"
        "      if (t < 0 || t > 1) { hide(); return; }\n"
        "      var i = Math.round(t * (data.x.length - 1));\n"
        "      var sx = hx + i / (data.x.length - 1) * hw;\n"
        "      hair.setAttribute('x1', sx); hair.setAttribute('x2', sx);\n"
        "      hair.style.display = '';\n"
        "      var rows = '<b>t = ' + data.x[i].toFixed(3) + ' s</b>';\n"
        "      data.charts[ci].series.forEach(function(s, k){\n"
        "        var v = s.y[i];\n"
        "        var txt = (Math.abs(v) >= 1e-3 || v === 0) ? v.toFixed(4) : v.toExponential(2);\n"
        "        rows += '<div><i style=\"background:var(--series-' + (k+1) + ')\"></i>'\n"
        "              + s.label + ': ' + txt + '</div>';\n"
        "      });\n"
        "      tip.innerHTML = rows;\n"
        "      tip.style.display = '';\n"
        "      var pxc = sx / vb.width * box.clientWidth;\n"
        "      var flip = pxc > box.clientWidth - tip.offsetWidth - 24;\n"
        "      tip.style.left = (flip ? pxc - tip.offsetWidth - 14 : pxc + 14) + 'px';\n"
        "      tip.style.top = '8px';\n"
        "    });\n"
        "  });\n"
        "})();\n"
        "</script>\n";

    return out;
}

}  // namespace plot
