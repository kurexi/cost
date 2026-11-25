#!/usr/bin/env python3
"""The README's charts, drawn from the graded runs as SVG.

    python3 charts.py                 # rewrite every chart in charts/

Each chart is written twice, `<name>-light.svg` and `<name>-dark.svg`, and the
README shows the one that matches the reader's theme. The numbers come from
each run's `grade.json` in the records folder (`$SPENDING_EVAL_RECORDS`,
like board.py), so run `board.py --publish` first when a run has changed.
Standard library only:

    SPENDING_EVAL_RECORDS=../eval-records python3 charts.py

- `path`: the shipped model's path from Qwen3.5-0.8B as released, through
  distillation (SFT r3) and reinforcement learning (GRPO r2), to the Q8_0
  GGUF: records right on `dev` and `dev-novel`, and the wrong records it
  would file unreviewed.
- `tools`: tool calls a case on `dev`, split into useful and wasted, for the
  same steps, the teacher and the 4-bit file.
- `fields`: each field of the record right on `dev`, for the teacher, the
  0.8B as released and the shipped file.
- `languages`: records right on `dev` by the content's language, for the
  same three.
"""

import json
import os
from xml.sax.saxutils import escape

from suite import RUNS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "charts")

FONT = 'system-ui, -apple-system, "Segoe UI", Helvetica, Arial, sans-serif'
# The validated default categorical palette (slots 1-3), stepped per theme,
# on GitHub's own page colors.
THEMES = {
    "light": {"surface": "#ffffff", "ink": "#191919", "ink2": "#666666", "grid": "#e7e7e7", "axis": "#c2c2c2",
              "s1": "#257de0", "s2": "#f47230", "s3": "#00b894"},
    "dark": {"surface": "#0d1117", "ink": "#efefef", "ink2": "#9c9c9c", "grid": "#262c36", "axis": "#474747",
             "s1": "#338ced", "s2": "#e76624", "s3": "#00a888"},
}

TEACHER = "qwen3.8-27b-fp8-v5"
RELEASED = "qwen3.5-0.8b-q8_0"
SHIPPED = "qwen3.5-0.8b-grpo-r2-q8_0"
FOUR_BIT = "qwen3.5-0.8b-grpo-r2-q4_k_m"
PATH = (
    ("As released", "fp16", "qwen3.5-0.8b-fp16"),
    ("Distilled", "SFT r3, bf16", "qwen3.5-0.8b-sft-r3-bf16"),
    ("+ RL", "GRPO r2, bf16", "qwen3.5-0.8b-grpo-r2-bf16"),
    ("Shipped", "Q8_0 GGUF", SHIPPED),
)
COMPARED = (("27B teacher", TEACHER, "s3"), ("0.8B as released", RELEASED, "s2"), ("0.8B trained, shipped", SHIPPED, "s1"))
FIELDS = (
    ("Spending or not", "is_spending"), ("Amount", "amount"), ("Currency", "currency"), ("Merchant", "merchant"),
    ("Date", "date"), ("Payment method", "instrument"), ("Category", "category"), ("Tags", "tags"),
    ("All of the record's core", "record_core"),
)
LANGUAGES = {"en": "English", "zh": "Chinese", "fr": "French", "de": "German", "es": "Spanish", "ja": "Japanese"}


def summary(run):
    path = os.path.join(RUNS, run, "grade.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)["summary"]


def pct(value):
    return f"{value * 100:.0f}%"


class Svg:
    def __init__(self, width, height, theme, title, subtitle):
        self.w, self.h, self.t = width, height, THEMES[theme]
        self.parts = [f'<rect width="{width}" height="{height}" rx="8" fill="{self.t["surface"]}"/>']
        self.title = title
        self.text(24, 34, title, size=16, weight=600)
        self.text(24, 56, subtitle, size=13, color="ink2")

    def text(self, x, y, s, size=12, color="ink", anchor="start", weight=400):
        self.parts.append(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" font-weight="{weight}" '
                          f'fill="{self.t[color]}" text-anchor="{anchor}">{escape(s)}</text>')

    def line(self, x1, y1, x2, y2, color="grid", width=1, dash=None):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                          f'stroke="{self.t[color]}" stroke-width="{width}"{extra}/>')

    def polyline(self, points, color):
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        self.parts.append(f'<polyline points="{d}" fill="none" stroke="{self.t[color]}" stroke-width="2" '
                          'stroke-linejoin="round" stroke-linecap="round"/>')

    def dot(self, x, y, color, r=5):
        self.parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{self.t[color]}" '
                          f'stroke="{self.t["surface"]}" stroke-width="2"/>')

    def bar(self, x, y, w, h, color, round_end=True):
        """A horizontal bar from x, with a 4px rounded data-end and a square base."""
        if w <= 0:
            return
        r = min(4, w / 2, h / 2) if round_end else 0
        d = (f"M{x:.1f},{y:.1f} H{x + w - r:.1f} Q{x + w:.1f},{y:.1f} {x + w:.1f},{y + r:.1f} "
             f"V{y + h - r:.1f} Q{x + w:.1f},{y + h:.1f} {x + w - r:.1f},{y + h:.1f} H{x:.1f} Z")
        self.parts.append(f'<path d="{d}" fill="{self.t[color]}"/>')

    def legend(self, x, y, items):
        """items: (label, color, kind) with kind 'dot', 'line' or 'box'."""
        for label, color, kind in items:
            if kind == "line":
                self.line(x, y - 4, x + 18, y - 4, color=color, width=2)
                self.dot(x + 9, y - 4, color, r=4)
                x += 24
            elif kind == "box":
                self.parts.append(f'<rect x="{x}" y="{y - 10}" width="12" height="12" rx="3" fill="{self.t[color]}"/>')
                x += 18
            else:
                self.dot(x + 5, y - 4, color, r=5)
                x += 16
            self.text(x, y, label, color="ink2")
            x += text_width(label, 12) + 20

    def write(self, path, description):
        body = "\n".join(self.parts)
        svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
               f'viewBox="0 0 {self.w} {self.h}" font-family=\'{FONT}\' role="img">\n'
               f"<title>{escape(self.title)}</title>\n<desc>{escape(description)}</desc>\n{body}\n</svg>\n")
        with open(path, "w", encoding="utf-8") as f:
            f.write(svg)


def text_width(s, size):
    return len(s) * size * 0.56


def path_chart(theme):
    runs = [(label, detail, summary(run), summary(f"{run}-dev-novel")) for label, detail, run in PATH]
    teacher = summary(TEACHER)
    svg = Svg(760, 400, theme, "Distillation makes it right; RL makes it careful",
              "The shipped model at each step, on the 280 dev cases and the dev-novel ones (names never seen)")
    left, right, top, bottom = 64, 600, 104, 330
    y_of = lambda v: bottom - v * (bottom - top)
    x_of = lambda i: left + 40 + i * (right - left - 80) / (len(runs) - 1)
    for tick in (0, 0.25, 0.5, 0.75, 1):
        svg.line(left, y_of(tick), right, y_of(tick), color="axis" if tick == 0 else "grid")
        svg.text(left - 10, y_of(tick) + 4, pct(tick), color="ink2", anchor="end")
    if teacher:
        value = teacher["overall"]["record_core"]
        svg.line(left, y_of(value), right, y_of(value), color="ink2", dash="4 4")
        svg.text(right + 10, y_of(value) + 4, f"27B teacher, dev {pct(value)}", color="ink2")
    for i, (label, detail, *_ ) in enumerate(runs):
        svg.text(x_of(i), bottom + 22, label, anchor="middle")
        svg.text(x_of(i), bottom + 38, detail, color="ink2", anchor="middle")

    series = (
        ("Records right on dev", "s1", lambda dev, novel: dev["overall"]["record_core"], -12),
        ("Records right on dev-novel", "s3", lambda dev, novel: novel["overall"]["record_core"] if novel else None, 20),
        ("Wrong records it would file unreviewed", "s2", lambda dev, novel: dev["overall"]["false_approval"], -12),
    )
    for name, color, value_of, offset in series:
        points = [(x_of(i), y_of(v), v) for i, (_, _, dev, novel) in enumerate(runs)
                  if dev and (v := value_of(dev, novel)) is not None]
        svg.polyline([(x, y) for x, y, _ in points], color)
        for x, y, v in points:
            svg.dot(x, y, color)
            svg.text(x, y + offset, pct(v), size=12, color="ink", anchor="middle")
    svg.legend(24, 84, [(name, color, "line") for name, color, *_ in series])
    describe = "; ".join(
        f"{label} ({detail}): dev {pct(dev['overall']['record_core'])}, "
        f"dev-novel {pct(novel['overall']['record_core']) if novel else 'not run'}, "
        f"wrong records filed unreviewed {pct(dev['overall']['false_approval'])}"
        for label, detail, dev, novel in runs if dev)
    return svg, describe


def tools_chart(theme):
    rows = [("27B teacher", TEACHER)] + [(f"{label} ({detail})", run) for label, detail, run in PATH] + \
           [("4-bit (Q4_K_M GGUF)", FOUR_BIT)]
    rows = [(label, s["process"]) for label, run in rows if (s := summary(run))]
    row_h, top = 34, 104
    svg = Svg(760, top + row_h * len(rows) + 50, theme, "Tool calls a case: fewer, and almost none wasted",
              "Mean over the 280 dev cases; 12 allowed; a wasted call was refused, repeated, unknown or invalid")
    left, right = 230, 660
    scale = max(p["tool_calls"] for _, p in rows)
    step = 1
    x_of = lambda v: left + v / (scale * 1.05) * (right - left)
    bottom = top + row_h * len(rows)
    for tick in range(0, int(scale) + 1, step):
        svg.line(x_of(tick), top - 6, x_of(tick), bottom, color="axis" if tick == 0 else "grid")
        svg.text(x_of(tick), bottom + 18, str(tick), color="ink2", anchor="middle")
    for i, (label, p) in enumerate(rows):
        y = top + i * row_h + (row_h - 16) / 2
        useful, wasted = p["tool_calls"] - p["wasted_calls"], p["wasted_calls"]
        svg.text(left - 12, y + 12, label, anchor="end")
        svg.bar(x_of(0), y, x_of(useful) - x_of(0), 16, "s1", round_end=wasted < 0.05)
        if wasted >= 0.05:
            svg.bar(x_of(useful) + 2, y, x_of(useful + wasted) - x_of(useful) - 2, 16, "s2")
        svg.text(x_of(p["tool_calls"]) + 8, y + 12, f"{p['tool_calls']:.1f} ({wasted:.1f} wasted)", color="ink2")
    svg.legend(24, 84, [("Useful calls", "s1", "box"), ("Wasted calls", "s2", "box")])
    describe = "; ".join(f"{label}: {p['tool_calls']:.1f} calls, {p['wasted_calls']:.1f} wasted" for label, p in rows)
    return svg, describe


def dot_chart(theme, title, subtitle, rows):
    """rows: (label, {compared label: value}); one track a row, a dot a model, the shipped value labelled."""
    row_h, top = 32, 108
    svg = Svg(760, top + row_h * len(rows) + 44, theme, title, subtitle)
    left, right = 220, 700
    x_of = lambda v: left + v * (right - left)
    bottom = top + row_h * len(rows)
    for tick in (0, 0.25, 0.5, 0.75, 1):
        svg.line(x_of(tick), top - 8, x_of(tick), bottom - 6, color="grid")
        svg.text(x_of(tick), bottom + 14, pct(tick), color="ink2", anchor="middle")
    for i, (label, values) in enumerate(rows):
        y = top + i * row_h + row_h / 2 - 6
        svg.text(left - 14, y + 4, label, anchor="end", weight=600 if i == len(rows) - 1 and label.startswith("All") else 400)
        known = [values[name] for name, *_ in COMPARED if name in values]
        if len(known) > 1:
            svg.line(x_of(min(known)), y, x_of(max(known)), y, color="axis", width=2)
        for name, _, color in COMPARED:
            if name in values:
                svg.dot(x_of(values[name]), y, color)
        shipped = values.get(COMPARED[-1][0])
        if shipped is not None:
            svg.text(x_of(shipped), y - 10, pct(shipped), size=11, color="ink", anchor="middle")
    svg.legend(24, 86, [(name, color, "dot") for name, _, color in COMPARED])
    describe = "; ".join(f"{label}: " + ", ".join(f"{name} {pct(v)}" for name, v in values.items())
                         for label, values in rows)
    return svg, describe


def fields_chart(theme):
    summaries = {name: s for name, run, _ in COMPARED if (s := summary(run))}
    rows = [(label, {name: s["overall"][key] for name, s in summaries.items()}) for label, key in FIELDS]
    return dot_chart(theme, "Field by field: the trained 0.8B closes most of the gap",
                     "Share of the dev cases with each field of the record right; the shipped file's value is labelled",
                     rows)


def languages_chart(theme):
    summaries = {name: s for name, run, _ in COMPARED if (s := summary(run))}
    codes = sorted(summaries[COMPARED[-1][0]]["by_language"],
                   key=lambda c: -summaries[COMPARED[-1][0]]["by_language"][c]["n"])
    rows = []
    for code in codes:
        n = summaries[COMPARED[-1][0]]["by_language"][code]["n"]
        rows.append((f"{LANGUAGES.get(code, code)} ({n} cases)",
                     {name: s["by_language"][code]["record_core"] for name, s in summaries.items()
                      if code in s["by_language"]}))
    return dot_chart(theme, "Records right in every language",
                     "Records right on dev by the language of the content; the shipped file's value is labelled",
                     rows)


CHARTS = {"path": path_chart, "tools": tools_chart, "fields": fields_chart, "languages": languages_chart}


def main():
    missing = [run for run in (TEACHER, RELEASED, SHIPPED) if summary(run) is None]
    if missing:
        raise SystemExit(f"no grade.json for {', '.join(missing)} under {RUNS}: set SPENDING_EVAL_RECORDS")
    os.makedirs(OUT, exist_ok=True)
    for name, draw in CHARTS.items():
        for theme in THEMES:
            svg, describe = draw(theme)
            path = os.path.join(OUT, f"{name}-{theme}.svg")
            svg.write(path, describe)
            print(os.path.relpath(path, HERE))


if __name__ == "__main__":
    main()
