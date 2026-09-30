from __future__ import annotations

import csv
import html
from collections import defaultdict

import numpy as np

from common.project_paths import REPOSITORY_ROOT, RESULTS_ROOT


INPUT_CSV = (
    RESULTS_ROOT
    / "temporal"
    / "track_metrics"
    / "stable_detection_onset_per_track.csv"
)
OUTPUT_SVG = (
    REPOSITORY_ROOT
    / "figures"
    / "paper"
    / "stable-detection-onset-curves.svg"
)

MODELS = ("yolov8n", "yolo11n", "yolo12n")
MODEL_LABELS = {
    "yolov8n": "YOLOv8n",
    "yolo11n": "YOLO11n",
    "yolo12n": "YOLO12n",
}
CONDITIONS = (
    ("fog_l3", "Fog L3"),
    ("lowlight_l3", "Low-Light L3"),
    ("motionblur_l3", "Motion Blur L3"),
)
TIMES = np.arange(0.0, 10.0001, 0.25)

WIDTH = 2000
HEIGHT = 1390
Y_MAX = 65.0

PANEL_WIDTH = 770
PLOT_HEIGHT = 420
LEFT_COLUMN = 170
RIGHT_COLUMN = 1080
TOP_ROW = 140
BOTTOM_ROW = 760

PANELS = (
    ("fog_l3", "(a) Fog", LEFT_COLUMN, TOP_ROW, True),
    ("lowlight_l3", "(b) Low light", RIGHT_COLUMN, TOP_ROW, False),
    ("motionblur_l3", "(c) Motion blur", LEFT_COLUMN, BOTTOM_ROW, True),
)

Y_TICKS = (0, 20, 40, 60)
X_TICKS = (0, 2, 4, 6, 8, 10)

MODEL_COLORS = {
    "yolov8n": "#377EB8",
    "yolo11n": "#E69F00",
    "yolo12n": "#009E73",
}
MODEL_DASHES = {
    "yolov8n": "",
    "yolo11n": "",
    "yolo12n": "",
}


def load_starts() -> dict[str, dict[str, list[tuple[str, float]]]]:
    grouped: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    report_conditions = {item[0] for item in CONDITIONS}
    with INPUT_CSV.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            condition = row["condition"]
            model = row["model"]
            if condition not in report_conditions or model not in MODELS:
                continue
            if int(row["eligible"]) != 1:
                continue
            if int(row["stable"]) == 1 and row["stable_start_seconds"]:
                start = float(row["stable_start_seconds"])
            else:
                start = float("inf")
            grouped[condition][model].append((row["track_key"], start))

    data: dict[str, dict[str, list[tuple[str, float]]]] = defaultdict(dict)
    for condition, _ in CONDITIONS:
        for model in MODELS:
            track_starts = sorted(grouped[condition][model])
            if len(track_starts) != 434:
                raise RuntimeError(
                    f"Expected 434 eligible tracks for {condition}/{model}, "
                    f"got {len(track_starts)}"
                )
            data[condition][model] = track_starts
    return data


def x_map(panel_left: float, time: float) -> float:
    return panel_left + PANEL_WIDTH * time / 10.0


def y_map(plot_top: float, value: float) -> float:
    plot_bottom = plot_top + PLOT_HEIGHT
    return plot_bottom - PLOT_HEIGHT * value / Y_MAX


def cumulative_rates(track_starts: list[tuple[str, float]]) -> np.ndarray:
    starts = np.asarray([start for _, start in track_starts], dtype=np.float64)
    return (starts[:, None] <= TIMES[None, :]).mean(axis=0) * 100.0


def line_path(panel_left: float, plot_top: float, values: np.ndarray) -> str:
    commands = [
        f"M {x_map(panel_left, TIMES[0]):.2f} "
        f"{y_map(plot_top, values[0]):.2f}"
    ]
    for index in range(1, len(TIMES)):
        commands.append(
            f"L {x_map(panel_left, TIMES[index]):.2f} "
            f"{y_map(plot_top, values[index]):.2f}"
        )
    return " ".join(commands)


def build_svg() -> str:
    data = load_starts()
    curves = {
        (condition, model): cumulative_rates(data[condition][model])
        for condition, _ in CONDITIONS
        for model in MODELS
    }

    lines: list[str] = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
            f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">'
        ),
        f'<rect width="{WIDTH}" height="{HEIGHT}" fill="#ffffff"/>',
        "<style>",
        '.latin{font-family:Arial,"Malgun Gothic",sans-serif;fill:#111827}',
        ".tick{font-size:54px;font-weight:400}",
        ".panel-title{font-size:62px;font-weight:700}",
        ".axis-label{font-size:60px;font-weight:500}",
        ".legend{font-size:58px;font-weight:500}",
        "</style>",
    ]

    for condition, title, panel_left, plot_top, show_y_labels in PANELS:
        plot_bottom = plot_top + PLOT_HEIGHT
        panel_right = panel_left + PANEL_WIDTH
        panel_center = panel_left + PANEL_WIDTH / 2.0
        lines.append(
            f'<text x="{panel_center:.1f}" y="{plot_top - 48:.1f}" '
            f'text-anchor="middle" class="latin panel-title">'
            f"{html.escape(title)}</text>"
        )

        # A light horizontal grid is enough for reading values and avoids
        # cluttering the staircase geometry.
        for value in Y_TICKS:
            y = y_map(plot_top, value)
            lines.append(
                f'<line x1="{panel_left}" y1="{y:.2f}" '
                f'x2="{panel_right}" y2="{y:.2f}" '
                'stroke="#D9DDE3" stroke-width="2.2"/>'
            )
            if show_y_labels:
                lines.append(
                    f'<text x="{panel_left - 22}" y="{y + 18:.2f}" '
                    f'text-anchor="end" class="latin tick">{value}</text>'
                )

        for value in X_TICKS:
            x = x_map(panel_left, value)
            lines.append(
                f'<line x1="{x:.2f}" y1="{plot_bottom}" '
                f'x2="{x:.2f}" y2="{plot_bottom + 12}" '
                'stroke="#111111" stroke-width="3.0"/>'
            )
            lines.append(
                f'<text x="{x:.2f}" y="{plot_bottom + 62}" '
                f'text-anchor="middle" class="latin tick">{value}</text>'
            )

        for model in MODELS:
            color = MODEL_COLORS[model]
            dash = MODEL_DASHES[model]
            dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
            lines.append(
                f'<path d="{line_path(panel_left, plot_top, curves[(condition, model)])}" '
                f'fill="none" stroke="{color}" stroke-width="9.5"'
                f'{dash_attribute} stroke-linejoin="round" stroke-linecap="round"/>'
            )

        lines.append(
            f'<line x1="{panel_left}" y1="{plot_top}" '
            f'x2="{panel_left}" y2="{plot_bottom}" '
            'stroke="#111111" stroke-width="4.5"/>'
        )
        lines.append(
            f'<line x1="{panel_left}" y1="{plot_bottom}" '
            f'x2="{panel_right}" y2="{plot_bottom}" '
            'stroke="#111111" stroke-width="4.5"/>'
        )

    # The fourth cell holds the common legend.
    legend_x = RIGHT_COLUMN + 78
    legend_y_values = (865, 980, 1095)
    for y, model in zip(legend_y_values, MODELS):
        color = MODEL_COLORS[model]
        dash = MODEL_DASHES[model]
        dash_attribute = f' stroke-dasharray="{dash}"' if dash else ""
        lines.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 150}" '
            f'y2="{y}" stroke="{color}" stroke-width="9.5"'
            f'{dash_attribute} stroke-linecap="butt"/>'
        )
        lines.append(
            f'<text x="{legend_x + 185}" y="{y + 19}" '
            f'class="latin legend">{MODEL_LABELS[model]}</text>'
        )

    lines.append(
        '<text x="1015" y="1360" text-anchor="middle" '
        'class="latin axis-label">Time Since Track Start (s)</text>'
    )
    lines.append(
        '<text x="58" y="665" text-anchor="middle" '
        'class="latin axis-label" transform="rotate(-90 58 665)">'
        'Cumulative Stable-Detection Onset Rate (%)</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    OUTPUT_SVG.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_SVG.write_text(build_svg(), encoding="utf-8")
    print(OUTPUT_SVG.resolve())


if __name__ == "__main__":
    main()
