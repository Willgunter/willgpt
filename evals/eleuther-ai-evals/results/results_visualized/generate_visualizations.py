#!/usr/bin/env python3
"""Generate heatmap and grouped bar-chart comparisons for evaluated LLMs.
   Convert to SVG here: https://svgtopng.com/"""

import json
from pathlib import Path

OUTPUT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = OUTPUT_DIR.parent

LLM_ORDER = [
    "ServiceNow-AI/Apriel-5B-Base",
    "deepseek-ai/deepseek-llm-7b-base",
    "google/gemma-2-2b",
    "ibm-granite/granite-4.0-micro-base",
    "meta-llama/Llama-3.2-3B",
    "Qwen/Qwen2.5-3B",
    "Qwen/Qwen3-4B-Base",
    "HuggingFaceTB/SmolLM3-3B-Base",
]

DISPLAY_NAMES = {
    "ServiceNow-AI/Apriel-5B-Base": "Apriel 5B",
    "deepseek-ai/deepseek-llm-7b-base": "Deepseek 7B",
    "google/gemma-2-2b": "Gemma 2.2B",
    "ibm-granite/granite-4.0-micro-base": "Granite 4.0",
    "meta-llama/Llama-3.2-3B": "Llama 3.2 3B",
    "Qwen/Qwen2.5-3B": "Qwen 2.5 3B",
    "Qwen/Qwen3-4B-Base": "Qwen 3 4B",
    "HuggingFaceTB/SmolLM3-3B-Base": "SmolLM3 3B",
}

DATASETS = [
    ("arc_challenge", "ARC Challenge (norm acc)", "acc_norm,none"),
    ("mathqa", "MathQA (norm acc)", "acc_norm,none"),
    ("gsm8k", "GSM8K (strict exact match)", "exact_match,strict-match"),
    ("college_computer_science", "MMLU College CS", "acc,none"),
    ("college_mathematics", "MMLU College Math", "acc,none"),
    ("college_physics", "MMLU College Physics", "acc,none"),
    ("conceptual_physics", "MMLU Conceptual Physics", "acc,none"),
    ("electrical_engineering", "MMLU Electrical Engineering", "acc,none"),
]

FILTER_TO_SELECTED_MMLU = False
SELECTED_MMLU_ALIASES = {"conceptual_physics", "electrical_engineering"}

ACTIVE_DATASETS = [
    dataset for dataset in DATASETS if (not FILTER_TO_SELECTED_MMLU or dataset[0] in SELECTED_MMLU_ALIASES)
]

ALIAS_TO_METRIC = {alias: metric for alias, _label, metric in ACTIVE_DATASETS}

LLM_NAME_PATTERNS = [
    ("apriel", "ServiceNow-AI/Apriel-5B-Base"),
    ("deepseek", "deepseek-ai/deepseek-llm-7b-base"),
    ("gemma", "google/gemma-2-2b"),
    ("granite", "ibm-granite/granite-4.0-micro-base"),
    ("llama", "meta-llama/Llama-3.2-3B"),
    ("qwen2.5", "Qwen/Qwen2.5-3B"),
    ("qwen3-4b", "Qwen/Qwen3-4B-Base"),
    ("smollm3", "HuggingFaceTB/SmolLM3-3B-Base"),
    ("smallm3", "HuggingFaceTB/SmolLM3-3B-Base"),
    ("hf-lm3", "HuggingFaceTB/SmolLM3-3B-Base"),
]

LLM_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#17becf",
]


def guess_llm_from_filename(path: Path) -> str:
    lower = path.name.lower()
    for substring, llm in LLM_NAME_PATTERNS:
        if substring in lower:
            return llm
    raise RuntimeError(f"unrecognized model identifier in {path}")


def parse_file(path: Path):
    payload = json.loads(path.read_text())
    entries = []
    if "results" in payload and isinstance(payload["results"], dict):
        model_name = payload.get("model_name")
        if not model_name:
            config = payload.get("config")
            if isinstance(config, dict):
                model_name = config.get("model_name")
        if not model_name:
            model_name = guess_llm_from_filename(path)
        for info in payload["results"].values():
            alias = info.get("alias")
            if alias not in ALIAS_TO_METRIC:
                continue
            metric_key = ALIAS_TO_METRIC[alias]
            if metric_key not in info:
                continue
            entries.append((model_name, alias, float(info[metric_key])))
        return entries

    if all(isinstance(v, dict) for v in payload.values()):
        for llm_key, llm_block in payload.items():
            if not isinstance(llm_block, dict):
                continue
            for values in llm_block.values():
                if not isinstance(values, dict):
                    continue
                alias = values.get("alias")
                if alias not in ALIAS_TO_METRIC:
                    continue
                metric_key = ALIAS_TO_METRIC[alias]
                if metric_key not in values:
                    continue
                entries.append((llm_key, alias, float(values[metric_key])))
        if entries:
            return entries

    if "alias" in payload:
        alias = payload["alias"]
        if alias in ALIAS_TO_METRIC:
            metric_key = ALIAS_TO_METRIC[alias]
            if metric_key in payload:
                llm = guess_llm_from_filename(path)
                entries.append((llm, alias, float(payload[metric_key])))
    return entries


def load_scores():
    results = {llm: {} for llm in LLM_ORDER}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        for llm_name, alias, value in parse_file(path):
            if llm_name not in results:
                continue
            results[llm_name][alias] = value
    for llm in LLM_ORDER:
        missing = [alias for alias, _label, _ in ACTIVE_DATASETS if alias not in results[llm]]
        if missing:
            raise RuntimeError(f"{llm} missing data for {missing}")
    return results


def build_matrix(results):
    return [[results[llm][alias] for alias, _label, _metric in ACTIVE_DATASETS] for llm in LLM_ORDER]


def color_from_value(value: float) -> str:
    r = int(240 * (1 - value) + 40 * value)
    g = int(80 * (1 - value) + 200 * value)
    b = int(80 * (1 - value) + 120 * value)
    return f"rgb({r},{g},{b})"


def write_heatmap(matrix):
    cell_width = 140
    cell_height = 48
    margin_left = 190
    margin_top = 160
    margin_right = 40
    margin_bottom = 80
    width = margin_left + cell_width * len(ACTIVE_DATASETS) + margin_right
    height = margin_top + cell_height * len(LLM_ORDER) + margin_bottom

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" font-family="Verdana, sans-serif">',
        '<defs><linearGradient id="heatGradient" x1="0%" x2="100%" y1="0%" y2="0%"><stop offset="0%" stop-color="#e84a5f"/><stop offset="100%" stop-color="#2ecc71"/></linearGradient></defs>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="20" y="30" font-size="20" fill="#222">LLM benchmark heatmap (metric-specific scores)</text>',
    ]

    for j, (_alias, label, _metric) in enumerate(ACTIVE_DATASETS):
        x = margin_left + j * cell_width + cell_width / 2
        y = margin_top - 48
        lines.append(
            f'<text x="{x}" y="{y}" text-anchor="middle" font-size="12" fill="#333" transform="rotate(-30 {x} {y})">{label}</text>'
        )

    for i, llm in enumerate(LLM_ORDER):
        y = margin_top + i * cell_height
        lines.append(f'<text x="{margin_left - 10}" y="{y + cell_height/2 + 5}" text-anchor="end" font-size="12" fill="#111">{DISPLAY_NAMES[llm]}</text>')
        for j, (_alias, _label, _metric) in enumerate(ACTIVE_DATASETS):
            x = margin_left + j * cell_width
            value = matrix[i][j]
            lines.append(
                f'<rect x="{x}" y="{y}" width="{cell_width}" height="{cell_height}" fill="{color_from_value(value)}" stroke="#ccc" stroke-width="1"/>'
            )
            text_color = "#111" if value >= 0.45 else "#fff"
            lines.append(
                f'<text x="{x + cell_width/2}" y="{y + cell_height/2 + 5}" text-anchor="middle" font-size="11" fill="{text_color}">{value*100:.1f}%</text>'
            )

    legend_x = margin_left
    legend_y = margin_top + len(LLM_ORDER) * cell_height + 25
    lines.append(f'<text x="{legend_x}" y="{legend_y - 10}" font-size="12" fill="#333">Color scale: low (red) → high (green)</text>')
    gradient_width = cell_width * 2
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="{gradient_width}" height="16" fill="url(#heatGradient)" stroke="#555"/>')
    lines.append('</svg>')

    (OUTPUT_DIR / "heatmap.svg").write_text("\n".join(lines))


def write_bar_chart(matrix):
    bar_width = 14
    bar_spacing = 6
    group_spacing = 40
    margin_left = 180
    margin_right = 40
    margin_top = 80
    margin_bottom = 140
    group_width = len(LLM_ORDER) * bar_width + (len(LLM_ORDER) - 1) * bar_spacing
    total_width = margin_left + len(ACTIVE_DATASETS) * (group_width + group_spacing) + margin_right
    chart_height = 360
    total_height = margin_top + chart_height + margin_bottom
    max_value = 1.0
    show_value_labels = bar_width >= 24
    show_horizontal_gridlines = True

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{total_width}" height="{total_height}" viewBox="0 0 {total_width} {total_height}" font-family="Verdana, sans-serif">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="20" y="30" font-size="20" fill="#222">Grouped benchmark bar chart (all 8×8 scores)</text>',
    ]

    y_axis_x = margin_left - 20
    lines.append(f'<line x1="{y_axis_x}" y1="{margin_top}" x2="{y_axis_x}" y2="{margin_top + chart_height}" stroke="#111" stroke-width="1"/>')
    for tick in range(0, 101, 10):
        value = tick / 100
        y = margin_top + chart_height - (value / max_value) * chart_height
        if show_horizontal_gridlines:
            lines.append(
                f'<line x1="{y_axis_x}" y1="{y}" x2="{total_width - margin_right}" y2="{y}" stroke="#e0e0e0" stroke-width="1"/>'
            )
        lines.append(f'<line x1="{y_axis_x - 6}" y1="{y}" x2="{y_axis_x}" y2="{y}" stroke="#111" stroke-width="1"/>')
        lines.append(f'<text x="{y_axis_x - 10}" y="{y + 4}" text-anchor="end" font-size="10" fill="#222">{tick}%</text>')

    for idx, (_alias, label, _metric) in enumerate(ACTIVE_DATASETS):
        group_x = margin_left + idx * (group_width + group_spacing)
        lines.append(
            f'<text x="{group_x + group_width / 2}" y="{margin_top + chart_height + 30}" text-anchor="middle" font-size="11" fill="#333">{label}</text>'
        )
        for llm_index, llm in enumerate(LLM_ORDER):
            value = matrix[llm_index][idx]
            x = group_x + llm_index * (bar_width + bar_spacing)
            bar_height = (value / max_value) * chart_height
            y = margin_top + chart_height - bar_height
            color = LLM_COLORS[llm_index % len(LLM_COLORS)]
            lines.append(
                f'<rect x="{x}" y="{y}" width="{bar_width}" height="{bar_height}" fill="{color}" stroke="#222" stroke-width="0.5"/>'
            )
            if show_value_labels:
                if bar_height > 14:
                    text_color = "#fff"
                    text_y = y + 14
                else:
                    text_color = "#111"
                    text_y = y - 2
                lines.append(
                    f'<text x="{x + bar_width / 2}" y="{text_y}" text-anchor="middle" font-size="10" fill="{text_color}">{value*100:.1f}%</text>'
                )

    legend_y = margin_top + chart_height + 60
    lines.append(f'<text x="{margin_left}" y="{legend_y}" font-size="12" fill="#333">Model legend:</text>')
    legend_x = margin_left + 120
    for idx, llm in enumerate(LLM_ORDER):
        x = legend_x + idx * 140
        color = LLM_COLORS[idx % len(LLM_COLORS)]
        lines.append(f'<rect x="{x}" y="{legend_y - 12}" width="10" height="10" fill="{color}"/>')
        lines.append(
            f'<text x="{x + 14}" y="{legend_y - 2}" font-size="10" fill="#111">{DISPLAY_NAMES[llm]}</text>'
        )

    lines.append("</svg>")
    (OUTPUT_DIR / "barchart.svg").write_text("\n".join(lines))


def main():
    results = load_scores()
    matrix = build_matrix(results)
    write_heatmap(matrix)
    write_bar_chart(matrix)
    print("Generated heatmap.svg and barchart.svg in", OUTPUT_DIR)


if __name__ == "__main__":
    main()
