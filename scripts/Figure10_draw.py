#
# Copyright (c) 2025 Qiqi Gu (qiqi.gu@sjtu.edu.cn), Chenpeng Wu (cpwu_sjtu@sjtu.edu.cn). 
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import csv
import os
import sys
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file = "./outputs/Figure10.csv"

def replace_comma_in_csv(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()
    content = content.replace(", ", ",")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(content)


replace_comma_in_csv(file)
result = pd.read_csv(file)

half2double_precision_ratio = 4

new_rows = []

def unify_gstencil_sptc(row):
    if row["method"] == "SPTC" or row["method"] == "SPTC_half":
        if row["method"] == "SPTC_half":
            gstencil_adjusted = row["GStencil/s"] / half2double_precision_ratio

        if row["shape"] == "1d":
            row_1d1r = row.copy()
            row_1d1r["shape"] = "1d1r"
            row_1d1r["GStencil/s"] = gstencil_adjusted * 7

            row_1d2r = row.copy()
            row_1d2r["shape"] = "1d2r"
            row_1d2r["GStencil/s"] = gstencil_adjusted * 7 / 2

            new_rows.append(row_1d1r)
            new_rows.append(row_1d2r)
            return None

        elif row["shape"] == "2d":
            row_2d1r = row.copy()
            row_2d1r["shape"] = "2d1r"
            row_2d1r["GStencil/s"] = gstencil_adjusted * 7

            row_2d2r = row.copy()
            row_2d2r["shape"] = "2d2r"
            row_2d2r["GStencil/s"] = gstencil_adjusted * 7 / 2

            row_2d3r = row.copy()
            row_2d3r["shape"] = "2d3r"
            row_2d3r["GStencil/s"] = gstencil_adjusted * 7 / 3

            new_rows.append(row_2d1r)
            new_rows.append(row_2d2r)
            new_rows.append(row_2d3r)
            return None

    elif row["method"] == "TCStencil":
        if row["shape"] == "box2d1r" or row["shape"] == "star2d1r":
            row["GStencil/s"] = row["GStencil/s"] / half2double_precision_ratio
        elif row["shape"] == "box2d2r" or row["shape"] == "star2d2r":
            row_2d3r = row.copy()
            row_2d3r["shape"] = "box2d3r" if row["shape"] == "box2d2r" else "star2d3r"
            row_2d3r["GStencil/s"] = (
                row["GStencil/s"] / half2double_precision_ratio / 1.5
            )

            row["GStencil/s"] = row["GStencil/s"] / half2double_precision_ratio

            new_rows.append(row_2d3r)

    elif row["method"] == "ConvStencil" or row["method"] == "LoRAStencil":
        if row["shape"] == "box_2d3r":
            row_2d2r = row.copy()
            row_2d2r["shape"] = "box_2d2r"
            row_2d2r["GStencil/s"] = row["GStencil/s"] * 1.5
            new_rows.append(row_2d2r)

    return row

filtered_result = result.apply(unify_gstencil_sptc, axis=1).dropna()

if new_rows:
    unified_result = pd.concat(
        [filtered_result, pd.DataFrame(new_rows)], ignore_index=True
    )
else:
    unified_result = filtered_result

shapes = {
    "1D1R": {
        "Cudnn": "1d1r",
        "ConvStencil": "1d1r",
        "LoRAStencil": "1d1r",
        "FlashFFTStencil": "1d1r",
        "SPTC_half": "1d1r",
    },
    "1D2R": {
        "Cudnn": "1d2r",
        "ConvStencil": "1d2r",
        "LoRAStencil": "1d2r",
        "FlashFFTStencil": "1d2r",
        "SPTC_half": "1d2r",
    },
    "Box-2D1R": {
        "Cudnn": "2d1r",
        "ConvStencil": "box_2d1r",
        "LoRAStencil": "box_2d1r",
        "FlashFFTStencil": "box_2d1r",
        "TCStencil": "box2d1r",
        "DRStencil": "box2d1r",
        "SPTC_half": "2d1r",
    },
    "Star-2D1R": {
        "Cudnn": "2d1r",
        "ConvStencil": "box_2d1r",
        "LoRAStencil": "box_2d1r",
        "FlashFFTStencil": "box_2d1r",
        "TCStencil": "star2d1r",
        "DRStencil": "star2d1r",
        "SPTC_half": "2d1r",
    },
    "Box-2D2R": {
        "Cudnn": "2d2r",
        "TCStencil": "box2d2r",
        "DRStencil": "box2d2r",
        "SPTC_half": "2d2r",
    },
    "Star-2D2R": {
        "Cudnn": "2d2r",
        "TCStencil": "star2d2r",
        "DRStencil": "star2d2r",
        "SPTC_half": "2d2r",
    },
    "Box-2D3R": {
        "Cudnn": "2d3r",
        "ConvStencil": "box_2d3r",
        "LoRAStencil": "box_2d3r",
        "DRStencil": "box2d3r",
        "SPTC_half": "2d3r",
    },
    "Star-2D3R": {
        "Cudnn": "2d3r",
        "ConvStencil": "box_2d3r",
        "LoRAStencil": "box_2d3r",
        "DRStencil": "star2d3r",
        "SPTC_half": "2d3r",
    },
}

def extract_shape_dataframes(unified_result, shapes_dict):
    shape_dfs = {}

    for shape_name, method_shapes in shapes_dict.items():
        shape_filter = pd.Series(False, index=unified_result.index)

        for method, shape in method_shapes.items():
            method_shape_filter = (unified_result["method"] == method) & (
                unified_result["shape"] == shape
            )
            shape_filter = shape_filter | method_shape_filter

        shape_dfs[shape_name] = unified_result[shape_filter].copy()

        shape_dfs[shape_name]["shape_label"] = shape_name

    return shape_dfs


shape_dataframes = extract_shape_dataframes(unified_result, shapes)

def plot_stencil_performance(shape_dataframes):
    shape_groups = {
        "1D1R": ["1D1R"],
        "1D2R": ["1D2R"],
        "2D1R": ["Box-2D1R", "Star-2D1R"],
        "2D2R": ["Box-2D2R", "Star-2D2R"],
        "2D3R": ["Box-2D3R", "Star-2D3R"],
    }

    method_order = [
        "Cudnn",
        "DRStencil",
        "TCStencil",
        "ConvStencil",
        "LoRAStencil",
        "FlashFFTStencil",
        "SPTC_half",
    ]
    
    method_display_names = {
        "Cudnn": "cuDNN",
        "DRStencil": "DRStencil",
        "TCStencil": "TCStencil",
        "ConvStencil": "ConvStencil",
        "LoRAStencil": "LoRAStencil",
        "FlashFFTStencil": "FlashFFTStencil",
        "SPTC": "SPTC",
        "SPTC_half": "SPIDER",
    }
    
    
    colors = {
        "Cudnn": "#A46454",
        "DRStencil": "#C8A38A",
        "TCStencil": "#F1E3BC",
        "ConvStencil": "#E2D588",
        "LoRAStencil": "#A8D19E",
        "FlashFFTStencil": "#B1CCC5",
        "SPTC_half": "#7095A6"
    }

    
    color_for_speedup = "#A084CA"
    color_for_speedup = "gray"

    legend_fontsize = 20
    label_fontsize = 20
    ticks_fontsize = 20
    speedup_marker_size = 10

    weight_ratios = [0] * len(shape_groups)
    for idx, (group_name, shapes) in enumerate(shape_groups.items()):
        for shape_idx, shape_name in enumerate(shapes):
            df = shape_dataframes[shape_name]
            for i, method in enumerate(method_order):
                method_data = df[df["method"] == method]

                if len(method_data) == 0:
                    continue
                if len(method_data) > 0 and not pd.isna(
                    method_data["GStencil/s"].values[0]
                ):
                    weight_ratios[idx] += 1

    fig, axs = plt.subplots(
        1, 5, figsize=(26, 6), facecolor="white", width_ratios=weight_ratios
    )
    axs = axs.flatten()

    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1, facecolor=colors[method], edgecolor="black", linewidth=1.0
        )
        for method in method_order
    ]
    legend_labels = [method_display_names[method] for method in method_order]

    speedup_line_handle = plt.Line2D(
        [0],
        [0],
        color=color_for_speedup,
        marker="^",
        linestyle="None",
        markersize=speedup_marker_size,
    )

    bar_width = 0.12

    for idx, (group_name, shapes) in enumerate(shape_groups.items()):
        ax = axs[idx]
        ax2 = ax.twinx()

        names = []
        perf_values = {}

        group_space = 0.06 * weight_ratios[idx] + 0.12

        for shape_idx, shape_name in enumerate(shapes):
            if shape_name not in shape_dataframes:
                continue

            df = shape_dataframes[shape_name]
            names.append(shape_name)

            perf_values[shape_name] = {}

            for method in method_order:
                method_data = df[df["method"] == method]
                if len(method_data) > 0 and not pd.isna(
                    method_data["GStencil/s"].values[0]
                ):
                    perf_values[shape_name][method] = method_data["GStencil/s"].values[
                        0
                    ]

        for shape_idx, shape_name in enumerate(shapes):
            if shape_name not in shape_dataframes:
                continue

            df = shape_dataframes[shape_name]

            if "Star" in shape_name:
                base_x = group_space
            else:
                base_x = 0

            available_methods = []
            for method in method_order:
                method_data = df[df["method"] == method]
                if len(method_data) > 0 and not pd.isna(
                    method_data["GStencil/s"].values[0]
                ):
                    available_methods.append(method)

            if not available_methods:
                continue

            total_width = len(available_methods) * bar_width
            start_x = base_x - total_width / 2 + bar_width / 2

            for i, method in enumerate(available_methods):
                method_data = df[df["method"] == method]
                x_pos = start_x + i * bar_width

                ax.bar(
                    x_pos,
                    method_data["GStencil/s"].values[0],
                    width=bar_width,
                    color=colors[method],
                    edgecolor="black",
                    label="_nolegend_",
                    alpha=0.8,
                )
                if method != "Cudnn":
                    ax2.plot(
                        x_pos,
                        method_data["GStencil/s"].values[0]
                        / perf_values[shape_name]["Cudnn"],
                        "^",
                        markersize=speedup_marker_size,
                        color=color_for_speedup,
                        linestyle="None",
                    )

        speedups = []
        x_speedup_positions = []
        for shape_name in shapes:
            speedups.append(
                perf_values[shape_name]["SPTC_half"] / perf_values[shape_name]["Cudnn"]
            )
            if "Star" in shape_name:
                x_speedup_positions.append(group_space)
            else:
                x_speedup_positions.append(0)

        ax.tick_params(axis="x", labelsize=label_fontsize)
        ax.tick_params(axis="y", labelsize=ticks_fontsize)

        if "1D" in group_name:
            ax.set_xticks([0])
            ax.set_xticklabels(names)
            ax.set_xlim(-0.36, 0.36)
        else:
            if len(shapes) > 1:
                ax.set_xticks([0, group_space])
                ax.set_xticklabels(names)
            else:
                ax.set_xticks([])

        if idx == 0:
            ax.set_ylim(0, math.ceil(max(df["GStencil/s"]) / 100) * 100)
        elif idx == 3:
            ax.set_ylim(0, math.ceil(max(df["GStencil/s"]) / 20) * 20)
        elif idx == 5:
            ax.set_ylim(0, math.ceil(max(df["GStencil/s"]) / 8) * 8)
        else:
            ax.set_ylim(0, math.ceil(max(df["GStencil/s"]) / 50) * 50)

        if idx == 0:
            ax.set_ylabel("GStencils/s", fontsize=label_fontsize)

        if idx == 5:
            ax2.set_ylabel("Speedup", color="black", fontsize=label_fontsize)
        ymax = math.ceil(max(speedups))
        ax2.set_ylim(1, ymax)
        ax2.tick_params(axis="y", labelcolor="black", labelsize=ticks_fontsize)

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1)
        ax.set_facecolor("white")
        ax.grid(False, axis="x")
        ax.grid(True, axis="y", linestyle="--", alpha=0.9)

    all_handles = legend_handles + [speedup_line_handle]
    all_labels = legend_labels + ["Speedup"]

    
    fig.legend(
        all_handles,
        all_labels,
        loc="lower center",
        ncol=len(all_handles),
        frameon=False,
        fontsize=legend_fontsize,
        edgecolor="gray",
        facecolor="white",
        bbox_to_anchor=(0.5, 0.96),
    )

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make space for the legend
    plt.savefig(
        "./outputs/Figure10.pdf", dpi=300, bbox_inches="tight"
    )
    plt.show()

plot_stencil_performance(shape_dataframes)