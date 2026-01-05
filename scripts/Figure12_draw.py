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

file = "./outputs/Figure12.csv"

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
    if row["method"] == "SPTC_half" or row["method"] == "SPTC_half_dense" or row["method"] == "SPTC_half_wo_kernel_optimization":
        gstencil_adjusted = row["GStencil/s"] / half2double_precision_ratio

        row_2d2r = row.copy()
        row_2d2r["shape"] = "2d2r"
        row_2d2r["GStencil/s"] = gstencil_adjusted * 7 / 2

        new_rows.append(row_2d2r)
        return pd.Series([None] * len(row), index=row.index)  # Original row will be dropped

    elif row["method"] == "TCStencil":
        row["GStencil/s"] = row["GStencil/s"] / half2double_precision_ratio
        
    return row

filtered_result = result.apply(unify_gstencil_sptc, axis=1).dropna()

if new_rows:
    unified_result = pd.concat(
        [filtered_result, pd.DataFrame(new_rows)], ignore_index=True
    )
else:
    unified_result = filtered_result

shapes = {
    "2d2r_Box": {
        "Cudnn": "2d2r",
        "SPTC_half_dense": "2d2r",
        "SPTC_half_wo_kernel_optimization": "2d2r",
        "SPTC_half": "2d2r",
        "TCStencil": "box2d2r"
    }
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

for shape_name, df in shape_dataframes.items():
    df = df[df['dim_1'] <= 10240]
    df = df[df['dim_1'] >= 1280]
        
    df['dim_1'] = df['dim_1'].astype(int)
    
    cudnn_gstencil = df[df['method'] == 'TCStencil'].set_index('dim_1')['GStencil/s']
    df['Speedup'] = df['GStencil/s'] / df['dim_1'].map(cudnn_gstencil)
    shape_dataframes[shape_name] = df  # 赋值回去

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import matplotlib.font_manager as fm

font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf"  # Linux常见路径
prop = fm.FontProperties(fname=font_path)

legend_fontsize = 16
label_fontsize = 22
ticks_fontsize = 22
title_fontsize = 22

line_width = 1.5
marker_size = 10


def plot_selected_shapes(shape_dataframes, shape_names):
    n_shapes = len(shape_names)
    fig, axes = plt.subplots(1, n_shapes, figsize=(12, 5))

    if n_shapes == 1:
        axes = [axes]

    all_methods = set()
    for shape_name in shape_names:
        df = shape_dataframes[shape_name]
        all_methods.update(df["method"].unique())

    all_methods = sorted(list(all_methods))

    last_y = []
    for i, shape_name in enumerate(shape_names):
        df = shape_dataframes[shape_name]
        ax = axes[i]
        method_order = [
            "TCStencil",
            "SPTC_half_dense",
            "SPTC_half_wo_kernel_optimization",
            "SPTC_half",
        ]
        
        method_colors = {
            "TCStencil": "#F1E3BC",
            "SPTC_half_dense": "#C2D3DE",
            "SPTC_half_wo_kernel_optimization": "#99B4C2",
            "SPTC_half": "#7095A6",
        }
        methods = [m for m in method_order if m in df["method"].unique()]
        x = np.arange(len(df["dim_1"].unique()))
        width = 0.18
        dim_1s = sorted(df["dim_1"].unique())
        for j, method in enumerate(methods):
            method_df = df[df["method"] == method].sort_values("dim_1")

            y = [
                (
                    method_df[method_df["dim_1"] == d]["Speedup"].values[0]
                    if d in method_df["dim_1"].values
                    else 0
                )
                for d in dim_1s
            ]

            ax.bar(
                x + j * width,
                y,
                width=width,
                label=method,
                color=method_colors[method],
                edgecolor="black",
                linewidth=line_width,
            )

            for k, v in enumerate(y):
                if j > 0:
                    last_v = last_y[k]
                    speedup = ((v / last_v) - 1) * 100

                    x_start = x[k] + (j - 1) * width
                    x_end = x[k] + (j - 0.5) * width
                    y_start = last_v
                    y_end = v
                    if y_end - y_start > 0.35:
                        y_end = y_start + 0.35

                    if j == 3 and (k == 0 or k == 1):
                        ax.annotate(
                            "",
                            xy=(x_end - 0.01, y_end + 0.04),
                            xytext=(x_start - 0.02, y_start + 0.04),
                            arrowprops=dict(
                                color="#D12C25",
                                arrowstyle="->",
                                connectionstyle="arc3,rad=0.25",
                                linewidth=1.5,
                            ),
                        )
                    else:
                        ax.annotate(
                            "",
                            xy=(x_end - 0.02, y_end + 0.04),
                            xytext=(x_start - 0.02, y_start + 0.04),
                            arrowprops=dict(
                                color="#D12C25",
                                arrowstyle="->",
                                connectionstyle="angle3,angleA=0,angleB=90",
                                linewidth=1.5,
                            ),
                        )

                    x_mid = (x_start + x_end) / 2 - 0.12
                    y_mid = (y_start + y_end) / 2 + 0.03

                    if j == 3 and (k == 0 or k == 1):
                        ax.text(
                            x_mid + 0.03,
                            y_mid + 0.06,
                            f"{speedup:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=16,
                            color="#D12C25",
                            weight="bold"
                        )
                    elif j == 3 and (k == 2 or k == 3):
                        ax.text(
                            x_mid + 0.01,
                            y_mid + 0.02,
                            f"{speedup:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=16,
                            color="#D12C25",
                            weight="bold"
                        )
                    elif j < 3:
                        ax.text(
                            x_mid,
                            y_mid,
                            f"{speedup:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=16,
                            color="#D12C25",
                            weight="bold"
                        )

            last_y = y
        ax.set_xticks(x + width * (len(methods) - 1) / 2)
        ax.set_xticklabels(dim_1s, fontsize=ticks_fontsize)
        xlabel = {"2d2r_Box": "Box-2D2R"}
        ax.set_xlabel("Problem Size($X$, $X$)", fontsize=label_fontsize)
        if i == 0:
            ax.set_ylabel("Speedup", fontsize=label_fontsize)
        ax.tick_params(axis="y", labelsize=ticks_fontsize)

        ax.grid(True, axis="y", linestyle="--", alpha=0.7)
        for spine in ax.spines.values():
            spine.set_linewidth(line_width)

    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)

    method_order = [
        "TCStencil",
        "SPTC_half_dense",
        "SPTC_half_wo_kernel_optimization",
        "SPTC_half",
    ]

    sorted_handles_labels = []
    for method in method_order:
        for i, label in enumerate(labels):
            if label == method:
                sorted_handles_labels.append((handles[i], label))
                break

    sorted_handles, sorted_labels = (
        zip(*sorted_handles_labels) if sorted_handles_labels else ([], [])
    )

    sorted_labels = [
        label.replace("TCStencil", "TCStencil")
        .replace("SPTC_half_dense", "SPIDER w. TC")
        .replace("SPTC_half_wo_kernel_optimization", "SPIDER w. SpTC")
        .replace("SPTC_half", "SPIDER w. SpTC+CO")
        for label in sorted_labels
    ]

    fig.legend(
        sorted_handles,
        sorted_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=len(sorted_labels),
        fontsize=legend_fontsize,
        frameon=False,
        columnspacing=1
    )

    plt.tight_layout()
    plt.savefig("./outputs/Figure12.pdf", dpi=600, bbox_inches="tight")
    plt.show()


plot_selected_shapes(shape_dataframes, ["2d2r_Box"])