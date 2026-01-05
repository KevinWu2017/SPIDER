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

file = "./outputs/Figure11.csv"

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

def drop_some_points(row):
    if row["shape"] == "1d1r" or row["shape"] == "1d2r" or row["shape"] == "1d":
        if row["method"] == "SPTC" or row["method"] == "SPTC_half":
            if row["dim_2"] == 1024 or row["dim_2"] % 2048 == 0:
                return row
            else:
                return None
        elif (row["dim_1"] / 1024) == 1024 or (row["dim_1"] / 1024) % 2048 == 0:
            return row
        else:
            return None
    elif row["method"] == "TCStencil":
        if row["dim_1"] < 640:
            return None
        else:
            return row
    return row
unified_result = unified_result.apply(drop_some_points, axis=1).dropna()

shapes = {
    "1D1R": {
        "Cudnn": "1d1r",
        "ConvStencil": "1d1r",
        "LoRAStencil": "1d1r",
        "SPTC_half": "1d1r",
    },
    "1D2R": {
        "Cudnn": "1d2r",
        "ConvStencil": "1d2r",
        "LoRAStencil": "1d2r",
        "SPTC_half": "1d2r",
    },
    "Box-2D1R": {
        "Cudnn": "2d1r",
        "ConvStencil": "box_2d1r",
        "LoRAStencil": "box_2d1r",
        "TCStencil": "box2d1r",
        "DRStencil": "box2d1r",
        "SPTC_half": "2d1r",
    },
    "Star-2D1R": {
        "Cudnn": "2d1r",
        "ConvStencil": "box_2d1r",
        "LoRAStencil": "box_2d1r",
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

shape_uppercase = {
    "1d1r": "1D1R",
    "1d2r": "1D2R",
    "2d1r_Box": "Box-2D1R",
    "2d1r_Star": "Star-2D1R",
    "2d2r_Box": "Box-2D2R",
    "2d2r_Star": "Star-2D2R",
    "2d3r_Box": "Box-2D3R",
    "2d3r_Star": "Star-2D3R",
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
    if shape_name == '1D1R' or shape_name == '1D2R':
        df.loc[df['method'] == 'SPTC', 'dim_1'] *= df['dim_2']
        df.loc[df['method'] == 'SPTC', 'dim_2'] = 1
        df.loc[df['method'] == 'SPTC_half', 'dim_1'] *= df['dim_2']
        df.loc[df['method'] == 'SPTC_half', 'dim_2'] = 1

import matplotlib.pyplot as plt

legend_fontsize = 19
label_fontsize = 22
ticks_fontsize = 20
title_fontsize = label_fontsize

line_width = 3
marker_size = 9

def plot_selected_shapes(shape_dataframes, shape_names):
    """
    Plot specific shapes in a single figure with subplots with shared legend
    """
    n_shapes = len(shape_names)
    fig, axes = plt.subplots(n_shapes, 1, figsize=(12, 3.5*n_shapes))
    
    if n_shapes == 1:
        axes = [axes]
    
    all_methods = set()
    for shape_name in shape_names:
        df = shape_dataframes[shape_name]
        all_methods.update(df['method'].unique())
    
    all_methods = sorted(list(all_methods))
    
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
    
    method_colors = {
        "Cudnn": "#A46454",
        "DRStencil": "#B3927B",
        "TCStencil": "#E3CA82",
        "ConvStencil": "#DBCC73",
        "LoRAStencil": "#A8D19E",
        "FlashFFTStencil": "#B1CCC5",
        "SPTC_half": "#7095A6"
    }
    
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    method_markers = dict(zip(all_methods, markers[:len(all_methods)]))
    
    plotted_methods = set()
    
    for i, shape_name in enumerate(shape_names):
        df = shape_dataframes[shape_name]
        ax = axes[i]
        
        methods_in_df = df['method'].unique()
        for method in all_methods:
            if method in methods_in_df:
                method_data = df[df['method'] == method]
                method_data = method_data.sort_values('dim_1')
                
                label = method_display_names.get(method, method) if method not in plotted_methods else ""
                
                if shape_name == '1D1R' or shape_name == '1D2R':
                    ax.plot(method_data['dim_1']/1024, method_data['GStencil/s'], 
                       marker=method_markers[method], linewidth=line_width, markersize=marker_size, label=label, 
                       color=method_colors[method])
                else:
                    ax.plot(method_data['dim_1'], method_data['GStencil/s'], 
                       marker=method_markers[method], linewidth=line_width, markersize=marker_size, label=label, 
                       color=method_colors[method])
                
                plotted_methods.add(method)
        
        ax.text(0.02, 0.93, f'{shape_name}', transform=ax.transAxes, 
                fontsize=title_fontsize,
                verticalalignment='top', horizontalalignment='left')
        
        ax.set_ylabel('GStencils/s', fontsize=label_fontsize)

        for spine in ax.spines.values():
            spine.set_color('black')
            spine.set_linewidth(1)

        if shape_name == '1D1R' or shape_name == '1D2R':
            ax.set_xlim(0, int(ax.get_xlim()[1]))
            interval = 8192
            ax.set_xticks([1024] + list(range(interval, int(ax.get_xlim()[1]), interval)))
            ax.set_xlabel('Problem Size (1, $2^{10}X$)', fontsize=label_fontsize)
        else:
            ax.set_xlim(0, int(ax.get_xlim()[1]))
            interval = 2048
            ax.set_xticks([512] + list(range(interval, int(ax.get_xlim()[1]), interval)))
            ax.set_xlabel('Problem Size ($X$, $X$)', fontsize=label_fontsize)

        if i == 2:
            ax.set_ylim(0, math.ceil(max(df["GStencil/s"]) / 100) * 100)
        elif i == 3:
            ax.set_ylim(0, math.ceil(max(df["GStencil/s"]) / 50) * 50)
        elif i == 4:
            ax.set_ylim(0, math.ceil(max(df["GStencil/s"]) / 25) * 25)

        ax.tick_params(axis="x", labelsize=ticks_fontsize)
        ax.tick_params(axis="y", labelsize=ticks_fontsize)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        for handle, label in zip(h, l):
            if label and label not in labels:
                handles.append(handle)
                labels.append(label)
    
    method_order = ["Cudnn", "DRStencil", "TCStencil", "ConvStencil", "LoRAStencil", "SPTC_half"]
    display_name_order = [method_display_names.get(method, method) for method in method_order]
    
    sorted_handles_labels = []
    for display_name in display_name_order:
        for i, label in enumerate(labels):
            if label == display_name:
                sorted_handles_labels.append((handles[i], label))
                break
    
    sorted_handles, sorted_labels = zip(*sorted_handles_labels) if sorted_handles_labels else ([], [])
    
    fig.legend(sorted_handles, sorted_labels, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=len(sorted_labels), fontsize=legend_fontsize, handletextpad=0.3, columnspacing=0.6, handlelength=1.2, frameon=False)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.05, hspace=0.38)  # Reduce spacing between subplots
    plt.savefig(
        "./outputs/Figure11.pdf", dpi=300, bbox_inches="tight"
    )
    plt.show()

plot_selected_shapes(shape_dataframes, ['1D1R', '1D2R', 'Box-2D1R', 'Box-2D2R', 'Box-2D3R'])