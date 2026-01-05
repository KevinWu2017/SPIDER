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

import argparse
import subprocess
import csv

gpu_name = ''
try:
    gpu_name = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=name", "--format=csv,noheader"],
        universal_newlines=True
    ).strip()
    gpu_name = gpu_name.replace(' ', '_')
except Exception as e:
    raise RuntimeError(f"Failed to detect GPU name: {e}")

gpu_name = ''
gpu_cc = ''
try:
    output = subprocess.check_output(
        ["nvidia-smi", "--id=0", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        universal_newlines=True
    ).strip()
    gpu_name, gpu_cc = [s.strip() for s in output.split(',')]
    gpu_name = gpu_name.replace(' ', '_')
    if gpu_cc not in ["8.0", "8.6", "8.9", "9.0", "12.0"]:
        raise RuntimeError(f"Unsupported GPU compute capability: {gpu_cc}")
    if "A100" not in gpu_name:
        raise RuntimeError(f"The default TCStencil results only support A100 GPU. in this evaluation, you need to clone, build and run TCStencil by yourself to get the results for other GPUs.")
except Exception as e:
    raise RuntimeError(f"Failed to detect GPU compute capability: {e}")

result_file = f'./outputs/TCStencil_best_A100.csv'
with open(result_file, 'r') as f:
    csv_reader = csv.reader(f)
    csv_rows = list(csv_reader)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--shape', type=str)
    argparser.add_argument('--dim1', type=int)
    argparser.add_argument('--dim2', type=int)
    args = argparser.parse_args()

    if args.dim1 != args.dim2:
        exit()

    for row in csv_rows:
        if row[0] == 'tensor' and row[1] == args.shape and int(row[2]) == args.dim1:
            time = float(row[3]) / 1e3
            secs = float(row[3]) / 1e6
            iterations = 1
            gstencil = args.dim1 * args.dim2 / secs / 1e9
            print(f"TCStencil, {args.shape}, 1, {args.dim2}, {args.dim1}, 1, {time}, {gstencil}")
            break