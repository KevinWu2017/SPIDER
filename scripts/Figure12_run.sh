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

mkdir -p outputs
output_file=outputs/Figure12.csv

STEPS=5

SIZE_2D_1=1280
SIZE_2D_2=1280
TIMES_2D=10000

echo "method, shape, dim_3, dim_2, dim_1, iters, time, GStencil/s" | tee $output_file

{   
    CURRENT_SIZE_2D_DIM=$SIZE_2D_1
    for i in $(seq 1 $STEPS); do
        CURRENT_SIZE_2D_1=$CURRENT_SIZE_2D_DIM
        CURRENT_SIZE_2D_2=$CURRENT_SIZE_2D_DIM
        CURRENT_SIZE_2D_DIM=$((CURRENT_SIZE_2D_DIM * 2))

        # Extract TCStencil results
        python ./scripts/extract_TCStencil_result.py --shape box2d2r --dim1 $CURRENT_SIZE_2D_1 --dim2 $CURRENT_SIZE_2D_2

        ./build/bin/stencil_2d_half_dense -M $CURRENT_SIZE_2D_1 -N $CURRENT_SIZE_2D_2 -T $TIMES_2D --profile
        ./build/bin/stencil_2d_half_sparse_for_ablation -M $CURRENT_SIZE_2D_1 -N $CURRENT_SIZE_2D_2 -T $TIMES_2D --profile
        ./build/bin/stencil_2d_half_sparse -M $CURRENT_SIZE_2D_1 -N $CURRENT_SIZE_2D_2 -T $TIMES_2D --profile
    done
} | tee -a $output_file