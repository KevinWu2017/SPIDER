#!/bin/bash

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

# ~ 30 minutes

mkdir -p outputs
output_file=outputs/Figure11.csv

SIZE_1D_1=1024
SIZE_1D_2_ARRAY=(1024)
for i in $(seq 1 20); do
    SIZE_1D_2_ARRAY+=($((2048 * i)))
done
TIMES_1D=10000

SIZE_2D_1=512
SIZE_2D_2=512
STEP_2D=20
TIMES_2D=10240

echo "method, shape, dim_3, dim_2, dim_1, iters, time, GStencil/s" | tee $output_file

{
    for SIZE_1D_2 in ${SIZE_1D_2_ARRAY[@]}; do
        CURRENT_SIZE_1D_2=$((SIZE_1D_2))
        CURRENT_SIZE_1D=$((SIZE_1D_1 * CURRENT_SIZE_1D_2))

        ./third_party/ConvStencil/build/cudnn_1d3p          $CURRENT_SIZE_1D $TIMES_1D
        ./third_party/ConvStencil/build/cudnn_1d5p          $CURRENT_SIZE_1D $TIMES_1D

        ./third_party/LoRAStencil/build/lorastencil_1d 1d1r $CURRENT_SIZE_1D $TIMES_1D 
        ./third_party/LoRAStencil/build/lorastencil_1d 1d2r $CURRENT_SIZE_1D $TIMES_1D

        ./third_party/ConvStencil/build/convstencil_1d 1d1r $CURRENT_SIZE_1D $TIMES_1D
        ./third_party/ConvStencil/build/convstencil_1d 1d2r $CURRENT_SIZE_1D $TIMES_1D
        
        ./build/bin/stencil_1d_half_sparse -M $SIZE_1D_1 -N $CURRENT_SIZE_1D_2 -T $TIMES_1D --profile
    done
    
    for i in $(seq 1 $STEP_2D); do
        CURRENT_SIZE_2D_1=$((SIZE_2D_1 * i))
        CURRENT_SIZE_2D_2=$((SIZE_2D_2 * i))
    
        ./third_party/ConvStencil/build/cudnn_box2d9p           $CURRENT_SIZE_2D_1 $CURRENT_SIZE_2D_2 $TIMES_2D
        ./third_party/ConvStencil/build/cudnn_box2d25p           $CURRENT_SIZE_2D_1 $CURRENT_SIZE_2D_2 $TIMES_2D
    
        ./third_party/LoRAStencil/build/lorastencil_2d box2d1r  $CURRENT_SIZE_2D_1 $CURRENT_SIZE_2D_2 $TIMES_2D
        ./third_party/LoRAStencil/build/lorastencil_2d box2d3r  $CURRENT_SIZE_2D_1 $CURRENT_SIZE_2D_2 $TIMES_2D
    
        ./third_party/ConvStencil/build/convstencil_2d box2d1r  $CURRENT_SIZE_2D_1 $CURRENT_SIZE_2D_2 $TIMES_2D
        ./third_party/ConvStencil/build/convstencil_2d box2d3r  $CURRENT_SIZE_2D_1 $CURRENT_SIZE_2D_2 $TIMES_2D
    
        ./build/bin/stencil_2d_half_sparse -M $CURRENT_SIZE_2D_1 -N $CURRENT_SIZE_2D_2 -T $TIMES_2D --profile
    done

} | tee -a $output_file
