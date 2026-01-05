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

# ~ 6 minutes

mkdir -p outputs
output_file=outputs/Figure10.csv

SIZE_1D_1=1024
SIZE_1D_2=10000
SIZE_1D=$((SIZE_1D_1 * SIZE_1D_2))
TIMES_1D=100000

SIZE_2D_1=10240
SIZE_2D_2=10240
TIMES_2D=10240

echo "method, shape, dim_3, dim_2, dim_1, iters, time, GStencil/s" | tee $output_file

{
    # Bench Cudnn 
    ./third_party/ConvStencil/build/cudnn_1d3p          $SIZE_1D $TIMES_1D
    ./third_party/ConvStencil/build/cudnn_1d5p          $SIZE_1D $TIMES_1D
    ./third_party/ConvStencil/build/cudnn_box2d9p           $SIZE_2D_1 $SIZE_2D_2 $TIMES_2D
    ./third_party/ConvStencil/build/cudnn_box2d25p           $SIZE_2D_1 $SIZE_2D_2 $TIMES_2D
    ./third_party/ConvStencil/build/cudnn_box2d49p          $SIZE_2D_1 $SIZE_2D_2 $TIMES_2D

    # Bench LoRAStencil
    ./third_party/LoRAStencil/build/lorastencil_1d 1d1r $SIZE_1D $TIMES_1D 
    ./third_party/LoRAStencil/build/lorastencil_1d 1d2r $SIZE_1D $TIMES_1D
    ./third_party/LoRAStencil/build/lorastencil_2d box2d1r  $SIZE_2D_1 $SIZE_2D_2 $TIMES_2D
    ./third_party/LoRAStencil/build/lorastencil_2d box2d3r  $SIZE_2D_1 $SIZE_2D_2 $TIMES_2D

    # Bench ConvStencil
    ./third_party/ConvStencil/build/convstencil_1d 1d1r $SIZE_1D $TIMES_1D
    ./third_party/ConvStencil/build/convstencil_1d 1d2r $SIZE_1D $TIMES_1D
    ./third_party/ConvStencil/build/convstencil_2d box2d1r  $SIZE_2D_1 $SIZE_2D_2 $TIMES_2D
    ./third_party/ConvStencil/build/convstencil_2d box2d3r  $SIZE_2D_1 $SIZE_2D_2 $TIMES_2D

    # Bench SPTC
    ./build/bin/stencil_1d_half_sparse -M $SIZE_1D_1 -N $SIZE_1D_2 -T $TIMES_1D --profile 
    ./build/bin/stencil_2d_half_sparse -M $SIZE_2D_1 -N $SIZE_2D_2 -T $TIMES_2D --profile
} | tee -a $output_file