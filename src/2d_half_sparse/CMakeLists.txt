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

function(cuda_add_executable target)
    add_executable(${target} ${ARGN})
    set_target_properties(
        ${target}
        PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math --ptxas-options=-v -lineinfo>)
endfunction()

cuda_add_executable(stencil_2d_half_sparse 2d_stencil_half.cu gpu_2d_7r_half.h gpu_2d_7r_half.cu)