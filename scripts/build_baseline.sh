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

GPU_CC=$(nvidia-smi --id=0 --query-gpu=compute_cap --format=csv,noheader)

if [ "$GPU_CC" = "8.0" ]; then
    CUDA_COMPUTE_CAPABILITY=80
elif [ "$GPU_CC" = "8.6" ]; then
    CUDA_COMPUTE_CAPABILITY=86
elif [ "$GPU_CC" = "8.9" ]; then
    CUDA_COMPUTE_CAPABILITY=89
elif [ "$GPU_CC" = "9.0" ]; then
    CUDA_COMPUTE_CAPABILITY=90
else
    echo "Unsupported GPU compute capability: $GPU_CC"
    exit 1
fi

# Build LoRAStencil
rm -rf third_party/LoRAStencil/build
cmake -B third_party/LoRAStencil/build -S third_party/LoRAStencil -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CUDA_COMPUTE_CAPABILITY
cmake --build third_party/LoRAStencil/build --config Release -j

# Build ConvStencil
rm -rf third_party/ConvStencil/build
cmake -B third_party/ConvStencil/build -S third_party/ConvStencil -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=$CUDA_COMPUTE_CAPABILITY
cmake --build third_party/ConvStencil/build --config Release -j

