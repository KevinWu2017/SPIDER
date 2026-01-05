//
// Copyright (c) 2025 Qiqi Gu (qiqi.gu@sjtu.edu.cn), Chenpeng Wu (cpwu_sjtu@sjtu.edu.cn). 
// 
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// 
//     http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

#pragma once

#include <cuda_fp16.h>

#define TYPE half

#define CUDAKERNELCHECK(expr)                                                               \
    do                                                                                        \
    {                                                                                         \
        expr;                                                                                 \
                                                                                              \
        cudaError_t __err = cudaGetLastError();                                               \
        if (__err != cudaSuccess)                                                             \
        {                                                                                     \
            printf("Line %d: '%s' failed: %s\n", __LINE__, #expr, cudaGetErrorString(__err)); \
            abort();                                                                          \
        }                                                                                     \
    } while (0)


#include <stdio.h>

#define CUDA_CHECK(call)                              \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

static const int metadata_template[16 * 16] = {
    1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 0, 2, 0, 2, 0, 2,
    0, 3, 1, 3, 1, 3, 1, 3, 1, 2, 0, 2, 0, 2, 0, 2,
    0, 3, 1, 3, 1, 3, 1, 3, 1, 3, 0, 2, 0, 2, 0, 2,
    0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 0, 2, 0, 2, 0, 2,
    0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 0, 2, 0, 2,
    0, 2, 0, 3, 1, 3, 1, 3, 1, 3, 1, 2, 0, 2, 0, 2,
    0, 2, 0, 3, 1, 3, 1, 3, 1, 3, 1, 3, 0, 2, 0, 2,
    0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 0, 2, 0, 2,
    0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2, 0, 2,
    0, 2, 0, 2, 0, 3, 1, 3, 1, 3, 1, 3, 1, 2, 0, 2,
    0, 2, 0, 2, 0, 3, 1, 3, 1, 3, 1, 3, 1, 3, 0, 2,
    0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 0, 2,
    0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3, 1, 2,
    0, 2, 0, 2, 0, 2, 0, 3, 1, 3, 1, 3, 1, 3, 1, 2,
    0, 2, 0, 2, 0, 2, 0, 3, 1, 3, 1, 3, 1, 3, 1, 3,
    0, 2, 0, 2, 0, 2, 0, 2, 1, 3, 1, 3, 1, 3, 1, 3
};

void gpu_vector_reduction_7r(const TYPE *__restrict__ in, TYPE *__restrict__ out, TYPE *__restrict__ params, const int input_m, const int input_n, int times, const bool check);