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

#include <iostream>
#include <iomanip>
#include <cmath>
#include "argparse/argparse.hpp"
#include "gpu_7r_half.h"

#include <cuda_fp16.h>

#define IDX(x, y, ldm) ((x) * (ldm) + (y))

void cpu_vector_reduction_7r(const TYPE *__restrict__ in, TYPE *__restrict__ out, TYPE *__restrict__ params, const int m, const int n){
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            half sum = 0;
            for (int k = 0; k < 15; k++) {
                sum += in[IDX(i+k+1, j, n)] * params[k];
            }
            out[IDX(i, j, n)] = sum;
        }
    }
}

void check_result(TYPE *params, TYPE *in, TYPE *out, int m, int n) {
    TYPE *cpu_out = (TYPE *)malloc(m*n*sizeof(TYPE));
    TYPE *cpu_params = (TYPE *)malloc(15*sizeof(TYPE));
    for (int i = 0; i < 15; i++) {
        cpu_params[i] = params[i+1];
    }

    gpu_vector_reduction_7r(in, out, params, m, n, 1, true);

    cpu_vector_reduction_7r(in, cpu_out, cpu_params, m, n);

    bool isCorrect = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            if (fabs(__half2float(out[IDX(i+8, j, n)] - cpu_out[IDX(i, j, n)])) > 0.01) {
                isCorrect = false;
                std::cout << "Index: [" << i << ", " << j << "], Expected Result: " << __half2float(cpu_out[IDX(i, j, n)]) << ", GPU Result: " << __half2float(out[IDX(i+8, j, n)]) << std::endl;
                std::cout << "Result: " << (isCorrect ? "Correct" : "Wrong") << std::endl;
                return;
            }
        }
    }
    std::cout << "Result: " << (isCorrect ? "Correct" : "Wrong") << std::endl;

    free(cpu_out);
    free(cpu_params);
}

int main(int argc, const char **argv) {
    argparse::ArgumentParser parser("vector_reduction");
    parser.add_argument("-M").help("Dimension M of the input matrix").default_value(16384).scan<'i', int>();
    parser.add_argument("-N").help("Dimension N of the input matrix").default_value(16384).scan<'i', int>();
    parser.add_argument("-T", "--times").help("Repeat times of the Convolution operation").default_value(100).scan<'i', int>();
    parser.add_argument("--check").help("Check the result").default_value(false).implicit_value(true);
    parser.add_argument("--profile").help("Profile the kernel").default_value(false).implicit_value(true);
    parser.parse_args(argc, argv);
    
    int m = parser.get<int>("-M");
    int n = parser.get<int>("-N");
    int times = parser.get<int>("--times");
    
    std::cout << "SPTC_half, " << "1d" << ", 1, " << n << ", " << m << ", " << times << ", ";


    TYPE params[16*32] = {0.0};
    for (int i = 1; i < 16; i++) {
        TYPE param = i % 13 + 1;
        for (int j = 0; j < 16; j++) {
            params[IDX(j, i+j, 32)] = param;
        }
    }

    int rows = m + 2*8;
    int cols = n;
    TYPE *in = (TYPE *)malloc(rows*cols*sizeof(TYPE));
    TYPE *out = (TYPE *)malloc(rows*cols*sizeof(TYPE));
    for (int i = 0; i < m+8*2; i++) {
        for (int j = 0; j < cols; j++) {
            if(i < 8 || i >= m+8) {
                in[IDX(i, j, cols)] = 0;
            } else {
                in[IDX(i, j, cols)] = IDX(i, j, cols) % 23;
            }
        }
    }

    for (int i = 0; i < rows*cols; i++)
    {
        out[i] = 0;
    }

    if (parser.get<bool>("--check")) {
        std::cout << "Checking result..." << std::endl;
        check_result(params, in, out, m, n);    
    }
    
    if (parser.get<bool>("--profile")) {
        gpu_vector_reduction_7r(in, out, params, m, n, times, false);
    }

    free(in);
    free(out);

    return 0;
}