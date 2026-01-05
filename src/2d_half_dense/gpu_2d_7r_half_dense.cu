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
#include <chrono>
#include <bitset>

#include "gpu_2d_7r_half_dense.h"

#define BLOCK_ITER_ROW 1
#define BLOCK_ITER_COL 1

#define BLOCK_ROW 64
#define BLOCK_COL 128

#define WARP_PER_BLOCK (BLOCK_ROW/16)
#define THREAD_PER_BLOCK WARP_PER_BLOCK * 32

#define HALO 8

#define D_BLOCK_ROW (BLOCK_ROW + HALO * 2)
#define D_BLOCK_COL (BLOCK_COL + HALO * 2)
#define D_BLOCK_COL_NOPAD (BLOCK_COL + HALO * 2)

#define TC_M 16
#define TC_N 8
#define TC_K 16

#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define CEIL(a, b) (((a) + (b) - 1) / (b))

#define SWIZZLE(x) (x)

__global__ void
kernel_2d_7r(const TYPE *__restrict__ in, const TYPE *__restrict__ params, TYPE *__restrict__ out, const int ldm) {
    int warp_idx = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    __shared__ TYPE sharedmem[D_BLOCK_ROW * D_BLOCK_COL];

    uint frag_params[32 / TC_K][4];
    uint frag_in[2];
    uint frag_acc[BLOCK_COL / TC_N][2];

    int tid = threadIdx.x;
    int warp_begin = IDX(warp_idx * 16, 0, D_BLOCK_COL);

    int base_addr = __cvta_generic_to_shared(sharedmem);

    #pragma unroll
    for (int block_iter_row = 0; block_iter_row < BLOCK_ITER_ROW; block_iter_row++) {
        #pragma unroll
        for (int block_iter_col = 0; block_iter_col < BLOCK_ITER_COL; block_iter_col++) {
            int begin = IDX((blockIdx.x * BLOCK_ITER_ROW + block_iter_row) * BLOCK_ROW , (blockIdx.y * BLOCK_ITER_COL + block_iter_col) * BLOCK_COL, ldm);

            #pragma unroll
            for (int i = tid; i < D_BLOCK_ROW * D_BLOCK_COL_NOPAD / 8; i += THREAD_PER_BLOCK) {
                int row = (i * 8) / D_BLOCK_COL_NOPAD;
                int col = (i * 8) % D_BLOCK_COL_NOPAD;

                const half* src = in + (begin + IDX(row, col, ldm));
                int dst = base_addr + IDX(row, col, D_BLOCK_COL) * sizeof(TYPE);

                asm ("cp.async.cg.shared.global [%0], [%1], 16;\n" :
                        : "r"(dst), "l"(src));
            }

            asm ("cp.async.commit_group;\n"::);
            asm ("cp.async.wait_group 0;\n"::);
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < BLOCK_COL / TC_N; i++) {
                for (int j = 0; j < 2; j++) {
                    frag_acc[i][j] = 0;
                }
            }
        
            #pragma unroll
            for (int iter_n = 0; iter_n < BLOCK_COL / TC_N; iter_n++){
                #pragma unroll
                for (int iter_param = 0; iter_param < 15; iter_param++){
                    #pragma unroll
                    for (int iter_k = 0; iter_k < 32 / TC_K; iter_k++) {
                        frag_params[iter_k][0] = *((uint *)(params + IDX(lane_id / 4, iter_k * 16 + lane_id % 4 * 2, 32) + IDX(iter_param, 0, 16*32)));
                        frag_params[iter_k][1] = *((uint *)(params + IDX(lane_id / 4 + 8, iter_k * 16 + lane_id % 4 * 2, 32) + IDX(iter_param, 0, 16*32)));
                        frag_params[iter_k][2] = *((uint *)(params + IDX(lane_id / 4, iter_k * 16 + lane_id % 4 * 2 + 8, 32) + IDX(iter_param, 0, 16*32)));
                        frag_params[iter_k][3] = *((uint *)(params + IDX(lane_id / 4 + 8, iter_k * 16 + lane_id % 4 * 2 + 8, 32) + IDX(iter_param, 0, 16*32)));


                        int iter_begin = warp_begin + IDX(iter_k*TC_K, iter_n*TC_N+iter_param+1, D_BLOCK_COL);

                        ((half*)frag_in)[0] = sharedmem[iter_begin + IDX(lane_id%4*2    , lane_id / 4, D_BLOCK_COL)];
                        ((half*)frag_in)[1] = sharedmem[iter_begin + IDX(lane_id%4*2 + 1, lane_id / 4, D_BLOCK_COL)];
                        ((half*)frag_in)[2] = sharedmem[iter_begin + IDX(lane_id%4*2 + 8, lane_id / 4, D_BLOCK_COL)];
                        ((half*)frag_in)[3] = sharedmem[iter_begin + IDX(lane_id%4*2 + 9, lane_id / 4, D_BLOCK_COL)];

                        asm volatile(
                            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                            "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, "
                            "{%8,%9};\n"
                            :"=r"(frag_acc[iter_n][0]), "=r"(frag_acc[iter_n][1])
                            :"r"(frag_params[iter_k][0]), "r"(frag_params[iter_k][1]), "r"(frag_params[iter_k][2]), "r"(frag_params[iter_k][3]),
                            "r"(frag_in[0]), "r"(frag_in[1]),
                            "r"(frag_acc[iter_n][0]), "r"(frag_acc[iter_n][1])
                        );
                    }
                }
            }

            __syncthreads();

            int OUT_SM_BLOCK_COL = BLOCK_COL+8;
            int out_warp_begin = IDX(warp_idx * 16, 0, OUT_SM_BLOCK_COL);
            #pragma unroll
            for (int iter_n = 0; iter_n < BLOCK_COL / TC_N; iter_n++){
                *((uint*)(sharedmem + SWIZZLE(out_warp_begin + IDX(lane_id / 4, lane_id % 4 * 2 + iter_n * TC_N, OUT_SM_BLOCK_COL)))) = frag_acc[iter_n][0];
                *((uint*)(sharedmem + SWIZZLE(out_warp_begin + IDX(lane_id / 4 + 8, lane_id % 4 * 2 + iter_n * TC_N, OUT_SM_BLOCK_COL)))) = frag_acc[iter_n][1];
            }

            __syncthreads();

            int out_offset = IDX(HALO, HALO, ldm);
            #pragma unroll
            for (int i = tid; i < BLOCK_ROW * BLOCK_COL / 8; i += THREAD_PER_BLOCK) {
                int row = (i * 8) / BLOCK_COL;
                int col = (i * 8) % BLOCK_COL;

                *(float4*)(out + begin + IDX(row, col, ldm) + out_offset) = *(float4*)(sharedmem + SWIZZLE(IDX(row, col, OUT_SM_BLOCK_COL)));
            }
        }
    }
}

void gpu_2d_7r(const TYPE *__restrict__ in, TYPE *__restrict__ out, TYPE *__restrict__ params, const int input_m, const int input_n, int times, const bool check){
    TYPE *params_d;
    CUDA_CHECK(cudaMalloc(&params_d, 15*16*32*sizeof(TYPE)));
    CUDA_CHECK(cudaMemcpy(params_d, params, 15*16*32*sizeof(TYPE), cudaMemcpyHostToDevice));

    const int rows = input_m + 2 * HALO;
    const int cols = input_n + 2 * HALO;
    const size_t array_size = rows * cols * sizeof(TYPE);

    TYPE *array_d[2];
    CUDA_CHECK(cudaMalloc(&array_d[0], array_size));
    CUDA_CHECK(cudaMalloc(&array_d[1], array_size));
    CUDA_CHECK(cudaMemset(array_d[0], 0, array_size));
    CUDA_CHECK(cudaMemcpy(array_d[0], in, array_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(array_d[1], 0, array_size));

    const int BLOCK_M = CEIL(input_m, BLOCK_ROW * BLOCK_ITER_ROW);
    const int BLOCK_N = CEIL(input_n, BLOCK_COL * BLOCK_ITER_COL);
    dim3 grid_config(BLOCK_M, BLOCK_N);
    dim3 block_config(32 * WARP_PER_BLOCK);

    if (check) {
        times = 0;
        kernel_2d_7r<<<grid_config, block_config>>>(array_d[0], params_d, array_d[1], cols);
        cudaDeviceSynchronize();
    } else {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float time_elapsed = 0.0f;

        cudaEventRecord(start);
        int i = 0;
        for (; i < times; i++) {
            kernel_2d_7r<<<grid_config, block_config>>>(array_d[i % 2], params_d, array_d[(i + 1) % 2], cols);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_elapsed, start, stop);

        std::cout << time_elapsed << ", " << ((double)(input_m * input_n) / (time_elapsed / 1000.0f) / 1e9) * times << std::endl;
    }
    
    CUDA_CHECK(cudaMemcpy(out, array_d[(times + 1) % 2], array_size, cudaMemcpyDeviceToHost));
}