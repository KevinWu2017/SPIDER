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

#include "gpu_7r_half.h"

#define BLOCK_ITER_ROW 1
#define BLOCK_ITER_COL 1

#define BLOCK_ROW 64
#define BLOCK_COL 64

#define WARP_PER_BLOCK (BLOCK_ROW/16)
#define THREAD_PER_BLOCK WARP_PER_BLOCK * 32

#define HALO 8

#define D_BLOCK_ROW (BLOCK_ROW + HALO * 2)
#define D_BLOCK_COL (BLOCK_COL + 16)
#define D_BLOCK_COL_NOPAD BLOCK_COL

#define SPTC_M 16
#define SPTC_N 8
#define SPTC_K 16

#define IDX(x, y, ldm) ((x) * (ldm) + (y))
#define CEIL(a, b) (((a) + (b) - 1) / (b))

#define SWIZZLE(x) (x)

__global__ void
kernel_7r(const TYPE *__restrict__ in, const TYPE *__restrict__ params, const uint *__restrict__ metadata, TYPE *__restrict__ out, const int ldm) {
    int warp_idx = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;
    int warp_begin = IDX(warp_idx * 16, 0, D_BLOCK_COL);

    __shared__ TYPE sharedmem[D_BLOCK_ROW * D_BLOCK_COL];

    uint frag_params[4];
    uint frag_meta[1];
    uint frag_in[2];
    uint frag_acc[BLOCK_COL / SPTC_N][2];

    frag_meta[0] = metadata[lane_id % 2 * 8 + lane_id / 4];

    int begin = IDX(blockIdx.x * BLOCK_ROW, blockIdx.y * BLOCK_COL, ldm);
    int tid = threadIdx.x;

    *((float4 *)frag_params) = *((float4 *)(params + lane_id * 8));

    #pragma unroll
    for (int block_iter_row = 0; block_iter_row < BLOCK_ITER_ROW; block_iter_row++) {
        #pragma unroll
        for (int block_iter_col = 0; block_iter_col < BLOCK_ITER_COL; block_iter_col++) {
            int begin = IDX((blockIdx.x * BLOCK_ITER_ROW + block_iter_row) * BLOCK_ROW , (blockIdx.y * BLOCK_ITER_COL + block_iter_col) * BLOCK_COL, ldm);

            int base_addr = __cvta_generic_to_shared(sharedmem);

            #pragma unroll
            for (int i = tid; i < D_BLOCK_ROW * D_BLOCK_COL_NOPAD / 8; i += THREAD_PER_BLOCK) {
                int row = (i * 8) / D_BLOCK_COL_NOPAD;
                int col = (i * 8) % D_BLOCK_COL_NOPAD;

                int dst = base_addr + (row * D_BLOCK_COL + col) * sizeof(TYPE);
                
                asm ("cp.async.cg.shared.global [%0], [%1], 16;\n" :
                        : "r"(dst), "l"(&in[begin + IDX(row, col, ldm)]));
            }

            asm ("cp.async.commit_group;\n"::);
            asm ("cp.async.wait_group 0;\n"::);
            __syncthreads();

            #pragma unroll
            for (int i = 0; i < BLOCK_COL / SPTC_N; i++) {
                for (int j = 0; j < 2; j++) {
                    frag_acc[i][j] = 0;
                }
            }
        
            #pragma unroll
            for (int iter_k = 0; iter_k < 32 / SPTC_K; iter_k++) {
                #pragma unroll
                for (int data = 0; data < BLOCK_COL / SPTC_N; data++){
                    int iter_begin = warp_begin + IDX(iter_k*16, data*8, D_BLOCK_COL);

                    if(lane_id % 2 == 0){
                        if(iter_k == 0){
                            iter_begin += IDX(16, 0, D_BLOCK_COL);
                        } else {
                            iter_begin -= IDX(16, 0, D_BLOCK_COL);
                        }
                    }
                    
                    iter_begin = iter_begin + IDX(lane_id % 16, 0, D_BLOCK_COL);
                    int src = base_addr + iter_begin * sizeof(TYPE);

                    asm volatile(
                        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                        : "=r"(frag_in[0]), "=r"(frag_in[1])
                        : "r"(src)
                    );

                    // compute
                    if (iter_k == 0) {
                        asm volatile(
                            "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                            "{%0, %1}, {%2, %3}, {%4, %5}, "
                            "{%6,%7}, %8, 0x0;\n"
                            :"=r"(frag_acc[data][0]), "=r"(frag_acc[data][1])
                            :"r"(frag_params[0]), "r"(frag_params[1]),
                            "r"(frag_in[0]), "r"(frag_in[1]),
                            "r"(frag_acc[data][0]), "r"(frag_acc[data][1]), "r"(frag_meta[0])
                        );
                    } else {
                        asm volatile(
                            "mma.sp::ordered_metadata.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                            "{%0, %1}, {%2, %3}, {%4, %5}, "
                            "{%6,%7}, %8, 0x1;\n"
                            :"=r"(frag_acc[data][0]), "=r"(frag_acc[data][1])
                            :"r"(frag_params[2]), "r"(frag_params[3]),
                            "r"(frag_in[0]), "r"(frag_in[1]),
                            "r"(frag_acc[data][0]), "r"(frag_acc[data][1]), "r"(frag_meta[0])
                        );
                    }
                }
            }

            __syncthreads();

            int OUT_SM_BLOCK_COL = BLOCK_COL+8;
            int out_warp_begin = IDX(warp_idx * 16, 0, OUT_SM_BLOCK_COL);
            #pragma unroll
            for (int data = 0; data < BLOCK_COL / SPTC_N; data++){
                *((uint*)(sharedmem + SWIZZLE(out_warp_begin + IDX(lane_id / 4, lane_id % 4 * 2 + data * SPTC_N, OUT_SM_BLOCK_COL)))) = frag_acc[data][0];
                *((uint*)(sharedmem + SWIZZLE(out_warp_begin + IDX(lane_id / 4 + 8, lane_id % 4 * 2 + data * SPTC_N, OUT_SM_BLOCK_COL)))) = frag_acc[data][1];
            }

            __syncthreads();

            int out_offset = IDX(HALO, 0, ldm);
            #pragma unroll
            for (int i = tid; i < BLOCK_ROW * BLOCK_COL / 8; i += THREAD_PER_BLOCK) {
                int row = (i * 8) / BLOCK_COL;
                int col = (i * 8) % BLOCK_COL;

                *(float4*)(out + begin + IDX(row, col, ldm) + out_offset) = *(float4*)(sharedmem + SWIZZLE(IDX(row, col, OUT_SM_BLOCK_COL)));
            }
        }
    }
}

void param_swap_to_structured_sparsity(const TYPE * params, TYPE * sparse_params){
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j=j+2) {
            sparse_params[IDX(i, j, 32)] = params[IDX(i, j+16, 32)];
            sparse_params[IDX(i, j+16, 32)] = params[IDX(i, j, 32)];
        }
    }

    for (int i = 0; i < 16*32; i=i+4){
        int zero_count = 0;
        for (int j = 0; j < 4; j++) {
            if(__half2float(sparse_params[i+j]) == 0){
                zero_count++;
            }
        }
        if (zero_count < 2){
            std::cout << "i = " << i << std::endl;
            std::cout << "param: " << __half2float(sparse_params[i]) << " " << __half2float(sparse_params[i+1]) << std::endl;
            std::cout << "Error: Not 2:4 structured sparsity" << std::endl;
            return;
        }
    }
}

void compress_params(TYPE * params, TYPE * compressed_params, uint * metadata){
    TYPE *compressed_origin_params = (TYPE *)malloc(16*16*sizeof(TYPE));

    for (int i = 0; i < 16; i++){
        for (int j = 0; j < 32; j=j+4){
            int non_zero_idx[2] = {metadata_template[i * 16 + j / 4 * 2 + 1], metadata_template[i * 16 + j / 4 * 2]};

            compressed_origin_params[i*16+j/2] = params[i*32+j+non_zero_idx[0]];
            compressed_origin_params[i*16+j/2 + 1] = params[i*32+j+non_zero_idx[1]];
            uint temp_meta = non_zero_idx[1] << 2 | non_zero_idx[0];
            metadata[i] = metadata[i] | (temp_meta << (j));
        }
    }

    for (int iter_k = 0; iter_k < 2; iter_k++){
        for (int lane_id = 0; lane_id < 32; lane_id++){
            compressed_params[lane_id * 8 + iter_k * 4 + 0] = compressed_origin_params[IDX(lane_id / 4, lane_id % 4 * 2 + iter_k * 8, 16)];
            compressed_params[lane_id * 8 + iter_k * 4 + 1] = compressed_origin_params[IDX(lane_id / 4, lane_id % 4 * 2 + 1 + iter_k * 8, 16)];
            compressed_params[lane_id * 8 + iter_k * 4 + 2] = compressed_origin_params[IDX(lane_id / 4 + 8, lane_id % 4 * 2 + iter_k * 8, 16)];
            compressed_params[lane_id * 8 + iter_k * 4 + 3] = compressed_origin_params[IDX(lane_id / 4 + 8, lane_id % 4 * 2 + 1 + iter_k * 8, 16)];
        }
    }

    free(compressed_origin_params);

    for (int i = 0; i < 8; i++) {        
        uint upper_16 = (metadata[i] >> 16) & 0xFFFF;
        uint lower_16 = metadata[i+8] & 0xFFFF;

        metadata[i] = (metadata[i] & 0x0000FFFF) | (lower_16 << 16);
        metadata[i+8] = (metadata[i+8] & 0xFFFF0000) | upper_16;
    }
}

void gpu_vector_reduction_7r(const TYPE *__restrict__ in, TYPE *__restrict__ out, TYPE *__restrict__ params, const int input_m, const int input_n, int times, const bool check){
    TYPE *sparse_params = (TYPE *)malloc(16*32*sizeof(TYPE));
    memcpy(sparse_params, params, 16*32*sizeof(TYPE));
    param_swap_to_structured_sparsity(params, sparse_params);

    TYPE *compressed_params = (TYPE *)malloc(16*16*sizeof(TYPE));
    uint *metadata = (uint *)malloc(16*sizeof(uint)); 
    memset(compressed_params, 0, 16*16*sizeof(TYPE));
    memset(metadata, 0, 16*sizeof(uint));

    compress_params(sparse_params, compressed_params, metadata);

    TYPE *compressed_params_d;
    uint *metadata_d;
    CUDA_CHECK(cudaMalloc(&compressed_params_d, 16*16*sizeof(TYPE)));
    CUDA_CHECK(cudaMalloc(&metadata_d, 16*sizeof(uint)));
    CUDA_CHECK(cudaMemcpy(compressed_params_d, compressed_params, 16*16*sizeof(TYPE), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(metadata_d, metadata, 16*sizeof(uint), cudaMemcpyHostToDevice));

    const int rows = input_m + 2 * HALO;
    const int cols = input_n;
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
        kernel_7r<<<grid_config, block_config>>>(array_d[0], compressed_params_d, metadata_d, array_d[1], cols);
        cudaDeviceSynchronize();
    } else {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        float time_elapsed = 0.0f;

        cudaEventRecord(start);
        int i = 0;
        for (; i < times; i++) {
            kernel_7r<<<grid_config, block_config>>>(array_d[i % 2], compressed_params_d, metadata_d, array_d[(i + 1) % 2], cols);
            cudaDeviceSynchronize();
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_elapsed, start, stop);

        std::cout << time_elapsed << ", " << ((double)(input_m * input_n) / (time_elapsed / 1000.0f) / 1e9) * times << std::endl;
    }
    
    CUDA_CHECK(cudaMemcpy(out, array_d[(times + 1) % 2], array_size, cudaMemcpyDeviceToHost));
}