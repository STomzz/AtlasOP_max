/**
 * @file add_custom.cpp
 *
 * Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#include "kernel_operator.h"

constexpr int32_t BUFFER_NUM = 2;

// constexpr int32_t TOTAL_LENGTH = 16*16;                            // total length of data
// constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
// constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         //32 length computed of each core
// constexpr int32_t TILE_NUM = 1;                                       // split data into 8 tiles for each core
// constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
// constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; //16 separate to 2 parts, due to double buffer

class KernelMax
{
public:
    __aicore__ inline KernelMax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t totalLength, uint32_t tileNum)
    {
        this->BLOCK_LENGTH = totalLength / AscendC::GetBlockNum();
        this->TILE_NUM = tileNum;
        this->TILE_LENGTH = BLOCK_LENGTH / tileNum / BUFFER_NUM;

        xGm.SetGlobalBuffer((__gm__ half *)x + AscendC::GetBlockIdx() * BLOCK_LENGTH, BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half *)y + AscendC::GetBlockIdx() * BLOCK_LENGTH, BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ half *)z + AscendC::GetBlockIdx() * BLOCK_LENGTH, BLOCK_LENGTH);

        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(DTYPE_X));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(DTYPE_Y));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(DTYPE_Z));
    }
    __aicore__ inline void Process()
    {
        int32_t loopnums = TILE_NUM * BUFFER_NUM;
        for (int32_t loop = 0; loop < loopnums; loop++)
        {
            CopyIn(loop);
            Compute(loop);
            CopyOut(loop);
        }
    }

private:
    __aicore__ inline void CopyIn(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.AllocTensor<DTYPE_X>(); // 类似new 进行堆内存分配
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.AllocTensor<DTYPE_Y>();
        AscendC::DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH); // 数据搬运
        AscendC::DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue<DTYPE_X>(xLocal); // 将堆中的数据搬运到队列中
        inQueueY.EnQue<DTYPE_Y>(yLocal);
    }
    __aicore__ inline void Compute(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_X> xLocal = inQueueX.DeQue<DTYPE_X>(); // 将队列中的在堆中申请的数据放到局部变量xlocal中
        AscendC::LocalTensor<DTYPE_Y> yLocal = inQueueY.DeQue<DTYPE_Y>();
        AscendC::LocalTensor<DTYPE_Z> zLocal = outQueueZ.AllocTensor<DTYPE_Z>(); // 堆中申请内存
        // compute
        AscendC::Max(zLocal, xLocal, yLocal, TILE_LENGTH);
        outQueueZ.EnQue<DTYPE_Z>(zLocal);
        inQueueX.FreeTensor(xLocal); // 释放堆中申请的内存
        inQueueY.FreeTensor(yLocal);
    }
    __aicore__ inline void CopyOut(int32_t progress)
    {
        AscendC::LocalTensor<DTYPE_Z> zlocal = outQueueZ.DeQue<DTYPE_Z>();
        AscendC::DataCopy(zGm[progress * TILE_LENGTH], zlocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zlocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, BUFFER_NUM> inQueueX, inQueueY;
    AscendC::TQue<AscendC::TPosition::VECOUT, BUFFER_NUM> outQueueZ;
    AscendC::GlobalTensor<DTYPE_X> xGm;
    AscendC::GlobalTensor<DTYPE_Y> yGm;
    AscendC::GlobalTensor<DTYPE_Z> zGm;

    uint32_t BLOCK_LENGTH;
    uint32_t TILE_NUM;
    uint32_t TILE_LENGTH;
};

extern "C" __global__ __aicore__ void max_custom(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    // TODO: user kernel impl
    KernelMax op;
    op.Init(x, y, z, tiling_data.totalLength, tiling_data.tileNum);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
void max_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *z)
{
    max_custom<<<blockDim, nullptr, stream>>>(x, y, z);
}
#endif
