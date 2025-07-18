#include "kernel_operator.h"

// constexpr int32_t TOTAL_LENGTH = 16*16;                            // total length of data
// constexpr int32_t USE_CORE_NUM = 8;                                   // num of core used
// constexpr int32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;         //32 length computed of each core
// constexpr int32_t TILE_NUM = 1;                                       // split data into 8 tiles for each core
// constexpr int32_t BUFFER_NUM = 2;                                     // tensor num for each queue
// constexpr int32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM; //16 separate to 2 parts, due to double buffer

class KernelReduceMax
{
    const int32_t BUFFER_NUM = 1;

public:
    __aicore__ inline KernelReduceMax() {}
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, uint32_t totalLength, uint32_t dstDataSize)

    {
        this->totalLength = totalLength;
        this->dstDataSize = dstDataSize;

        xGm.SetGlobalBuffer((__gm__ half *)x, totalLength);
        yGm.SetGlobalBuffer((__gm__ half *)y, dstDataSize);

        pipe.InitBuffer(inQueueX, 1, totalLength * sizeof(half));
        pipe.InitBuffer(workQueue, 1, 32 * sizeof(half));
        pipe.InitBuffer(outQueueY, 1, dstDataSize * sizeof(half));
    }
    __aicore__ inline void Process()
    {
        CopyIn();
        Compute();
        CopyOut();
    }

private:
    __aicore__ inline void CopyIn()
    {
        AscendC::LocalTensor<half> xLocal = inQueueX.AllocTensor<half>(); // 类似new 进行堆内存分配
        AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        AscendC::DataCopy(xLocal, xGm, totalLength); // 数据搬运
        inQueueX.EnQue<half>(xLocal);
        // 将堆中的数据搬运到队列中
    }
    __aicore__ inline void Compute()

    {
        AscendC::LocalTensor<half> xLocal = inQueueX.DeQue<half>(); // 将队列中的在堆中申请的数据放到局部变量xlocal中
        AscendC::LocalTensor<half> yLocal = outQueueY.AllocTensor<half>();
        AscendC::LocalTensor<half> workLocal = workQueue.AllocTensor<half>();
        // compute
        AscendC::ReduceMax(xLocal, yLocal, workLocal, totalLength, true);
        outQueueY.EnQue<half>(yLocal);
        inQueueX.FreeTensor(xLocal);
        workQueue.FreeTensor(workLocal);
    }

    __aicore__ inline void CopyOut()
    {
        AscendC::LocalTensor<half> yLocal = outQueueY.DeQue<half>();
        AscendC::DataCopy(yGm, yLocal, dstDataSize);
        outQueueY.FreeTensor(yLocal);
    }

private:
    AscendC::TPipe pipe;
    AscendC::TQue<AscendC::TPosition::VECIN, 1> inQueueX;
    AscendC::TQue<AscendC::TPosition::VECOUT, 1> outQueueY, workQueue;
    AscendC::GlobalTensor<half> xGm, yGm;
    uint32_t dstDataSize;
    uint32_t totalLength;
};
extern "C" __global__ __aicore__ void reduce_custom(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    GET_TILING_DATA(tiling_data, tiling);
    KernelReduceMax op;
    op.Init(x, y, tiling_data.totalLength, tiling_data.outLength);
    op.Process();
    // TODO: user kernel impl
}

#ifndef ASCENDC_CPU_DEBUG
void reduce_custom_do(uint32_t blockDim, void *stream, uint8_t *x, uint8_t *y, uint8_t *workspace, uint8_t *tiling)
{
    reduce_custom<<<blockDim, nullptr, stream>>>(x, y, workspace, tiling);
}
#endif