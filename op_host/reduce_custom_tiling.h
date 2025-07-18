#ifndef ADD_CUSTOM_TILING_H
#define ADD_CUSTOM_TILING_H
#include "register/tilingdata_base.h"

namespace optiling
{
  BEGIN_TILING_DATA_DEF(ReduceCustomTilingData)
  TILING_DATA_FIELD_DEF(uint32_t, totalLength);
  TILING_DATA_FIELD_DEF(uint32_t, outLength);
  END_TILING_DATA_DEF;

  REGISTER_TILING_DATA_CLASS(ReduceCustom, ReduceCustomTilingData)
}
#endif