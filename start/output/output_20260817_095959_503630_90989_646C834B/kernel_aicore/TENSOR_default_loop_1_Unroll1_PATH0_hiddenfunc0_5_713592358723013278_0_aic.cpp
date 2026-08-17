#define __TILE_FWK_AICORE__ 1
#include "../kernel_aicpu/expression_0.h"
#include "TileOpImpl.h"

// funcHash: 876769220344535184

extern "C" [aicore] void TENSOR_default_loop_1_Unroll1_PATH0_hiddenfunc0_5_0_4503599627370496(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam) {
bfloat16_t __cbuf__ *L1_S0_E4096 = (bfloat16_t __cbuf__ *)get_imm(0x0); // size: 0x1000
bfloat16_t *L1_S0_E4096_T = (bfloat16_t *)get_imm(0x0); // size: 0x1000
bfloat16_t __ca__ *L0A_S0_E4096 = (bfloat16_t __ca__ *)get_imm(0x0); // size: 0x1000
bfloat16_t *L0A_S0_E4096_T = (bfloat16_t *)get_imm(0x0); // size: 0x1000
bfloat16_t __cbuf__ *L1_S4096_E12288 = (bfloat16_t __cbuf__ *)get_imm(0x1000); // size: 0x2000
bfloat16_t *L1_S4096_E12288_T = (bfloat16_t *)get_imm(0x1000); // size: 0x2000
bfloat16_t __cb__ *L0B_S0_E8192 = (bfloat16_t __cb__ *)get_imm(0x0); // size: 0x2000
bfloat16_t *L0B_S0_E8192_T = (bfloat16_t *)get_imm(0x0); // size: 0x2000
bfloat16_t __cbuf__ *L1_S12288_E16384 = (bfloat16_t __cbuf__ *)get_imm(0x3000); // size: 0x1000
bfloat16_t *L1_S12288_E16384_T = (bfloat16_t *)get_imm(0x3000); // size: 0x1000
bfloat16_t __cbuf__ *L1_S16384_E24576 = (bfloat16_t __cbuf__ *)get_imm(0x4000); // size: 0x2000
bfloat16_t *L1_S16384_E24576_T = (bfloat16_t *)get_imm(0x4000); // size: 0x2000
float __cc__ *L0C_S0_E8192 = (float __cc__ *)get_imm(0x0); // size: 0x2000
float *L0C_S0_E8192_T = (float *)get_imm(0x0); // size: 0x2000
bfloat16_t __ca__ *L0A_S61440_E65536 = (bfloat16_t __ca__ *)get_imm(0xf000); // size: 0x1000
bfloat16_t *L0A_S61440_E65536_T = (bfloat16_t *)get_imm(0xf000); // size: 0x1000
bfloat16_t __cb__ *L0B_S57344_E65536 = (bfloat16_t __cb__ *)get_imm(0xe000); // size: 0x2000
bfloat16_t *L0B_S57344_E65536_T = (bfloat16_t *)get_imm(0xe000); // size: 0x2000
uint64_t sym_19_dim_0 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(1, 32, 2, 37, 0)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 4, 37, 2, 0);
uint64_t sym_19_dim_1 = (RUNTIME_COA_GET_PARAM_VALID_SHAPE_MAYBE_CONST(1, 64, 2, 37, 1)); //GET_PARAM_VALID_SHAPE_BY_IDX(param, 4, 37, 2, 1);
uint64_t sym_56_dim_0 = sym_19_dim_0;
uint64_t sym_56_dim_1 = sym_19_dim_1;
uint64_t sym_58_dim_0 = sym_19_dim_1;
uint64_t sym_58_dim_1 = sym_19_dim_1;
uint64_t sym_61_dim_0 = sym_19_dim_0;
uint64_t sym_61_dim_1 = sym_19_dim_1;
uint64_t sym_63_dim_0 = sym_19_dim_1;
uint64_t sym_63_dim_1 = sym_19_dim_1;
using GMTileTensorBF16Dim2_7 = TileTensor<__gm__ bfloat16_t, DynLayout2Dim, Hardware::GM>;
using L0CTileTensorFP32Dim2_6 = TileTensor<float, LocalLayout2Dim<32, 64>, Hardware::L0C>;
using L0BTileTensorBF16Dim2_5 = TileTensor<bfloat16_t, LocalLayout2Dim<64, 64>, Hardware::L0B>;
using GMTileTensorBF16Dim2_4 = TileTensor<__gm__ bfloat16_t, DynLayout2Dim, Hardware::GM>;
using L1TileTensorBF16Dim2_3 = TileTensor<bfloat16_t, LocalLayout2Dim<64, 64>, Hardware::L1>;
using L0ATileTensorBF16Dim2_2 = TileTensor<bfloat16_t, LocalLayout2Dim<32, 64>, Hardware::L0A>;
using GMTileTensorBF16Dim2_1 = TileTensor<__gm__ bfloat16_t, DynLayout2Dim, Hardware::GM>;
using L1TileTensorBF16Dim2_0 = TileTensor<bfloat16_t, LocalLayout2Dim<32, 64>, Hardware::L1>;
GMTileTensorBF16Dim2_7 gmTensor_23((__gm__ bfloat16_t*)GET_PARAM_ADDR(param, 4, 37), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 4, 37)), Stride2Dim(GET_PARAM_STRIDE_2(param, 4, 37))));
L1TileTensorBF16Dim2_3 l1Tensor_18((uint64_t)L1_S16384_E24576_T, (Shape2Dim(sym_63_dim_0, sym_63_dim_1)));
L1TileTensorBF16Dim2_0 l1Tensor_0((uint64_t)L1_S0_E4096_T, (Shape2Dim(sym_56_dim_0, sym_56_dim_1)));
GMTileTensorBF16Dim2_1 gmTensor_1((__gm__ bfloat16_t*)GET_PARAM_ADDR(param, 1, 10), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 1, 10)), Stride2Dim(GET_PARAM_STRIDE_2(param, 1, 10))));
L0BTileTensorBF16Dim2_5 l0bTensor_17((uint64_t)L0B_S57344_E65536_T, (Shape2Dim((RUNTIME_GetViewValidShapeDim(sym_63_dim_0, 0, 64)), (RUNTIME_GetViewValidShapeDim(sym_63_dim_1, 0, 64)))));
L0ATileTensorBF16Dim2_2 l0aTensor_2((uint64_t)L0A_S0_E4096_T, (Shape2Dim((RUNTIME_GetViewValidShapeDim(sym_56_dim_0, 0, 32)), (RUNTIME_GetViewValidShapeDim(sym_56_dim_1, 0, 64)))));
L1TileTensorBF16Dim2_0 l1Tensor_16((uint64_t)L1_S12288_E16384_T, (Shape2Dim(sym_61_dim_0, sym_61_dim_1)));
L0ATileTensorBF16Dim2_2 l0aTensor_15((uint64_t)L0A_S61440_E65536_T, (Shape2Dim((RUNTIME_GetViewValidShapeDim(sym_61_dim_0, 0, 32)), (RUNTIME_GetViewValidShapeDim(sym_61_dim_1, 0, 64)))));
GMTileTensorBF16Dim2_4 gmTensor_11((__gm__ bfloat16_t*)GET_PARAM_ADDR(param, 2, 19), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 2, 19)), Stride2Dim(GET_PARAM_STRIDE_2(param, 2, 19))));
L0CTileTensorFP32Dim2_6 l0cTensor_12((uint64_t)L0C_S0_E8192_T, (Shape2Dim((RUNTIME_GetViewValidShapeDim(sym_56_dim_0, 0, 32)), (RUNTIME_GetViewValidShapeDim(sym_58_dim_1, 0, 64)))));
L1TileTensorBF16Dim2_0 l1Tensor_3((uint64_t)L1_S0_E4096_T, (Shape2Dim(sym_56_dim_0, sym_56_dim_1)));
L1TileTensorBF16Dim2_3 l1Tensor_7((uint64_t)L1_S4096_E12288_T, (Shape2Dim(sym_58_dim_0, sym_58_dim_1)));
L0BTileTensorBF16Dim2_5 l0bTensor_6((uint64_t)L0B_S0_E8192_T, (Shape2Dim((RUNTIME_GetViewValidShapeDim(sym_58_dim_0, 0, 64)), (RUNTIME_GetViewValidShapeDim(sym_58_dim_1, 0, 64)))));
L1TileTensorBF16Dim2_3 l1Tensor_4((uint64_t)L1_S4096_E12288_T, (Shape2Dim(sym_58_dim_0, sym_58_dim_1)));
L1TileTensorBF16Dim2_0 l1Tensor_8((uint64_t)L1_S12288_E16384_T, (Shape2Dim(sym_61_dim_0, sym_61_dim_1)));
GMTileTensorBF16Dim2_4 gmTensor_5((__gm__ bfloat16_t*)GET_PARAM_ADDR(param, 0, 1), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 0, 1)), Stride2Dim(GET_PARAM_STRIDE_2(param, 0, 1))));
GMTileTensorBF16Dim2_1 gmTensor_9((__gm__ bfloat16_t*)GET_PARAM_ADDR(param, 3, 28), DynLayout2Dim(Shape2Dim(GET_PARAM_RAWSHAPE_2(param, 3, 28)), Stride2Dim(GET_PARAM_STRIDE_2(param, 3, 28))));
L1TileTensorBF16Dim2_3 l1Tensor_10((uint64_t)L1_S16384_E24576_T, (Shape2Dim(sym_63_dim_0, sym_63_dim_1)));
SUBKERNEL_PHASE1
TLoad<CopyInMode::ND2NZ, PaddingMode::NO_PADDING>(l1Tensor_0, gmTensor_1, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(0, 0, 2, 10, 0)), (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 10, 1))), GET_PARAM_RAWSHAPE_BY_IDX(param, 1, 10, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 1, 10, 2, 1));
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
SUBKERNEL_PHASE2
TExtract<0>(l0aTensor_2, l1Tensor_0, Coord2Dim(0, 0));
TLoad<CopyInMode::ND2NZ, PaddingMode::NO_PADDING>(l1Tensor_4, gmTensor_5, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 1, 0)), (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 1, 1))), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 1, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 0, 1, 2, 1));
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
TExtract<0>(l0bTensor_6, l1Tensor_4, Coord2Dim(0, 0));
set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
TLoad<CopyInMode::ND2NZ, PaddingMode::NO_PADDING>(l1Tensor_8, gmTensor_9, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(0, 0, 2, 28, 0)), (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 64, 2, 28, 1))), GET_PARAM_RAWSHAPE_BY_IDX(param, 3, 28, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 3, 28, 2, 1));
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
TLoad<CopyInMode::ND2NZ, PaddingMode::NO_PADDING>(l1Tensor_10, gmTensor_11, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 64, 2, 19, 0)), (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 19, 1))), GET_PARAM_RAWSHAPE_BY_IDX(param, 2, 19, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 2, 19, 2, 1));
set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
TMatmul<0, TransMode::CAST_NONE>(l0cTensor_12, l0aTensor_2, l0bTensor_6);
wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID2);
TExtract<0>(l0aTensor_15, l1Tensor_8, Coord2Dim(0, 0));
wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID3);
TExtract<0>(l0bTensor_17, l1Tensor_10, Coord2Dim(0, 0));
set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
pipe_barrier(PIPE_M);
TMatmul<1, TransMode::CAST_NONE>(l0cTensor_12, l0aTensor_15, l0bTensor_17);
set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
TStore<TStoreConfig<CopyOutMode::NZ2ND, 0, 0>>(gmTensor_23, l0cTensor_12, l0cTensor_12, Coord2Dim((RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(0, 0, 2, 37, 0)), (RUNTIME_COA_GET_PARAM_OFFSET_MAYBE_CONST(1, 0, 2, 37, 1))), GET_PARAM_RAWSHAPE_BY_IDX(param, 4, 37, 2, 0), GET_PARAM_RAWSHAPE_BY_IDX(param, 4, 37, 2, 1), 0);
}



// kernel compilation command:
// bisheng -c -O3 -g -x cce -std=c++17 -D__AIC__ -DSUPPORT_TILE_TENSOR -D__DAV_V220 -DMEMORY_BASE --cce-aicore-only --cce-aicore-arch=dav-c220-cube -I/usr/local/python3.12.13/lib/python3.12/site-packages/pypto/lib/include/tilefwk -I/usr/local/python3.12.13/lib/python3.12/site-packages/pypto/lib/include/tileop -I/usr/local/python3.12.13/lib/python3.12/site-packages/pypto/lib/include/tileop/arch32 -I/usr/local/python3.12.13/lib/python3.12/site-packages/pypto/lib/include -I/usr/local/Ascend/cann-9.0.0/include -mllvm -cce-aicore-stack-size=0x8000 -mllvm -cce-aicore-function-stack-size=0x8000 -mllvm -cce-aicore-record-overflow=false -mllvm -cce-aicore-addr-transform -mllvm -cce-aicore-dcci-insert-for-scalar=false  -o output/output_20260817_095959_503630_90989_646C834B/kernel_aicore/TENSOR_default_loop_1_Unroll1_PATH0_hiddenfunc0_5_713592358723013278_0_aic.o output/output_20260817_095959_503630_90989_646C834B/kernel_aicore/TENSOR_default_loop_1_Unroll1_PATH0_hiddenfunc0_5_713592358723013278_0_aic.cpp
