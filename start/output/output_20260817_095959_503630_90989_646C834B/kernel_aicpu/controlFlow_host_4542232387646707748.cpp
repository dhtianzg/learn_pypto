#define __TILE_FWK_AICPU__ 1
#include <stdint.h>
#include "expression_0.h"
#include "tilefwk/aikernel_data.h"
#include "tilefwk/aicpu_runtime.h"
#include "tilefwk/aicpu_distributed.h"
#include "control_flow_expr_table.h"
#define LOOP(idx, b, e, s) for (int64_t idx = (b), idxEnd = (e), idxStep = (s); idx < idxEnd; idx += idxStep)
namespace npu::tile_fwk {
// /data/tianzg/source_code/learn_pypto/examples/01_beginner/basic/basic_ops.py:67
// #name: TENSOR_matmul_kernel #hash: 4542232387646707748 #magic: 2
__attribute__((section(".pypto.entry")))
uint64_t ControlFlowEntry(void *ctx, int64_t *symbolTable, RuntimeCallEntryType runtimeCallList[], DevStartArgsBase *startArgs) {
  // hash=17071489252391066297
  LOOP(VAR_loop_idx_0, 0, 1, 1) {
    VALUE_loop_idx_0 = VAR_loop_idx_0;
    // #name: TENSOR_default_loop_1_Unroll1_PATH0 #hash: 6551093533333471805 #magic: 4
      // /usr/local/python3.12.13/lib/python3.12/site-packages/pypto/frontend/parser/parser.py:871
      // #name: TENSOR_default_loop_1_Unroll1_PATH0_hiddenfunc0 #hash: 713592358723013278 #magic: 5
      // #name: TENSOR_default_loop_1_Unroll1_PATH0_hiddenfunc0_root #hash: 15270094024039373665 #magic: 7
      uint64_t *exprList0 = (uint64_t *)RUNTIME_RootAlloc(0ULL);
      SetExprBatch_0_0(ctx, symbolTable, runtimeCallList, startArgs, exprList0);
      RUNTIME_RootStitch(0ULL);
  }
  RUNTIME_RootStitch(RUNTIME_FUNCKEY_FINISH); // Notify finish 
  return 0;
}
} // namespace npu::tile_fwk
