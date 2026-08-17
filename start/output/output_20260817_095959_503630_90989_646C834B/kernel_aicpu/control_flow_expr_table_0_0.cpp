#define __TILE_FWK_AICPU__ 1
#include <stdint.h>

#include "expression_0.h"
#include "tilefwk/aikernel_data.h"
#include "tilefwk/aicpu_runtime.h"
#include "tilefwk/aicpu_distributed.h"
namespace npu::tile_fwk {

__attribute__((section(".pypto.func")))
void SetExprBatch_0_0(void *ctx, int64_t *symbolTable, RuntimeCallEntryType runtimeCallList[], DevStartArgsBase *startArgs, uint64_t *exprList) {
    RUNTIME_SetExpr(exprList, 0, (RUNTIME_Select((RUNTIME_And(1, 0)), 1, 0)));
    RUNTIME_SetExpr(exprList, 1, VALUE_loop_idx_0);
}

} // namespace npu::tile_fwk
