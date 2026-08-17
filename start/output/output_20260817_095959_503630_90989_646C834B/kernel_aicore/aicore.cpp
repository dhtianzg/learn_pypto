
#include "tilefwk/aicore_entry.h"

extern "C" __global__ __aicore__ void KERNEL_ENTRY(__OPTYPE__, __TILINGKEY__)(int64_t ffts_addr, int64_t inputs,
        int64_t outputs, int64_t workspace, int64_t tilingdata, int64_t cfgdata) {
    return KernelEntry(ffts_addr, inputs, outputs, workspace, tilingdata, cfgdata);
}
