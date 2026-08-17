extern "C" [aicore] void TENSOR_default_loop_1_Unroll1_PATH0_hiddenfunc0_5_0_4503599627370496(CoreFuncParam* param, int64_t GMStackBase, __gm__ int64_t *hcclContext, __gm__ GMTensorInfo* oriAddrParam);
__attribute__((always_inline)) inline __aicore__ void CallSubFuncTask(uint64_t funcIdx, CoreFuncParam *param, int64_t gmStackAddr, __gm__ int64_t *hcclContext) {
    switch (funcIdx) {
        case 1: {
            TENSOR_default_loop_1_Unroll1_PATH0_hiddenfunc0_5_0_4503599627370496(param, gmStackAddr, hcclContext, nullptr);
            break;
        }
        default:
            return;
    };
    return;
}
