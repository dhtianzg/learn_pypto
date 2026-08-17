	.arch armv8-a
	.file	"control_flow_expr_table_0_0.cpp"
	.text
	.section	.pypto.func,"ax",@progbits
	.align	2
	.p2align 3,,7
	.global	_ZN3npu8tile_fwk16SetExprBatch_0_0EPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseEPm
	.type	_ZN3npu8tile_fwk16SetExprBatch_0_0EPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseEPm, %function
_ZN3npu8tile_fwk16SetExprBatch_0_0EPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseEPm:
.LFB1176:
	.cfi_startproc
	cbz	x4, .L1
	str	xzr, [x4]
	ldr	x0, [x1]
	str	x0, [x4, 8]
.L1:
	ret
	.cfi_endproc
.LFE1176:
	.size	_ZN3npu8tile_fwk16SetExprBatch_0_0EPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseEPm, .-_ZN3npu8tile_fwk16SetExprBatch_0_0EPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseEPm
	.ident	"GCC: (Do-Compiler V100R001C30B0032) 7.3.0"
	.section	.note.GNU-stack,"",@progbits
