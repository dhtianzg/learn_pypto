	.arch armv8-a
	.file	"controlFlow_dev_4542232387646707748.cpp"
	.text
	.section	.pypto.entry,"ax",@progbits
	.align	2
	.p2align 3,,7
	.global	_ZN3npu8tile_fwk16ControlFlowEntryEPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseE
	.type	_ZN3npu8tile_fwk16ControlFlowEntryEPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseE, %function
_ZN3npu8tile_fwk16ControlFlowEntryEPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseE:
.LFB1176:
	.cfi_startproc
	stp	x29, x30, [sp, -48]!
	.cfi_def_cfa_offset 48
	.cfi_offset 29, -48
	.cfi_offset 30, -40
	add	x29, sp, 0
	.cfi_def_cfa_register 29
	stp	x21, x22, [sp, 32]
	.cfi_offset 21, -16
	.cfi_offset 22, -8
	mov	x21, x1
	stp	x19, x20, [sp, 16]
	.cfi_offset 19, -32
	.cfi_offset 20, -24
	mov	x19, x2
	ldr	x2, [x2]
	mov	x20, x0
	str	xzr, [x21]
	mov	x22, x3
	mov	x1, 0
	blr	x2
	mov	x4, x0
	mov	x3, x22
	mov	x2, x19
	mov	x1, x21
	mov	x0, x20
	bl	_ZN3npu8tile_fwk16SetExprBatch_0_0EPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseEPm
	ldr	x2, [x19, 8]
	mov	x1, 0
	mov	x0, x20
	blr	x2
	cmp	x0, 1
	beq	.L2
	ldr	x2, [x19, 8]
	mov	x1, -1
	mov	x0, x20
	blr	x2
.L2:
	mov	x0, 0
	ldp	x19, x20, [sp, 16]
	ldp	x21, x22, [sp, 32]
	ldp	x29, x30, [sp], 48
	.cfi_restore 30
	.cfi_restore 29
	.cfi_restore 21
	.cfi_restore 22
	.cfi_restore 19
	.cfi_restore 20
	.cfi_def_cfa 31, 0
	ret
	.cfi_endproc
.LFE1176:
	.size	_ZN3npu8tile_fwk16ControlFlowEntryEPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseE, .-_ZN3npu8tile_fwk16ControlFlowEntryEPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseE
	.ident	"GCC: (Do-Compiler V100R001C30B0032) 7.3.0"
	.section	.note.GNU-stack,"",@progbits
