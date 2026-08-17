	.file	"controlFlow_host_4542232387646707748.cpp"
	.text
	.section	.pypto.entry,"ax",@progbits
	.p2align 4
	.globl	_ZN3npu8tile_fwk16ControlFlowEntryEPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseE
	.type	_ZN3npu8tile_fwk16ControlFlowEntryEPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseE, @function
_ZN3npu8tile_fwk16ControlFlowEntryEPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseE:
.LFB1204:
	.cfi_startproc
	endbr64
	pushq	%r13
	.cfi_def_cfa_offset 16
	.cfi_offset 13, -16
	movq	%rcx, %r13
	pushq	%r12
	.cfi_def_cfa_offset 24
	.cfi_offset 12, -24
	movq	%rsi, %r12
	pushq	%rbp
	.cfi_def_cfa_offset 32
	.cfi_offset 6, -32
	movq	%rdi, %rbp
	pushq	%rbx
	.cfi_def_cfa_offset 40
	.cfi_offset 3, -40
	movq	%rdx, %rbx
	subq	$8, %rsp
	.cfi_def_cfa_offset 48
	movq	$0, (%rsi)
	xorl	%esi, %esi
	call	*(%rdx)
	movq	%r12, %rsi
	movq	%rbp, %rdi
	movq	%r13, %rcx
	movq	%rax, %r8
	movq	%rbx, %rdx
	call	_ZN3npu8tile_fwk16SetExprBatch_0_0EPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseEPm@PLT
	xorl	%esi, %esi
	movq	%rbp, %rdi
	call	*8(%rbx)
	cmpq	$1, %rax
	jne	.L8
	addq	$8, %rsp
	.cfi_remember_state
	.cfi_def_cfa_offset 40
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.p2align 4,,10
	.p2align 3
.L8:
	.cfi_restore_state
	movq	%rbp, %rdi
	movq	$-1, %rsi
	call	*8(%rbx)
	addq	$8, %rsp
	.cfi_def_cfa_offset 40
	xorl	%eax, %eax
	popq	%rbx
	.cfi_def_cfa_offset 32
	popq	%rbp
	.cfi_def_cfa_offset 24
	popq	%r12
	.cfi_def_cfa_offset 16
	popq	%r13
	.cfi_def_cfa_offset 8
	ret
	.cfi_endproc
.LFE1204:
	.size	_ZN3npu8tile_fwk16ControlFlowEntryEPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseE, .-_ZN3npu8tile_fwk16ControlFlowEntryEPvPlPPFS1_S1_mEPNS0_16DevStartArgsBaseE
	.ident	"GCC: (Ubuntu 11.4.0-1ubuntu1~22.04.3) 11.4.0"
	.section	.note.GNU-stack,"",@progbits
	.section	.note.gnu.property,"a"
	.align 8
	.long	1f - 0f
	.long	4f - 1f
	.long	5
0:
	.string	"GNU"
1:
	.align 8
	.long	0xc0000002
	.long	3f - 2f
2:
	.long	0x3
3:
	.align 8
4:
