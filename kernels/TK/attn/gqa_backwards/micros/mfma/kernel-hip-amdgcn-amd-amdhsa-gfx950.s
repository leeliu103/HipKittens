	.text
	.amdgcn_target "amdgcn-amd-amdhsa--gfx950"
	.amdhsa_code_object_version 6
	.protected	_Z8micro_tk13micro_globals ; -- Begin function _Z8micro_tk13micro_globals
	.globl	_Z8micro_tk13micro_globals
	.p2align	8
	.type	_Z8micro_tk13micro_globals,@function
_Z8micro_tk13micro_globals:             ; @_Z8micro_tk13micro_globals
; %bb.0:                                ; %entry
	s_cmp_lg_u32 0, -1
	s_cselect_b32 s6, 0, 0
	s_and_b32 s7, s6, -16
	s_load_dwordx2 s[4:5], s[0:1], 0x0
	s_load_dword s11, s[0:1], 0x20
	s_load_dwordx2 s[8:9], s[0:1], 0x30
	s_load_dword s12, s[0:1], 0x50
	s_mov_b32 s3, 0
	s_and_b32 s2, s6, 15
	s_add_i32 s7, s7, 16
	s_cmp_eq_u64 s[2:3], 0
	s_cselect_b32 s13, s6, s7
	s_add_i32 s14, s13, 0x400
	v_lshlrev_b32_e32 v1, 4, v0
	v_and_b32_e32 v2, 32, v0
	s_and_b32 s6, s14, -16
	v_bitop3_b32 v2, v1, v2, 48 bitop3:0x6c
	v_lshrrev_b32_e32 v1, 1, v0
	s_and_b32 s2, s14, 15
	s_add_i32 s15, s6, 16
	s_waitcnt lgkmcnt(0)
	s_lshl_b32 s6, s11, 5
	s_lshl_b32 s10, s12, 5
	v_and_b32_e32 v1, 30, v1
	s_cmp_eq_u64 s[2:3], 0
	s_mov_b32 s7, 0x110000
	v_mad_u64_u32 v[4:5], s[2:3], v1, s11, v[2:3]
	s_mov_b32 m0, s13
	s_mov_b32 s11, s7
	buffer_load_dwordx4 v4, s[4:7], 0 offen lds
	s_cselect_b32 s4, s14, s15
	v_mad_u64_u32 v[2:3], s[2:3], v1, s12, v[2:3]
	s_mov_b32 m0, s4
	v_lshl_or_b32 v1, v0, 6, v0
	buffer_load_dwordx4 v2, s[8:11], 0 offen lds
	s_load_dwordx2 s[2:3], s[0:1], 0x60
	s_load_dword s5, s[0:1], 0x80
	v_lshlrev_b32_e32 v2, 2, v0
	v_and_b32_e32 v2, 32, v2
	s_movk_i32 s0, 0x3f0
	v_bitop3_b32 v1, v1, v2, s0 bitop3:0x6c
	v_and_b32_e32 v6, 15, v0
	v_lshrrev_b32_e32 v0, 2, v0
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	; wave barrier
	v_add_u32_e32 v2, s13, v1
	;;#ASMSTART
	ds_read_b128 a[10:13], v2 offset:0
	;;#ASMEND
	v_add_u32_e32 v1, s4, v1
	v_and_b32_e32 v0, 12, v0
	;;#ASMSTART
	ds_read_b128 a[20:23], v1 offset:0
	;;#ASMEND
	s_waitcnt lgkmcnt(0)
	v_mad_u64_u32 v[0:1], s[0:1], v0, s5, v[6:7]
	v_ashrrev_i32_e32 v1, 31, v0
	v_mov_b32_e32 v2, 0
	v_lshl_add_u64 v[6:7], v[0:1], 2, s[2:3]
	v_add_u32_e32 v0, s5, v0
	v_mov_b32_e32 v3, v2
	v_mov_b32_e32 v4, v2
	v_mov_b32_e32 v5, v2
	v_ashrrev_i32_e32 v1, 31, v0
	s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)
	; wave barrier
	;;#ASMSTART
	v_mfma_f32_16x16x32_bf16 v[2:5], a[10:13], a[20:23], v[2:5]
	;;#ASMEND
	;;#ASMSTART
	s_nop 2
	;;#ASMEND
	global_store_dword v[6:7], v2, off
	v_lshl_add_u64 v[6:7], v[0:1], 2, s[2:3]
	v_add_u32_e32 v0, s5, v0
	v_ashrrev_i32_e32 v1, 31, v0
	global_store_dword v[6:7], v3, off
	v_lshl_add_u64 v[2:3], v[0:1], 2, s[2:3]
	v_add_u32_e32 v0, s5, v0
	v_ashrrev_i32_e32 v1, 31, v0
	v_lshl_add_u64 v[0:1], v[0:1], 2, s[2:3]
	global_store_dword v[2:3], v4, off
	global_store_dword v[0:1], v5, off
	s_endpgm
	.section	.rodata,"a",@progbits
	.p2align	6, 0x0
	.amdhsa_kernel _Z8micro_tk13micro_globals
		.amdhsa_group_segment_fixed_size 0
		.amdhsa_private_segment_fixed_size 0
		.amdhsa_kernarg_size 144
		.amdhsa_user_sgpr_count 2
		.amdhsa_user_sgpr_dispatch_ptr 0
		.amdhsa_user_sgpr_queue_ptr 0
		.amdhsa_user_sgpr_kernarg_segment_ptr 1
		.amdhsa_user_sgpr_dispatch_id 0
		.amdhsa_user_sgpr_kernarg_preload_length 0
		.amdhsa_user_sgpr_kernarg_preload_offset 0
		.amdhsa_user_sgpr_private_segment_size 0
		.amdhsa_uses_dynamic_stack 0
		.amdhsa_enable_private_segment 0
		.amdhsa_system_sgpr_workgroup_id_x 1
		.amdhsa_system_sgpr_workgroup_id_y 0
		.amdhsa_system_sgpr_workgroup_id_z 0
		.amdhsa_system_sgpr_workgroup_info 0
		.amdhsa_system_vgpr_workitem_id 0
		.amdhsa_next_free_vgpr 32
		.amdhsa_next_free_sgpr 16
		.amdhsa_accum_offset 8
		.amdhsa_reserve_vcc 0
		.amdhsa_float_round_mode_32 0
		.amdhsa_float_round_mode_16_64 0
		.amdhsa_float_denorm_mode_32 3
		.amdhsa_float_denorm_mode_16_64 3
		.amdhsa_dx10_clamp 1
		.amdhsa_ieee_mode 1
		.amdhsa_fp16_overflow 0
		.amdhsa_tg_split 0
		.amdhsa_exception_fp_ieee_invalid_op 0
		.amdhsa_exception_fp_denorm_src 0
		.amdhsa_exception_fp_ieee_div_zero 0
		.amdhsa_exception_fp_ieee_overflow 0
		.amdhsa_exception_fp_ieee_underflow 0
		.amdhsa_exception_fp_ieee_inexact 0
		.amdhsa_exception_int_div_zero 0
	.end_amdhsa_kernel
	.text
.Lfunc_end0:
	.size	_Z8micro_tk13micro_globals, .Lfunc_end0-_Z8micro_tk13micro_globals
                                        ; -- End function
	.section	.AMDGPU.csdata,"",@progbits
; Kernel info:
; codeLenInByte = 440
; NumSgprs: 22
; NumVgprs: 8
; NumAgprs: 24
; TotalNumVgprs: 32
; ScratchSize: 0
; MemoryBound: 0
; FloatMode: 240
; IeeeMode: 1
; LDSByteSize: 0 bytes/workgroup (compile time only)
; SGPRBlocks: 2
; VGPRBlocks: 3
; NumSGPRsForWavesPerEU: 22
; NumVGPRsForWavesPerEU: 32
; AccumOffset: 8
; Occupancy: 8
; WaveLimiterHint : 0
; COMPUTE_PGM_RSRC2:SCRATCH_EN: 0
; COMPUTE_PGM_RSRC2:USER_SGPR: 2
; COMPUTE_PGM_RSRC2:TRAP_HANDLER: 0
; COMPUTE_PGM_RSRC2:TGID_X_EN: 1
; COMPUTE_PGM_RSRC2:TGID_Y_EN: 0
; COMPUTE_PGM_RSRC2:TGID_Z_EN: 0
; COMPUTE_PGM_RSRC2:TIDIG_COMP_CNT: 0
; COMPUTE_PGM_RSRC3_GFX90A:ACCUM_OFFSET: 1
; COMPUTE_PGM_RSRC3_GFX90A:TG_SPLIT: 0
	.text
	.p2alignl 6, 3212836864
	.fill 256, 4, 3212836864
	.type	__hip_cuid_d554d5163bda5de8,@object ; @__hip_cuid_d554d5163bda5de8
	.section	.bss,"aw",@nobits
	.globl	__hip_cuid_d554d5163bda5de8
__hip_cuid_d554d5163bda5de8:
	.byte	0                               ; 0x0
	.size	__hip_cuid_d554d5163bda5de8, 1

	.ident	"AMD clang version 19.0.0git (ssh://github-emu/AMD-Lightning-Internal/llvm-project  25164 2b159522a6e9b34fe13b1d7b4c4ae751ef122765)"
	.section	".note.GNU-stack","",@progbits
	.addrsig
	.addrsig_sym __shm
	.addrsig_sym __hip_cuid_d554d5163bda5de8
	.amdgpu_metadata
---
amdhsa.kernels:
  - .agpr_count:     24
    .args:
      - .offset:         0
        .size:           144
        .value_kind:     by_value
    .group_segment_fixed_size: 0
    .kernarg_segment_align: 8
    .kernarg_segment_size: 144
    .language:       OpenCL C
    .language_version:
      - 2
      - 0
    .max_flat_workgroup_size: 64
    .name:           _Z8micro_tk13micro_globals
    .private_segment_fixed_size: 0
    .sgpr_count:     22
    .sgpr_spill_count: 0
    .symbol:         _Z8micro_tk13micro_globals.kd
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     32
    .vgpr_spill_count: 0
    .wavefront_size: 64
amdhsa.target:   amdgcn-amd-amdhsa--gfx950
amdhsa.version:
  - 1
  - 2
...

	.end_amdgpu_metadata
