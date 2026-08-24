/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup gpu
 *
 * GPU CCD IK 批量烘焙（R7-VMD）：与 mmd_ccd_ik_v8.cc 同构的计算着色器。
 * 每个 global invocation 求解一帧；帧内链按顺序求解（与 CPU 顺序一致，
 * 保证逐位精度可对比）。全部数据为 MMD Y-up row-major 空间。
 */

#ifdef GPU_SHADER
#  pragma once
#  include "gpu_shader_compat.hh"
#  include "GPU_shader_shared.hh"
#endif

#include "gpu_shader_create_info.hh"

GPU_SHADER_CREATE_INFO(gpu_shader_mmd_ccd_bake)
LOCAL_GROUP_SIZE(64, 1, 1)
STORAGE_BUF(0, read, MmdBakeBoneConst, bone_const_buf[])
STORAGE_BUF(1, read, MmdBakeChainConst, chain_const_buf[])
STORAGE_BUF(2, read, MmdBakeLinkConst, link_const_buf[])
STORAGE_BUF(3, read_write, MmdBakeFrameBone, frame_buf[])
STORAGE_BUF(4, read_write, MmdBakeFrameOut, frame_out_buf[])
PUSH_CONSTANT(int, bone_count)
PUSH_CONSTANT(int, chain_count)
PUSH_CONSTANT(int, frame_count)
PUSH_CONSTANT(int, link_count)
COMPUTE_SOURCE("gpu_shader_mmd_ccd_bake_comp.glsl")
GPU_SHADER_CREATE_END()
