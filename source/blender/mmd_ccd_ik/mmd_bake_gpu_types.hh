/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup mmd_ccd_ik
 *
 * MMD CCD 烘焙的 GPU 缓冲结构（Vulkan/GLSL 与 CUDA 共用，主机打包与设备
 * 布局逐字节一致）。
 *
 * 布局约定（与 GPU_shader_shared.hh 的 MmdBake* 对应，std430）：
 * - BoneConst：base@0(12), out_index@12, parent@16, flags@20, pad@24/28 = 32B
 * - ChainConst：24B
 * - LinkConst：bone@0, has_limit@4, out_index@8, pad@12, min@16, max@32 = 48B
 * - FrameBone：5×float4 = 80B（平移在 row3）
 * - FrameOut：float4 = 16B（紧凑：仅链骨，frame × link_count）
 */

#pragma once

namespace blender::mmd::bake {

struct BoneConst {
  float base_pos[3];
  int out_index; /* 链骨紧凑输出槽位，非链骨 -1 */
  int parent;
  int flags; /* 1=target 2=link 4=anchor */
  float _pad1;
  float _pad2;
};

struct ChainConst {
  int target;
  int effector;
  int link_offset;
  int link_count;
  int iterations;
  float runtime_angle;
};

struct LinkConst {
  int bone;
  int has_limit;
  int out_index; /* 该 link 的紧凑输出槽位 */
  float _pad1;
  float limit_min[4];
  float limit_max[4];
};

struct FrameBone {
  float q_base[4];
  float m0_row0[4];
  float m0_row1[4];
  float m0_row2[4];
  float m0_row3[4];
};

struct FrameOut {
  float q_current[4];
};

}  // namespace blender::mmd::bake
