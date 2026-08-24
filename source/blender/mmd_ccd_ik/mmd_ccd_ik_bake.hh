/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup mmd_ccd_ik
 *
 * GPU CCD IK 批量烘焙（R7-VMD）。
 *
 * 流程：逐帧采集 FK 姿态（与 mmd_ccd_ik_eval.cc 相同的 Blender↔MMD 转换）
 * → 打包帧缓冲 → Vulkan compute 一次求解全部帧 → 回读 q_current。
 * 同时提供 CPU 参照路径（调用 mmd_ccd_v8_solve_all_chains）用于精度对比。
 *
 * 每个 invocation 求解一帧，帧内链按顺序求解，与 CPU 顺序一致，
 * 保证逐位可对比。
 */

#pragma once

#include <vector>

namespace blender::mmd {

struct MmdCCDBakeBuffers {
  /** 骨骼常量（帧无关）：base_pos/parent/flags，按拓扑顺序。 */
  struct BoneInfo {
    float base_pos[3];
    int parent;
    int flags; /* 1=target 2=link 4=anchor */
  };
  /** 链常量（帧无关）。 */
  struct ChainInfo {
    int target_bone;
    int effector_bone;
    int link_offset;
    int link_count;
    int iterations;
    float runtime_angle;
  };
  /** link 常量（帧无关，tip→root）。 */
  struct LinkInfo {
    int bone;
    int has_limit;
    float limit_min[3];
    float limit_max[3];
  };
  /** 每帧每骨输入：q_base（MMD (w,x,y,z)）+ 初始 m0（row-major，4 行）。 */
  struct FrameBone {
    float q_base[4];
    float m0[4][4];
  };

  std::vector<BoneInfo> bones;
  std::vector<ChainInfo> chains;
  std::vector<LinkInfo> links;
  int frame_count = 0;
  /** frame × bone，行优先。 */
  std::vector<FrameBone> frames;
};

/**
 * 运行 GPU 批量求解。
 * 输入 frame 数据就地更新 m0（调用方可忽略）；输出 q_current（MMD (w,x,y,z)）。
 * 返回 false 表示 GPU 不可用（调用方回退 CPU 参照路径）。
 */
bool mmd_ccd_bake_gpu(MmdCCDBakeBuffers &buffers,
                      std::vector<float> &r_q_current /* frame×bone×4 */);

/** 上次 GPU 烘焙实际使用的后端（"CUDA" / "Vulkan" / "none"）。 */
const char *mmd_ccd_bake_gpu_last_backend();

/** CPU 参照路径：对相同输入调用 mmd_ccd_v8_solve_all_chains。 */
void mmd_ccd_bake_cpu_reference(MmdCCDBakeBuffers &buffers,
                                std::vector<float> &r_q_current /* frame×bone×4 */);

}  // namespace blender::mmd
