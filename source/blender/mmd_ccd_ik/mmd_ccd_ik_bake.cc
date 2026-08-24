/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "mmd_ccd_ik_bake.hh"

#include "mmd_ccd_ik_v8.hh"

#include "GPU_compute.hh"
#include "GPU_context.hh"
#include "GPU_state.hh"
#include "GPU_shader.hh"
#include "GPU_storage_buffer.hh"
#include "GPU_uniform_buffer.hh"

#include "WM_api.hh"
#include "DRW_engine.hh"

#include <cstring>

namespace blender::mmd {

/* -------------------------------------------------------------------- */
/* GPU 批量求解                                                          */
/* -------------------------------------------------------------------- */

bool mmd_ccd_bake_gpu(MmdCCDBakeBuffers &buffers, std::vector<float> &r_q_current)
{
  const int bone_count = int(buffers.bones.size());
  const int frame_count = buffers.frame_count;
  if (bone_count <= 0 || frame_count <= 0 || buffers.chains.empty()) {
    return false;
  }

  /* 无头/后台线程没有活动上下文时，通过 WM 的 offscreen GPU 初始化创建。 */
  bool temp_context = false;
  if (GPU_context_active_get() == nullptr) {
    WM_init_gpu_offscreen();
    temp_context = true;
    if (GPU_context_active_get() == nullptr) {
      DRW_gpu_context_enable();
    }
    if (GPU_context_active_get() == nullptr) {
      std::fprintf(stderr, "[BAKEDBG] context still null after enable\n");
      return false;
    }
  }

  gpu::Shader *shader = GPU_shader_create_from_info_name("gpu_shader_mmd_ccd_bake");
  if (shader == nullptr) {
    std::fprintf(stderr, "[BAKEDBG] shader create failed\n");
    return false;
  }
  std::fprintf(stderr, "[BAKEDBG] shader ok\n");

  /* 打包 GPU 结构（与 gpu_shader_mmd_ccd_infos.hh 布局一致）。
   * 注意：GLSL 侧 base_pos 是 vec3（packed_float3），std430 下 vec3 成员
   * 对齐 16 → 结构体对齐 16 → 数组步长补齐到 32B。C++ 侧必须逐字节一致
   * （base@0, _pad0@12, parent@16, flags@20, _pad1@24, _pad2@28）。 */
  struct GPUBoneConst {
    float base_pos[3];
    float _pad0;
    int parent;
    int flags;
    float _pad1;
    float _pad2;
  };
  struct GPUChainConst {
    int target;
    int effector;
    int link_offset;
    int link_count;
    int iterations;
    float runtime_angle;
  };
  struct GPULinkConst {
    int bone;
    int has_limit;
    float pad[2]; /* float4 aligns to 16 */
    float limit_min[4];
    float limit_max[4];
  };
  struct GPUFrameBone {
    float q_base[4];
    float m0_row0[4];
    float m0_row1[4];
    float m0_row2[4];
    float m0_row3[4];
  };
  struct GPUFrameOut {
    float q_current[4];
  };

  std::vector<GPUBoneConst> gpu_bones(bone_count);
  for (int i = 0; i < bone_count; i++) {
    gpu_bones[i].base_pos[0] = buffers.bones[i].base_pos[0];
    gpu_bones[i].base_pos[1] = buffers.bones[i].base_pos[1];
    gpu_bones[i].base_pos[2] = buffers.bones[i].base_pos[2];
    gpu_bones[i]._pad0 = 0.0f;
    gpu_bones[i].parent = buffers.bones[i].parent;
    gpu_bones[i].flags = buffers.bones[i].flags;
    gpu_bones[i]._pad1 = 0.0f;
    gpu_bones[i]._pad2 = 0.0f;
  }

  std::vector<GPUChainConst> gpu_chains(buffers.chains.size());
  for (size_t i = 0; i < buffers.chains.size(); i++) {
    gpu_chains[i].target = buffers.chains[i].target_bone;
    gpu_chains[i].effector = buffers.chains[i].effector_bone;
    gpu_chains[i].link_offset = buffers.chains[i].link_offset;
    gpu_chains[i].link_count = buffers.chains[i].link_count;
    gpu_chains[i].iterations = buffers.chains[i].iterations;
    gpu_chains[i].runtime_angle = buffers.chains[i].runtime_angle;
  }

  std::vector<GPULinkConst> gpu_links(buffers.links.size());
  for (size_t i = 0; i < buffers.links.size(); i++) {
    gpu_links[i].bone = buffers.links[i].bone;
    gpu_links[i].has_limit = buffers.links[i].has_limit;
    for (int k = 0; k < 3; k++) {
      gpu_links[i].limit_min[k] = buffers.links[i].limit_min[k];
      gpu_links[i].limit_max[k] = buffers.links[i].limit_max[k];
    }
    gpu_links[i].limit_min[3] = 0.0f;
    gpu_links[i].limit_max[3] = 0.0f;
  }

  const size_t frame_data_count = size_t(frame_count) * size_t(bone_count);
  std::vector<GPUFrameBone> gpu_frames(frame_data_count);
  std::vector<GPUFrameOut> gpu_out(frame_data_count);
  for (size_t i = 0; i < frame_data_count; i++) {
    const MmdCCDBakeBuffers::FrameBone &src = buffers.frames[i];
    GPUFrameBone &dst = gpu_frames[i];
    for (int k = 0; k < 4; k++) {
      dst.q_base[k] = src.q_base[k];
      dst.m0_row0[k] = src.m0[0][k];
      dst.m0_row1[k] = src.m0[1][k];
      dst.m0_row2[k] = src.m0[2][k];
      dst.m0_row3[k] = src.m0[3][k];
    }
    /* q_current 初始为 identity（与 CPU 参照路径一致）。零四元数会在
     * quat_normalize 的零保护下吞掉首轮 delta，导致与 CPU 结果发散。 */
    gpu_out[i].q_current[0] = 1.0f;
    gpu_out[i].q_current[1] = 0.0f;
    gpu_out[i].q_current[2] = 0.0f;
    gpu_out[i].q_current[3] = 0.0f;
  }

  gpu::StorageBuf *bone_buf = GPU_storagebuf_create_ex(
      sizeof(GPUBoneConst) * gpu_bones.size(), gpu_bones.data(), GPU_USAGE_STATIC, "mmd_bake_bones");
  gpu::StorageBuf *chain_buf = GPU_storagebuf_create_ex(sizeof(GPUChainConst) * gpu_chains.size(),
                                                      gpu_chains.data(),
                                                      GPU_USAGE_STATIC,
                                                      "mmd_bake_chains");
  gpu::StorageBuf *link_buf = GPU_storagebuf_create_ex(
      sizeof(GPULinkConst) * gpu_links.size(), gpu_links.data(), GPU_USAGE_STATIC, "mmd_bake_links");
  gpu::StorageBuf *frame_buf = GPU_storagebuf_create_ex(
      sizeof(GPUFrameBone) * gpu_frames.size(), gpu_frames.data(), GPU_USAGE_STATIC, "mmd_bake_frames");
  gpu::StorageBuf *out_buf = GPU_storagebuf_create_ex(
      sizeof(GPUFrameOut) * gpu_out.size(), gpu_out.data(), GPU_USAGE_STATIC, "mmd_bake_out");

  /* 调试缓冲：链数据 + iter0 追踪（仅 frame 0 写入）。 */
  const int dbg_chain_count = int(buffers.chains.size());
  const int dbg_float_count = dbg_chain_count * 6 + dbg_chain_count * 4 * 32;
  std::vector<float> dbg_buf_data(dbg_float_count, -777.0f);
  gpu::StorageBuf *dbg_buf = GPU_storagebuf_create_ex(
      sizeof(float) * dbg_float_count, dbg_buf_data.data(), GPU_USAGE_STATIC, "mmd_bake_dbg");

  if (bone_buf == nullptr || chain_buf == nullptr || link_buf == nullptr || frame_buf == nullptr ||
      out_buf == nullptr || dbg_buf == nullptr)
  {
    GPU_storagebuf_free(bone_buf);
    GPU_storagebuf_free(chain_buf);
    GPU_storagebuf_free(link_buf);
    GPU_storagebuf_free(frame_buf);
    GPU_storagebuf_free(out_buf);
    GPU_storagebuf_free(dbg_buf);
    GPU_shader_free(shader);
    return false;
  }

  const int bone_binding = GPU_shader_get_ssbo_binding(shader, "bone_const_buf");
  const int chain_binding = GPU_shader_get_ssbo_binding(shader, "chain_const_buf");
  const int link_binding = GPU_shader_get_ssbo_binding(shader, "link_const_buf");
  const int frame_binding = GPU_shader_get_ssbo_binding(shader, "frame_buf");
  const int out_binding = GPU_shader_get_ssbo_binding(shader, "frame_out_buf");
  const int dbg_binding = GPU_shader_get_ssbo_binding(shader, "debug_buf");
  std::fprintf(stderr,
               "[BAKEDBG] bindings bone=%d chain=%d link=%d frame=%d out=%d dbg=%d\n",
               bone_binding,
               chain_binding,
               link_binding,
               frame_binding,
               out_binding,
               dbg_binding);

  GPU_storagebuf_bind(bone_buf, bone_binding);
  GPU_storagebuf_bind(chain_buf, chain_binding);
  GPU_storagebuf_bind(link_buf, link_binding);
  GPU_storagebuf_bind(frame_buf, frame_binding);
  GPU_storagebuf_bind(out_buf, out_binding);
  GPU_storagebuf_bind(dbg_buf, dbg_binding);

  GPU_shader_bind(shader);
  GPU_shader_uniform_1i(shader, "bone_count", bone_count);
  GPU_shader_uniform_1i(shader, "chain_count", int(buffers.chains.size()));
  GPU_shader_uniform_1i(shader, "frame_count", frame_count);

  const int local_size = 64;
  const int group_count = (frame_count + local_size - 1) / local_size;
  GPU_compute_dispatch(shader, group_count, 1, 1);
  GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE);

  GPU_storagebuf_read(out_buf, gpu_out.data());

  /* 验证 frame_buf 上传数据完好性（bone 32/436 的 q_base 与 m0）。 */
  {
    std::vector<GPUFrameBone> fb_check(frame_data_count);
    GPU_storagebuf_read(frame_buf, fb_check.data());
    std::fprintf(stderr,
                 "[BAKEDBG] frame_buf[32]  q_base=(%g,%g,%g,%g) m0_00=(%g,%g,%g,%g)\n",
                 fb_check[32].q_base[0],
                 fb_check[32].q_base[1],
                 fb_check[32].q_base[2],
                 fb_check[32].q_base[3],
                 fb_check[32].m0_row0[0],
                 fb_check[32].m0_row0[1],
                 fb_check[32].m0_row0[2],
                 fb_check[32].m0_row0[3]);
    std::fprintf(stderr,
                 "[BAKEDBG] frame_buf[436] q_base=(%g,%g,%g,%g) m0_00=(%g,%g,%g,%g)\n",
                 fb_check[436].q_base[0],
                 fb_check[436].q_base[1],
                 fb_check[436].q_base[2],
                 fb_check[436].q_base[3],
                 fb_check[436].m0_row0[0],
                 fb_check[436].m0_row0[1],
                 fb_check[436].m0_row0[2],
                 fb_check[436].m0_row0[3]);
  }

  /* 读回调试数据并打印。 */
  GPU_storagebuf_read(dbg_buf, dbg_buf_data.data());
  std::fprintf(stderr, "[BAKEDBG] chain data on GPU:\n");
  for (int c = 0; c < dbg_chain_count; c++) {
    std::fprintf(stderr,
                 "[BAKEDBG]   c%d target %.0f effector %.0f offset %.0f count %.0f iter %.0f angle %.6f\n",
                 c,
                 dbg_buf_data[c * 6 + 0],
                 dbg_buf_data[c * 6 + 1],
                 dbg_buf_data[c * 6 + 2],
                 dbg_buf_data[c * 6 + 3],
                 dbg_buf_data[c * 6 + 4],
                 dbg_buf_data[c * 6 + 5]);
  }
  for (int c = 0; c < dbg_chain_count; c++) {
    for (int o = 0; o < 4; o++) {
      const float *d = dbg_buf_data.data() + dbg_chain_count * 6 + (c * 4 + o) * 32;
      if (d[18] < -100.0f) {
        continue; /* 未写入 */
      }
      std::fprintf(stderr,
                   "[BAKEDBG]   c%d o%d joint(%+.5f,%+.5f,%+.5f) eff(%+.5f,%+.5f,%+.5f) tgt(%+.5f,%+.5f,%+.5f) "
                   "axis_w(%+.5f,%+.5f,%+.5f) cl(%+.5f,%+.5f,%+.5f) axis_l(%+.5f,%+.5f,%+.5f) "
                   "cos %.6f half %.6f delta(%+.5f,%+.5f,%+.5f,%+.5f) q(%+.5f,%+.5f,%+.5f,%+.5f)\n",
                   c,
                   o,
                   d[0], d[1], d[2],
                   d[3], d[4], d[5],
                   d[6], d[7], d[8],
                   d[9], d[10], d[11],
                   d[12], d[13], d[14],
                   d[15], d[16], d[17],
                   d[18], d[19],
                   d[20], d[21], d[22], d[23],
                   d[24], d[25], d[26], d[27]);
    }
  }

  GPU_storagebuf_unbind(bone_buf);
  GPU_storagebuf_unbind(chain_buf);
  GPU_storagebuf_unbind(link_buf);
  GPU_storagebuf_unbind(frame_buf);
  GPU_storagebuf_unbind(out_buf);
  GPU_storagebuf_unbind(dbg_buf);
  GPU_storagebuf_free(bone_buf);
  GPU_storagebuf_free(chain_buf);
  GPU_storagebuf_free(link_buf);
  GPU_storagebuf_free(frame_buf);
  GPU_storagebuf_free(out_buf);
  GPU_storagebuf_free(dbg_buf);
  GPU_shader_unbind();
  GPU_shader_free(shader);

  if (temp_context) {
    /* 注意：后台模式下不在此禁用临时上下文。退出路径（WM_exit → gpu_is_init
     * 分支）会自行 DRW_gpu_context_enable_ex / disable_ex 并销毁上下文；
     * 这里提前 disable 再让退出路径 re-enable，在 -b 模式会挂死。 */
  }

  r_q_current.resize(frame_data_count * 4);
  for (size_t i = 0; i < frame_data_count; i++) {
    for (int k = 0; k < 4; k++) {
      r_q_current[i * 4 + k] = gpu_out[i].q_current[k];
    }
  }
  return true;
}

/* -------------------------------------------------------------------- */
/* CPU 参照路径                                                          */
/* -------------------------------------------------------------------- */

void mmd_ccd_bake_cpu_reference(MmdCCDBakeBuffers &buffers, std::vector<float> &r_q_current)
{
  const int bone_count = int(buffers.bones.size());
  const int frame_count = buffers.frame_count;
  r_q_current.assign(size_t(frame_count) * size_t(bone_count) * 4, 0.0f);

  /* 构造 v8 链/骨结构（每帧独立求解，与 eval 路径一致）。 */
  std::vector<CCDIKV8Bone> bones(bone_count);
  std::vector<CCDIKV8Chain> chains(buffers.chains.size());
  std::vector<CCDIKV8Link> links(buffers.links.size());

  for (int i = 0; i < bone_count; i++) {
    CCDIKV8Bone &b = bones[i];
    for (int k = 0; k < 3; k++) {
      b.base_pos_mmd[k] = buffers.bones[i].base_pos[k];
    }
    b.parent_index = buffers.bones[i].parent;
  }
  for (size_t i = 0; i < buffers.chains.size(); i++) {
    CCDIKV8Chain &c = chains[i];
    c.target_bone_index = buffers.chains[i].target_bone;
    c.effector_bone_index = buffers.chains[i].effector_bone;
    c.links = links.data() + buffers.chains[i].link_offset;
    c.link_count = buffers.chains[i].link_count;
    c.iterations = buffers.chains[i].iterations;
    c.runtime_angle = buffers.chains[i].runtime_angle;
  }
  for (size_t i = 0; i < buffers.links.size(); i++) {
    CCDIKV8Link &l = links[i];
    l.bone_index = buffers.links[i].bone;
    l.has_limit = buffers.links[i].has_limit != 0;
    for (int k = 0; k < 3; k++) {
      l.limit_min_mmd[k] = buffers.links[i].limit_min[k];
      l.limit_max_mmd[k] = buffers.links[i].limit_max[k];
    }
  }

  for (int f = 0; f < frame_count; f++) {
    for (int i = 0; i < bone_count; i++) {
      CCDIKV8Bone &b = bones[i];
      const MmdCCDBakeBuffers::FrameBone &src = buffers.frames[size_t(f) * bone_count + i];
      for (int k = 0; k < 4; k++) {
        b.q_base_mmd[k] = src.q_base[k];
        b.q_current_mmd[k] = (k == 0) ? 1.0f : 0.0f;
      }
      for (int r = 0; r < 4; r++) {
        for (int c = 0; c < 4; c++) {
          b.initial_m0_mmd[r][c] = src.m0[r][c];
        }
      }
    }
    mmd_ccd_v8_solve_all_chains(chains.data(), int(chains.size()), bones.data(), bone_count);
    for (int i = 0; i < bone_count; i++) {
      for (int k = 0; k < 4; k++) {
        r_q_current[(size_t(f) * bone_count + i) * 4 + k] = bones[i].q_current_mmd[k];
      }
    }
  }
}

}  // namespace blender::mmd
