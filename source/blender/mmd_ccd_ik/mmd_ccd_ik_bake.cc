/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "mmd_ccd_ik_bake.hh"

#include "mmd_ccd_ik_v8.hh"
#include "mmd_bake_gpu_types.hh"

#include "GPU_compute.hh"
#include "GPU_context.hh"
#include "GPU_state.hh"
#include "GPU_shader.hh"
#include "GPU_storage_buffer.hh"
#include "GPU_uniform_buffer.hh"

#include "WM_api.hh"
#include "DRW_engine.hh"

#include "BLI_time.hh"

#include <cstring>

#ifdef WITH_MMD_BAKE_CUDA
#  include <cuda_runtime.h>
extern "C" cudaError_t mmd_ccd_bake_launch(const blender::mmd::bake::BoneConst *bones,
                                           const blender::mmd::bake::ChainConst *chains,
                                           const blender::mmd::bake::LinkConst *links,
                                           blender::mmd::bake::FrameBone *frames,
                                           blender::mmd::bake::FrameOut *out,
                                           int bone_count,
                                           int chain_count,
                                           int frame_count,
                                           int link_count);
#endif

namespace blender::mmd {

/* 与 GLSL/CUDA 侧逐字节一致的打包结构（见 mmd_bake_gpu_types.hh）。 */
using GPUBoneConst = bake::BoneConst;
using GPUChainConst = bake::ChainConst;
using GPULinkConst = bake::LinkConst;
using GPUFrameBone = bake::FrameBone;
using GPUFrameOut = bake::FrameOut;

static const char *g_bake_last_backend = "none";

const char *mmd_ccd_bake_gpu_last_backend()
{
  return g_bake_last_backend;
}

/* -------------------------------------------------------------------- */
/* GPU 资源缓存（R8-GPU）                                                */
/* -------------------------------------------------------------------- */

/* 烘焙的常量数据（骨骼/链/link）与求解器无关，且着色器编译是首烤 ~1-2s 的
 * 大头。把它们连同已编译着色器一起按内容缓存，跨烘焙调用复用，显著降低
 * 自动烘焙（VMD 导入触发）与重复烘焙的开销。缓存以活动 GPU 上下文为界：
 * 上下文变化（如 offscreen 上下文重建）时旧资源随上下文销毁，指针直接
 * 失效重置（不再 free，避免 double-free）。 */
namespace {

struct MmdBakeGpuCache {
  GPUContext *context = nullptr;
  gpu::Shader *shader = nullptr;
  gpu::StorageBuf *bone_buf = nullptr;
  gpu::StorageBuf *chain_buf = nullptr;
  gpu::StorageBuf *link_buf = nullptr;
  std::vector<uint8_t> bone_bytes;
  std::vector<uint8_t> chain_bytes;
  std::vector<uint8_t> link_bytes;
};

static MmdBakeGpuCache g_bake_cache;

static bool cache_matches(const std::vector<uint8_t> &cached,
                          const void *data,
                          const size_t size)
{
  return cached.size() == size && (size == 0 || std::memcmp(cached.data(), data, size) == 0);
}

static void bake_cache_reset()
{
  g_bake_cache = MmdBakeGpuCache();
}

static std::vector<uint8_t> bytes_of(const void *data, const size_t size)
{
  std::vector<uint8_t> bytes(size);
  if (size > 0) {
    std::memcpy(bytes.data(), data, size);
  }
  return bytes;
}

}  // namespace

/* -------------------------------------------------------------------- */
/* 紧凑输出散射（仅链骨）回 frame×bone 布局                               */
/* -------------------------------------------------------------------- */

namespace {

static void mmd_bake_scatter_out(MmdCCDBakeBuffers &buffers,
                                 const std::vector<GPUFrameOut> &gpu_out,
                                 const int bone_count,
                                 const int frame_count,
                                 const size_t link_count,
                                 std::vector<float> &r_q_current)
{
  const size_t frame_data_count = size_t(frame_count) * size_t(bone_count);
  r_q_current.assign(frame_data_count * 4, 0.0f);
  for (size_t i = 0; i < frame_data_count; i++) {
    r_q_current[i * 4 + 0] = 1.0f;
  }
  for (int f = 0; f < frame_count; f++) {
    for (size_t li = 0; li < link_count; li++) {
      const int bone = buffers.links[li].bone;
      if (bone < 0 || bone >= bone_count) {
        continue;
      }
      const GPUFrameOut &qo = gpu_out[size_t(f) * link_count + li];
      const size_t dst = (size_t(f) * bone_count + bone) * 4;
      for (int k = 0; k < 4; k++) {
        r_q_current[dst + k] = qo.q_current[k];
      }
    }
  }
}

}  // namespace

/* -------------------------------------------------------------------- */
/* CUDA 后端（R9-CUDA，NVIDIA 快路径；不可用回退 Vulkan/CPU）            */
/* -------------------------------------------------------------------- */

#ifdef WITH_MMD_BAKE_CUDA

namespace {

#define MMD_CUDA_CHECK(expr) \
  do { \
    cudaError_t _mmd_cuda_r = (expr); \
    if (_mmd_cuda_r != cudaSuccess) { \
      std::fprintf(stderr, \
                   "[BAKETIME] CUDA error %s (%d) at %s:%d\n", \
                   cudaGetErrorString(_mmd_cuda_r), \
                   int(_mmd_cuda_r), \
                   __FILE__, \
                   __LINE__); \
      return false; \
    } \
  } while (0)

struct MmdBakeCudaCache {
  bool inited = false;
  bool available = false;
  void *bone_ptr = nullptr;
  void *chain_ptr = nullptr;
  void *link_ptr = nullptr;
  std::vector<uint8_t> bone_bytes;
  std::vector<uint8_t> chain_bytes;
  std::vector<uint8_t> link_bytes;
};

static MmdBakeCudaCache g_cuda_cache;

static bool mmd_bake_cuda_init()
{
  if (g_cuda_cache.inited) {
    return g_cuda_cache.available;
  }
  g_cuda_cache.inited = true;

  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count <= 0) {
    std::fprintf(stderr, "[BAKETIME] CUDA backend unavailable (no CUDA device)\n");
    return false;
  }
  cudaDeviceProp props = {};
  cudaGetDeviceProperties(&props, 0);
  /* 预编译仅 sm_86/sm_89（SASS）；其它架构回退 Vulkan。 */
  const int arch = props.major * 10 + props.minor;
  if (arch != 86 && arch != 89) {
    std::fprintf(stderr,
                 "[BAKETIME] CUDA backend: kernel not built for sm_%d%d, falling back to "
                 "Vulkan\n",
                 props.major,
                 props.minor);
    return false;
  }
  g_cuda_cache.available = true;
  std::fprintf(stderr, "[BAKETIME] CUDA backend ready: %s (sm_%d%d)\n",
               props.name,
               props.major,
               props.minor);
  return true;
}

}  // namespace

/* CUDA 求解路径：成功返回 true 并写入 r_q_current；任何 CUDA 调用失败返回
 * false（调用方回退 Vulkan）。 */
static bool mmd_ccd_bake_gpu_cuda(MmdCCDBakeBuffers &buffers,
                                  const std::vector<GPUBoneConst> &gpu_bones,
                                  const std::vector<GPUChainConst> &gpu_chains,
                                  const std::vector<GPULinkConst> &gpu_links,
                                  const std::vector<GPUFrameBone> &gpu_frames,
                                  std::vector<GPUFrameOut> &gpu_out,
                                  const int bone_count,
                                  const int frame_count,
                                  const size_t link_count,
                                  std::vector<float> &r_q_current)
{
  const double t_cuda0 = BLI_time_now_seconds();

  /* 常量缓冲按内容缓存。 */
  const bool const_ok =
      cache_matches(g_cuda_cache.bone_bytes, gpu_bones.data(),
                    sizeof(GPUBoneConst) * gpu_bones.size()) &&
      cache_matches(g_cuda_cache.chain_bytes, gpu_chains.data(),
                    sizeof(GPUChainConst) * gpu_chains.size()) &&
      cache_matches(g_cuda_cache.link_bytes, gpu_links.data(),
                    sizeof(GPULinkConst) * gpu_links.size());
  if (g_cuda_cache.bone_ptr != nullptr && !const_ok) {
    cudaFree(g_cuda_cache.bone_ptr);
    cudaFree(g_cuda_cache.chain_ptr);
    cudaFree(g_cuda_cache.link_ptr);
    g_cuda_cache.bone_ptr = g_cuda_cache.chain_ptr = g_cuda_cache.link_ptr = nullptr;
  }
  if (g_cuda_cache.bone_ptr == nullptr) {
    MMD_CUDA_CHECK(cudaMalloc(&g_cuda_cache.bone_ptr, sizeof(GPUBoneConst) * gpu_bones.size()));
    MMD_CUDA_CHECK(cudaMemcpy(g_cuda_cache.bone_ptr,
                              gpu_bones.data(),
                              sizeof(GPUBoneConst) * gpu_bones.size(),
                              cudaMemcpyHostToDevice));
    MMD_CUDA_CHECK(cudaMalloc(&g_cuda_cache.chain_ptr, sizeof(GPUChainConst) * gpu_chains.size()));
    MMD_CUDA_CHECK(cudaMemcpy(g_cuda_cache.chain_ptr,
                              gpu_chains.data(),
                              sizeof(GPUChainConst) * gpu_chains.size(),
                              cudaMemcpyHostToDevice));
    MMD_CUDA_CHECK(cudaMalloc(&g_cuda_cache.link_ptr, sizeof(GPULinkConst) * gpu_links.size()));
    MMD_CUDA_CHECK(cudaMemcpy(g_cuda_cache.link_ptr,
                              gpu_links.data(),
                              sizeof(GPULinkConst) * gpu_links.size(),
                              cudaMemcpyHostToDevice));
    g_cuda_cache.bone_bytes = bytes_of(gpu_bones.data(),
                                       sizeof(GPUBoneConst) * gpu_bones.size());
    g_cuda_cache.chain_bytes = bytes_of(gpu_chains.data(),
                                        sizeof(GPUChainConst) * gpu_chains.size());
    g_cuda_cache.link_bytes = bytes_of(gpu_links.data(),
                                       sizeof(GPULinkConst) * gpu_links.size());
  }

  void *frame_ptr = nullptr;
  void *out_ptr = nullptr;
  MMD_CUDA_CHECK(cudaMalloc(&frame_ptr, sizeof(GPUFrameBone) * gpu_frames.size()));
  MMD_CUDA_CHECK(cudaMemcpy(
      frame_ptr, gpu_frames.data(), sizeof(GPUFrameBone) * gpu_frames.size(),
      cudaMemcpyHostToDevice));
  MMD_CUDA_CHECK(cudaMalloc(&out_ptr, sizeof(GPUFrameOut) * gpu_out.size()));
  MMD_CUDA_CHECK(cudaMemcpy(
      out_ptr, gpu_out.data(), sizeof(GPUFrameOut) * gpu_out.size(), cudaMemcpyHostToDevice));
  const double upload_ms = (BLI_time_now_seconds() - t_cuda0) * 1000.0;

  const double t_launch0 = BLI_time_now_seconds();
  MMD_CUDA_CHECK(mmd_ccd_bake_launch(static_cast<const bake::BoneConst *>(g_cuda_cache.bone_ptr),
                                     static_cast<const bake::ChainConst *>(g_cuda_cache.chain_ptr),
                                     static_cast<const bake::LinkConst *>(g_cuda_cache.link_ptr),
                                     static_cast<bake::FrameBone *>(frame_ptr),
                                     static_cast<bake::FrameOut *>(out_ptr),
                                     bone_count,
                                     int(buffers.chains.size()),
                                     frame_count,
                                     int(link_count)));
  MMD_CUDA_CHECK(cudaDeviceSynchronize());
  const double launch_ms = (BLI_time_now_seconds() - t_launch0) * 1000.0;

  const double t_readback0 = BLI_time_now_seconds();
  MMD_CUDA_CHECK(cudaMemcpy(
      gpu_out.data(), out_ptr, sizeof(GPUFrameOut) * gpu_out.size(), cudaMemcpyDeviceToHost));
  const double readback_ms = (BLI_time_now_seconds() - t_readback0) * 1000.0;
  cudaFree(frame_ptr);
  cudaFree(out_ptr);

  std::fprintf(stderr,
               "[BAKETIME] backend CUDA: upload %.1f ms, launch (%.2f ms), readback %.1f ms\n",
               upload_ms,
               launch_ms,
               readback_ms);

  mmd_bake_scatter_out(buffers, gpu_out, bone_count, frame_count, link_count, r_q_current);
  return true;
}

#endif /* WITH_MMD_BAKE_CUDA */

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
  g_bake_last_backend = "none";

  /* 打包 GPU 结构（与 mmd_bake_gpu_types.hh / GLSL std430 布局一致）。 */
  /* 骨骼 → 链骨输出槽位（仅链骨有效）。 */
  std::vector<int> bone_out_slot(bone_count, -1);
  for (size_t li = 0; li < buffers.links.size(); li++) {
    const int bone = buffers.links[li].bone;
    if (bone >= 0 && bone < bone_count) {
      bone_out_slot[bone] = int(li);
    }
  }

  std::vector<GPUBoneConst> gpu_bones(bone_count);
  for (int i = 0; i < bone_count; i++) {
    gpu_bones[i].base_pos[0] = buffers.bones[i].base_pos[0];
    gpu_bones[i].base_pos[1] = buffers.bones[i].base_pos[1];
    gpu_bones[i].base_pos[2] = buffers.bones[i].base_pos[2];
    gpu_bones[i].out_index = bone_out_slot[i];
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
    gpu_links[i].out_index = int(i);
    gpu_links[i]._pad1 = 0.0f;
    for (int k = 0; k < 3; k++) {
      gpu_links[i].limit_min[k] = buffers.links[i].limit_min[k];
      gpu_links[i].limit_max[k] = buffers.links[i].limit_max[k];
    }
    gpu_links[i].limit_min[3] = 0.0f;
    gpu_links[i].limit_max[3] = 0.0f;
  }

  const size_t frame_data_count = size_t(frame_count) * size_t(bone_count);
  const size_t link_count = buffers.links.size();
  const size_t out_data_count = size_t(frame_count) * link_count;
  std::vector<GPUFrameBone> gpu_frames(frame_data_count);
  std::vector<GPUFrameOut> gpu_out(out_data_count);
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
  }
  /* q_current 初始为 identity（与 CPU 参照路径一致）。零四元数会在
   * quat_normalize 的零保护下吞掉首轮 delta，导致与 CPU 结果发散。 */
  for (size_t i = 0; i < out_data_count; i++) {
    gpu_out[i].q_current[0] = 1.0f;
    gpu_out[i].q_current[1] = 0.0f;
    gpu_out[i].q_current[2] = 0.0f;
    gpu_out[i].q_current[3] = 0.0f;
  }

#ifdef WITH_MMD_BAKE_CUDA
  /* R9-CUDA：NVIDIA 机器优先走 CUDA 快路径（无需 Blender GPU 上下文）。 */
  if (mmd_bake_cuda_init()) {
    if (mmd_ccd_bake_gpu_cuda(buffers,
                              gpu_bones,
                              gpu_chains,
                              gpu_links,
                              gpu_frames,
                              gpu_out,
                              bone_count,
                              frame_count,
                              link_count,
                              r_q_current))
    {
      g_bake_last_backend = "CUDA";
      return true;
    }
    /* CUDA 调用失败 → 回退 Vulkan 路径。 */
  }
#endif /* WITH_MMD_BAKE_CUDA */

  /* 无头/后台线程没有活动上下文时，通过 WM 的 offscreen GPU 初始化创建。 */
  bool temp_context = false;
  if (GPU_context_active_get() == nullptr) {
    WM_init_gpu_offscreen();
    temp_context = true;
    if (GPU_context_active_get() == nullptr) {
      DRW_gpu_context_enable();
    }
    if (GPU_context_active_get() == nullptr) {
      std::fprintf(stderr, "[BAKETIME] GPU context still null after enable\n");
      return false;
    }
  }

  GPUContext *active_ctx = GPU_context_active_get();
  if (g_bake_cache.context != active_ctx) {
    /* 上下文变化：旧 GPU 资源已随旧上下文销毁，缓存直接失效（不再 free）。 */
    bake_cache_reset();
  }

  const double t0 = BLI_time_now_seconds();
  gpu::Shader *shader = g_bake_cache.shader;
  if (shader == nullptr) {
    shader = GPU_shader_create_from_info_name("gpu_shader_mmd_ccd_bake");
    if (shader == nullptr) {
      std::fprintf(stderr, "[BAKETIME] shader create failed\n");
      return false;
    }
    g_bake_cache.shader = shader;
    g_bake_cache.context = active_ctx;
  }
  const double shader_ms = (BLI_time_now_seconds() - t0) * 1000.0;
  std::fprintf(stderr, "[BAKETIME] backend Vulkan: shader ready (%.1f ms, %s)\n",
               shader_ms,
               shader_ms < 1.0 ? "cached" : "compiled");


  /* 常量缓冲（骨骼/链/link）按内容缓存：模型不变时零上传。 */
  gpu::StorageBuf *bone_buf = g_bake_cache.bone_buf;
  gpu::StorageBuf *chain_buf = g_bake_cache.chain_buf;
  gpu::StorageBuf *link_buf = g_bake_cache.link_buf;
  const bool const_ok =
      cache_matches(g_bake_cache.bone_bytes, gpu_bones.data(),
                    sizeof(GPUBoneConst) * gpu_bones.size()) &&
      cache_matches(g_bake_cache.chain_bytes, gpu_chains.data(),
                    sizeof(GPUChainConst) * gpu_chains.size()) &&
      cache_matches(g_bake_cache.link_bytes, gpu_links.data(),
                    sizeof(GPULinkConst) * gpu_links.size());
  if (bone_buf != nullptr && !const_ok) {
    GPU_storagebuf_free(bone_buf);
    GPU_storagebuf_free(chain_buf);
    GPU_storagebuf_free(link_buf);
    bone_buf = chain_buf = link_buf = nullptr;
  }
  const double t_upload0 = BLI_time_now_seconds();
  if (bone_buf == nullptr) {
    bone_buf = GPU_storagebuf_create_ex(sizeof(GPUBoneConst) * gpu_bones.size(),
                                        gpu_bones.data(),
                                        GPU_USAGE_STATIC,
                                        "mmd_bake_bones");
    chain_buf = GPU_storagebuf_create_ex(sizeof(GPUChainConst) * gpu_chains.size(),
                                         gpu_chains.data(),
                                         GPU_USAGE_STATIC,
                                         "mmd_bake_chains");
    link_buf = GPU_storagebuf_create_ex(sizeof(GPULinkConst) * gpu_links.size(),
                                        gpu_links.data(),
                                        GPU_USAGE_STATIC,
                                        "mmd_bake_links");
    g_bake_cache.bone_buf = bone_buf;
    g_bake_cache.chain_buf = chain_buf;
    g_bake_cache.link_buf = link_buf;
    g_bake_cache.bone_bytes = bytes_of(gpu_bones.data(),
                                       sizeof(GPUBoneConst) * gpu_bones.size());
    g_bake_cache.chain_bytes = bytes_of(gpu_chains.data(),
                                        sizeof(GPUChainConst) * gpu_chains.size());
    g_bake_cache.link_bytes = bytes_of(gpu_links.data(),
                                       sizeof(GPULinkConst) * gpu_links.size());
  }
  const double const_upload_ms = (BLI_time_now_seconds() - t_upload0) * 1000.0;
  std::fprintf(stderr, "[BAKETIME] const buffers (%.1f ms, %s)\n",
               const_upload_ms,
               const_upload_ms < 1.0 ? "cached" : "uploaded");

  const double t_frame0 = BLI_time_now_seconds();
  gpu::StorageBuf *frame_buf = GPU_storagebuf_create_ex(
      sizeof(GPUFrameBone) * gpu_frames.size(), gpu_frames.data(), GPU_USAGE_STATIC, "mmd_bake_frames");
  gpu::StorageBuf *out_buf = GPU_storagebuf_create_ex(
      sizeof(GPUFrameOut) * gpu_out.size(), gpu_out.data(), GPU_USAGE_STATIC, "mmd_bake_out");
  const double frame_upload_ms = (BLI_time_now_seconds() - t_frame0) * 1000.0;
  std::fprintf(stderr,
               "[BAKETIME] frame upload %zu bones x %d frames (%.1f ms)\n",
               gpu_bones.size(),
               frame_count,
               frame_upload_ms);

  if (bone_buf == nullptr || chain_buf == nullptr || link_buf == nullptr || frame_buf == nullptr ||
      out_buf == nullptr)
  {
    /* 常量缓冲创建失败时同步失效缓存，避免悬垂指针。 */
    g_bake_cache.bone_buf = bone_buf;
    g_bake_cache.chain_buf = chain_buf;
    g_bake_cache.link_buf = link_buf;
    GPU_storagebuf_free(bone_buf);
    GPU_storagebuf_free(chain_buf);
    GPU_storagebuf_free(link_buf);
    GPU_storagebuf_free(frame_buf);
    GPU_storagebuf_free(out_buf);
    return false;
  }

  const int bone_binding = GPU_shader_get_ssbo_binding(shader, "bone_const_buf");
  const int chain_binding = GPU_shader_get_ssbo_binding(shader, "chain_const_buf");
  const int link_binding = GPU_shader_get_ssbo_binding(shader, "link_const_buf");
  const int frame_binding = GPU_shader_get_ssbo_binding(shader, "frame_buf");
  const int out_binding = GPU_shader_get_ssbo_binding(shader, "frame_out_buf");

  GPU_storagebuf_bind(bone_buf, bone_binding);
  GPU_storagebuf_bind(chain_buf, chain_binding);
  GPU_storagebuf_bind(link_buf, link_binding);
  GPU_storagebuf_bind(frame_buf, frame_binding);
  GPU_storagebuf_bind(out_buf, out_binding);

  GPU_shader_bind(shader);
  GPU_shader_uniform_1i(shader, "bone_count", bone_count);
  GPU_shader_uniform_1i(shader, "chain_count", int(buffers.chains.size()));
  GPU_shader_uniform_1i(shader, "frame_count", frame_count);
  GPU_shader_uniform_1i(shader, "link_count", int(link_count));

  const int local_size = 64;
  const int group_count = (frame_count + local_size - 1) / local_size;
  const double t_dispatch0 = BLI_time_now_seconds();
  GPU_compute_dispatch(shader, group_count, 1, 1);
  GPU_memory_barrier(GPU_BARRIER_SHADER_STORAGE);
  const double dispatch_ms = (BLI_time_now_seconds() - t_dispatch0) * 1000.0;

  const double t_readback0 = BLI_time_now_seconds();
  GPU_storagebuf_read(out_buf, gpu_out.data());
  const double readback_ms = (BLI_time_now_seconds() - t_readback0) * 1000.0;
  std::fprintf(stderr,
               "[BAKETIME] dispatch %d groups x %d invocations (%.2f ms) + readback (%.2f ms)\n",
               group_count,
               local_size,
               dispatch_ms,
               readback_ms);

  GPU_storagebuf_unbind(bone_buf);
  GPU_storagebuf_unbind(chain_buf);
  GPU_storagebuf_unbind(link_buf);
  GPU_storagebuf_unbind(frame_buf);
  GPU_storagebuf_unbind(out_buf);
  /* 常量缓冲与着色器保留在缓存中复用；仅释放每调用资源。 */
  GPU_storagebuf_free(frame_buf);
  GPU_storagebuf_free(out_buf);
  GPU_shader_unbind();

  if (temp_context) {
    /* 注意：后台模式下不在此禁用临时上下文。退出路径（WM_exit → gpu_is_init
     * 分支）会自行 DRW_gpu_context_enable_ex / disable_ex 并销毁上下文；
     * 这里提前 disable 再让退出路径 re-enable，在 -b 模式会挂死。 */
  }

  mmd_bake_scatter_out(buffers, gpu_out, bone_count, frame_count, link_count, r_q_current);
  g_bake_last_backend = "Vulkan";
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
