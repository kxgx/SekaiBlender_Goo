/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup editor_io
 *
 * GPU CCD IK 烘焙（R7-VMD）：WM_OT_mmd_bake_motion。
 * 采集 FK 姿态 → GPU（Vulkan compute，全厂商）批量求解 IK 链 →
 * 写出烘焙 FK Action。CPU v8 求解器同时运行作为精度参照。
 */

#include "io_vmd_ops.hh"

#include "ANIM_action.hh"
#include "ANIM_action_iterators.hh"
#include "ANIM_animdata.hh"
#include "ANIM_fcurve.hh"
#include "BKE_anim_data.hh"
#include "BKE_armature.hh"
#include "BKE_context.hh"
#include "BKE_fcurve.hh"
#include "BKE_idprop.hh"
#include "BKE_lib_id.hh"
#include "BKE_pose.hh"
#include "BKE_report.hh"
#include "BKE_scene.hh"
#include "BKE_wm_runtime.hh"

#include "DEG_depsgraph.hh"

#include "ED_object.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "BLI_math_matrix_c.hh"
#include "BLI_math_rotation_c.hh"
#include "BLI_math_vector_c.hh"
#include "BLI_fileops.hh"
#include "BLI_path_utils.hh"
#include "BLI_string.hh"
#include "BLI_task.hh"
#include "BLI_time.hh"
#include "BLI_vector.hh"

#include "MEM_guardedalloc.h"

#include "DNA_action_types.h"
#include "DNA_constraint_types.h"
#include "DNA_armature_types.h"
#include "DNA_curve_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "mmd_ccd_ik_bake.hh"
#include "importer/pmx_import_bone_ik.hh"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace blender {

static void mmd_bake_collect_topological(bArmature *arm,
                                         std::vector<Bone *> &r_pool,
                                         std::map<Bone *, int> &r_index)
{
  std::function<void(Bone *)> walk = [&](Bone *b) {
    if (b == nullptr || r_index.count(b) != 0) {
      return;
    }
    r_index[b] = int(r_pool.size());
    r_pool.push_back(b);
    for (Bone *child = static_cast<Bone *>(b->childbase.first); child != nullptr;
         child = child->next)
    {
      walk(child);
    }
  };
  for (Bone *root = static_cast<Bone *>(arm->bonebase.first); root != nullptr; root = root->next) {
    walk(root);
  }
}

wmOperatorStatus wm_mmd_bake_motion_exec(bContext *C, wmOperator *op)
{
  using blender::mmd::MmdCCDBakeBuffers;
  Main *bmain = CTX_data_main(C);
  Scene *scene = CTX_data_scene(C);
  Object *armature_obj = CTX_data_active_object(C);
  if (armature_obj == nullptr || armature_obj->type != OB_ARMATURE) {
    BKE_report(op->reports, RPT_ERROR, "MMD bake requires an active Armature object");
    return OPERATOR_CANCELLED;
  }
  bArmature *arm = id_cast<bArmature *>(armature_obj->data);
  if (arm == nullptr || armature_obj->pose == nullptr) {
    BKE_report(op->reports, RPT_ERROR, "MMD bake target has invalid Armature data");
    return OPERATOR_CANCELLED;
  }
  const float global_scale = RNA_float_get(op->ptr, "coordinate_scale");
  int frame_start = RNA_int_get(op->ptr, "frame_start");
  int frame_end = RNA_int_get(op->ptr, "frame_end");
  /* 未显式指定帧范围时使用场景帧范围（手动从菜单/F3 调用更顺手）。 */
  if (frame_start == 0 && frame_end == 0 && scene != nullptr) {
    frame_start = scene->r.sfra;
    frame_end = scene->r.efra;
  }
  /* 子帧采样：subframes 倍帧率采样（默认 2 = 每帧采 2 个半帧点），逐帧走
   * scene->r.subframe，让烘焙曲线跟上 VMD 贝塞尔插值的中间值，导出到
   * UE5 用线性插值时更平滑、更精确。 */
  const int subframes = std::max(1, RNA_int_get(op->ptr, "subframes"));
  /* GPU 全权接管（默认 GPU-only）：use_gpu 默认开启，GPU 失败直接报错不回退
   * CPU。SEKAI_FORCE_GPU=1 时即使显式关掉 use_gpu 也强制走 GPU。 */
  const bool force_gpu = std::getenv("SEKAI_FORCE_GPU") != nullptr;
  const bool use_gpu = force_gpu || RNA_boolean_get(op->ptr, "use_gpu");
  if (frame_end < frame_start || frame_start < 0) {
    BKE_report(op->reports, RPT_ERROR, "MMD bake frame range must be non-negative and ordered");
    return OPERATOR_CANCELLED;
  }

  /* 1) PMX IK 定义。 */
  io::pmx::PMXBoneIKDefinitionSet ik_def;
  io::pmx::read_bone_ik_definition(armature_obj->id, ik_def);
  if (ik_def.ik_bones.empty()) {
    BKE_report(op->reports,
               RPT_WARNING,
               "No PMX IK definition found; bake will copy FK poses only");
  }

  /* 2) 拓扑骨池。 */
  std::vector<Bone *> pool;
  std::map<Bone *, int> pool_index;
  mmd_bake_collect_topological(arm, pool, pool_index);
  const int bone_count = int(pool.size());
  if (bone_count == 0) {
    BKE_report(op->reports, RPT_ERROR, "MMD bake target Armature has no bones");
    return OPERATOR_CANCELLED;
  }

  /* 3) 链/link 常量。 */
  MmdCCDBakeBuffers buffers;
  buffers.bones.resize(bone_count);
  for (int i = 0; i < bone_count; i++) {
    Bone *b = pool[i];
    buffers.bones[i].base_pos[0] = b->arm_head[0] / global_scale;
    buffers.bones[i].base_pos[1] = b->arm_head[2] / global_scale;
    buffers.bones[i].base_pos[2] = b->arm_head[1] / global_scale;
    buffers.bones[i].parent = (b->parent != nullptr) ? pool_index[b->parent] : -1;
    buffers.bones[i].flags = 0;
  }
  for (const io::pmx::PMXBoneIKDefinition &def : ik_def.ik_bones) {
    Bone *ik_bone = BKE_armature_find_bone_name(arm, def.bone_name.c_str());
    Bone *target_bone = BKE_armature_find_bone_name(arm, def.target_name.c_str());
    if (ik_bone == nullptr || target_bone == nullptr) {
      continue;
    }
    const int t = pool_index[ik_bone];
    const int e = pool_index[target_bone];
    if (t < 0 || e < 0) {
      continue;
    }
    buffers.bones[t].flags |= 1;
    MmdCCDBakeBuffers::ChainInfo chain = {
        t, e, int(buffers.links.size()), 0, 39, def.angle_limit * 0.25f};
    for (const io::pmx::PMXBoneIKLink &link : def.links) {
      Bone *lb = BKE_armature_find_bone_name(arm, link.bone_name.c_str());
      const int li = (lb != nullptr) ? pool_index[lb] : -1;
      if (li < 0) {
        chain.link_count = -1;
        break;
      }
      MmdCCDBakeBuffers::LinkInfo info;
      info.bone = li;
      info.has_limit = link.limit_angle ? 1 : 0;
      for (int k = 0; k < 3; k++) {
        info.limit_min[k] = link.limit_min[k];
        info.limit_max[k] = link.limit_max[k];
      }
      buffers.links.push_back(info);
      buffers.bones[li].flags |= 2;
      chain.link_count++;
    }
    if (chain.link_count > 0) {
      buffers.chains.push_back(chain);
    }
    else {
      buffers.links.resize(chain.link_offset);
    }
  }
  for (const MmdCCDBakeBuffers::ChainInfo &chain : buffers.chains) {
    const int root_link = buffers.links[chain.link_offset + chain.link_count - 1].bone;
    const int root_parent = buffers.bones[root_link].parent;
    if (root_parent >= 0 && (buffers.bones[root_parent].flags & 3) == 0) {
      buffers.bones[root_parent].flags |= 4;
    }
  }

  /* 收集源 Action 中被动画化的 `pose.bones["..."].location` 通道集合。
   * 烘焙目前只写旋转曲线：VMD 位移轨道（关键是 足IK 等 IK 目标骨——烘焙后
   * 若重新启用 IK，目标停在绑定位置就是"腿找原点"）会被丢掉。下面逐帧
   * 采样并回写这些位移通道，让烘焙动作自洽且可导出。 */
  std::set<std::string> loc_rna_paths;
  {
    AnimData *src_adt = BKE_animdata_from_id(&armature_obj->id);
    if (src_adt != nullptr) {
      for (FCurve *fc : animrig::fcurves_for_assigned_action(src_adt)) {
        if (!fc->rna_path().is_empty() && fc->rna_path().endswith(".location")) {
          loc_rna_paths.insert(fc->rna_path().data());
        }
      }
    }
  }
  std::vector<bool> bones_have_loc(bone_count, false);
  for (int i = 0; i < bone_count; i++) {
    char escaped[128] = {};
    BLI_str_escape(escaped, pool[i]->name, sizeof(escaped));
    const std::string loc_path = std::string("pose.bones[\"") + escaped + "\"].location";
    bones_have_loc[i] = loc_rna_paths.count(loc_path) > 0;
  }

  /* 4) 临时关闭 IK 约束（采集纯 FK）。 */
  /* 烘焙期间锁定界面：逐帧依赖图求值会让视口跟随中间采样帧重绘（重复烘焙
   * 时视觉上腿会"找原点"），锁住后烘焙完成再一次性刷新为最终结果。 */
  wmWindowManager *wm = CTX_wm_manager(C);
  wm->runtime->is_interface_locked = true;
  std::vector<std::pair<bConstraint *, float>> saved_enforce;
  for (bPoseChannel *pchan = static_cast<bPoseChannel *>(armature_obj->pose->chanbase.first);
       pchan != nullptr;
       pchan = pchan->next)
  {
    for (bConstraint *con = static_cast<bConstraint *>(pchan->constraints.first); con != nullptr;
         con = con->next)
    {
      if (con->type == CONSTRAINT_TYPE_KINEMATIC && con->enforce != 0.0f) {
        saved_enforce.emplace_back(con, con->enforce);
        con->enforce = 0.0f;
      }
    }
  }

  /* 5) 逐帧采集（含子帧采样）。 */
  Depsgraph *depsgraph = CTX_data_depsgraph_pointer(C);
  const int frame_count = (frame_end - frame_start) * subframes + 1;
  const float frame_step = 1.0f / float(subframes);
  buffers.frame_count = frame_count;
  buffers.frames.resize(size_t(frame_count) * size_t(bone_count));
  /* 姿势通道索引预解析：避免每帧每骨 BKE_pose_channel_find_name 字符串查找。 */
  std::vector<bPoseChannel *> pchans(bone_count, nullptr);
  for (int i = 0; i < bone_count; i++) {
    pchans[i] = BKE_pose_channel_find_name(armature_obj->pose, pool[i]->name);
  }
  const double t_collect0 = BLI_time_now_seconds();
  /* 逐帧位移采样（仅源 Action 有位移曲线的骨骼；IK 求解只用旋转，位移不进 GPU）。 */
  std::vector<float> sampled_loc(size_t(frame_count) * size_t(bone_count) * 3, 0.0f);
  for (int s = 0; s < frame_count; s++) {
    const float frame = float(frame_start) + float(s) * frame_step;
    scene->r.cfra = int(frame);
    scene->r.subframe = frame - float(scene->r.cfra);
    BKE_scene_graph_update_for_newframe(depsgraph);
    const size_t frame_base = size_t(s) * bone_count;
    for (int i = 0; i < bone_count; i++) {
      Bone *b = pool[i];
      bPoseChannel *pchan = pchans[i];
      MmdCCDBakeBuffers::FrameBone &fb = buffers.frames[frame_base + i];
      if (pchan == nullptr) {
        unit_qt(fb.q_base);
        unit_m4(fb.m0);
        continue;
      }
      /* q_conv：arm_mat 3x3 YZ 列交换 + 转置。 */
      float arm_rot[3][3];
      copy_m3_m4(arm_rot, b->arm_mat);
      normalize_m3(arm_rot);
      for (int r = 0; r < 3; r++) {
        std::swap(arm_rot[r][1], arm_rot[r][2]);
      }
      float conv_mat[3][3];
      transpose_m3_m3(conv_mat, arm_rot);
      float q_conv[4];
      mat3_to_quat(q_conv, conv_mat);
      normalize_qt(q_conv);
      /* bl 四元数（FK 动画层，约束已关闭）。 */
      float bl_q[4];
      const float4 bl_q_value = BKE_pchan_rot_to_quat(*pchan);
      copy_qt_qt(bl_q, bl_q_value);
      normalize_qt(bl_q);
      /* q_base_mmd = conj(q_conv) * bl_q * q_conv */
      float cq[4], tq[4];
      conjugate_qt_qt(cq, q_conv);
      mul_qt_qtqt(tq, cq, bl_q);
      mul_qt_qtqt(fb.q_base, tq, q_conv);
      normalize_qt(fb.q_base);
      /* initial_m0（v8_build_initial_m0 等价）。 */
      float r_mmd[3][3];
      copy_m3_m4(r_mmd, pchan->pose_mat);
      normalize_m3(r_mmd);
      for (int r = 0; r < 3; r++) {
        std::swap(r_mmd[r][1], r_mmd[r][2]);
      }
      for (int r = 0; r < 3; r++) {
        std::swap(r_mmd[1][r], r_mmd[2][r]);
      }
      const float world_pos[3] = {
          pchan->pose_mat[3][0] / global_scale,
          pchan->pose_mat[3][2] / global_scale,
          pchan->pose_mat[3][1] / global_scale,
      };
      float base_times_r[3] = {0.0f, 0.0f, 0.0f};
      for (int c = 0; c < 3; c++) {
        base_times_r[c] = buffers.bones[i].base_pos[0] * r_mmd[0][c] +
                          buffers.bones[i].base_pos[1] * r_mmd[1][c] +
                          buffers.bones[i].base_pos[2] * r_mmd[2][c];
      }
      for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) {
          fb.m0[r][c] = r_mmd[r][c];
        }
        fb.m0[r][3] = 0.0f;
      }
      for (int c = 0; c < 3; c++) {
        fb.m0[3][c] = world_pos[c] - base_times_r[c];
      }
      fb.m0[3][3] = 1.0f;
      if (bones_have_loc[i]) {
        copy_v3_v3(&sampled_loc[(frame_base + i) * 3], pchan->loc);
      }
    }
  }
  scene->r.cfra = frame_end;
  scene->r.subframe = 0.0f;
  const double collect_ms = (BLI_time_now_seconds() - t_collect0) * 1000.0;
  std::fprintf(stderr,
               "[BAKETIME] collect %d samples x %d bones: %.0f ms (depsgraph per sample)\n",
               frame_count,
               bone_count,
               collect_ms);
  /* 诊断：采集首/末样本的左膝位置（排查"烘焙中腿找原点"）。 */
  if (BLI_getenv("MMD_BAKE_DIAG") != nullptr) {
    int knee_i = -1;
    for (int i = 0; i < bone_count; i++) {
      if (STREQ(pool[i]->name, "左ひざ")) {
        knee_i = i;
        break;
      }
    }
    if (knee_i >= 0) {
      const MmdCCDBakeBuffers::FrameBone &f0 = buffers.frames[knee_i];
      const MmdCCDBakeBuffers::FrameBone &fN =
          buffers.frames[size_t(frame_count - 1) * bone_count + knee_i];
      std::fprintf(stderr,
                   "[BAKETIME] diag knee m0[3]: first=(%.4f %.4f %.4f) last=(%.4f %.4f %.4f)\n",
                   f0.m0[3][0],
                   f0.m0[3][1],
                   f0.m0[3][2],
                   fN.m0[3][0],
                   fN.m0[3][1],
                   fN.m0[3][2]);
    }
  }

  /* 5b) 每帧每链 IK 门控（mmd_tools 对齐）：VMD 带 mmd_ik_toggle 轨道时按
   * 帧值决定该帧是否求解此链（0=纯 FK，直接写 q_base）；无轨道时 IK 常开
   * ——一律求解（与 mmd_tools 默认一致：IK 约束常驻，腿链逐帧解向 IK
   * 控制骨）。门控曲线来源：优先当前活动动作；烘焙完成后活动动作被替换
   * 成无 toggle 曲线的烘焙结果，此时回退到导入时记录的 VMD 源动作
   * （mmd_source_action ID 属性），保证重复烘焙的门控与首次一致。 */
  const size_t chain_count = buffers.chains.size();
  std::vector<uint8_t> chain_gate(size_t(frame_count) * chain_count, 0);
  {
    const auto curves_have_toggle = [](Span<FCurve *> curves) {
      return std::any_of(curves.begin(), curves.end(), [](const FCurve *fc) {
        return !fc->rna_path().is_empty() && fc->rna_path().endswith(".mmd_ik_toggle");
      });
    };

    Vector<FCurve *> gate_curves;
    bAction *assigned_action = nullptr;
    if (AnimData *src_adt = BKE_animdata_from_id(&armature_obj->id)) {
      if (src_adt->action != nullptr) {
        assigned_action = src_adt->action;
        gate_curves = Vector<FCurve *>(animrig::fcurves_for_assigned_action(src_adt));
      }
    }
    if (!curves_have_toggle(gate_curves)) {
      /* 回退到 VMD 源动作（导入时记录在 ID 属性里）。 */
      bAction *source = nullptr;
      if (IDProperty *props = IDP_GetProperties(&armature_obj->id)) {
        if (const std::optional<StringRefNull> name = IDP_group_lookup_string(
                *props, "mmd_source_action"))
        {
          source = reinterpret_cast<bAction *>(
              BKE_libblock_find_name(bmain, ID_AC, name->c_str()));
        }
      }
      if (source != nullptr && source != assigned_action) {
        Vector<FCurve *> source_curves;
        animrig::foreach_fcurve_in_action(source->wrap(), [&](FCurve &fcurve) {
          source_curves.append(&fcurve);
        });
        if (curves_have_toggle(source_curves)) {
          gate_curves = std::move(source_curves);
        }
      }
    }

    if (!gate_curves.is_empty()) {
      for (size_t ci = 0; ci < chain_count; ci++) {
        const MmdCCDBakeBuffers::ChainInfo &chain = buffers.chains[ci];
        const Bone *ik_bone = (chain.target_bone >= 0 && chain.target_bone < bone_count) ?
                                  pool[chain.target_bone] :
                                  nullptr;
        /* 该 IK 骨在源动作里的 mmd_ik_toggle 曲线。 */
        FCurve *toggle_fc = nullptr;
        if (ik_bone != nullptr) {
          char escaped[128] = {};
          BLI_str_escape(escaped, ik_bone->name, sizeof(escaped));
          const std::string toggle_path =
              std::string("pose.bones[\"") + escaped + "\"].mmd_ik_toggle";
          for (FCurve *fc : gate_curves) {
            if (STREQ(fc->rna_path().c_str(), toggle_path.c_str()) && fc->array_index == 0) {
              toggle_fc = fc;
              break;
            }
          }
        }
        if (toggle_fc != nullptr) {
          for (int f = 0; f < frame_count; f++) {
            chain_gate[size_t(f) * chain_count + ci] =
                evaluate_fcurve(toggle_fc, float(frame_start) + float(f) * frame_step) > 0.5f ?
                    1 :
                    0;
          }
        }
        else {
          /* 无开关轨道：IK 常开（mmd_tools 默认），全程求解。 */
          for (int f = 0; f < frame_count; f++) {
            chain_gate[size_t(f) * chain_count + ci] = 1;
          }
        }
      }
    }
    else {
      /* 无源动作信息时保守起见全程求解（保持 mmd_tools IK 常开语义）。 */
      for (size_t ci = 0; ci < chain_count; ci++) {
        for (int f = 0; f < frame_count; f++) {
          chain_gate[size_t(f) * chain_count + ci] = 1;
        }
      }
    }
  }

  /* 骨 → 所在链（按帧门控判断是否采用求解结果）。 */
  std::vector<std::vector<int>> bone_chains(bone_count);
  for (size_t ci = 0; ci < chain_count; ci++) {
    const MmdCCDBakeBuffers::ChainInfo &chain = buffers.chains[ci];
    for (int li = 0; li < chain.link_count; li++) {
      const int bi = buffers.links[chain.link_offset + li].bone;
      if (bi >= 0 && bi < bone_count) {
        bone_chains[bi].push_back(int(ci));
      }
    }
  }

  /* 6) 恢复约束。 */
  for (auto &[con, value] : saved_enforce) {
    con->enforce = value;
  }

  /* 7) 求解：GPU 全权接管——正常使用零 CPU 计算。GPU（CUDA→Vulkan）失败
   * 直接报错终止，绝不回退 CPU；CPU 参照求解只在开发调参时运行
   * （MMD_BAKE_CPU_REF=1），或请求 MMD_BAKE_DEBUG_DUMP 需要对照数据时
   * 例外运行一次。显式把算子 use_gpu 关掉（开发用）才走 CPU 求解。 */
  std::vector<float> q_gpu;
  std::vector<float> q_cpu;
  const double t0 = BLI_time_now_seconds();
  const bool gpu_ok = use_gpu && blender::mmd::mmd_ccd_bake_gpu(buffers, q_gpu);
  const double gpu_ms = (BLI_time_now_seconds() - t0) * 1000.0;
  /* GPU-only：GPU 失败即终止（不写动作、不回退 CPU）。 */
  if (use_gpu && !gpu_ok) {
    wm->runtime->is_interface_locked = false;
    BKE_report(op->reports,
               RPT_ERROR,
               "GPU bake failed (no CUDA/Vulkan device available); "
               "the bake is GPU-only and will not fall back to CPU");
    return OPERATOR_CANCELLED;
  }
  const bool debug_dump_requested = BLI_getenv("MMD_BAKE_DEBUG_DUMP") != nullptr;
  const bool run_cpu_ref =
      !use_gpu || BLI_getenv("MMD_BAKE_CPU_REF") != nullptr || debug_dump_requested;
  double cpu_ms = 0.0;
  if (run_cpu_ref) {
    const double t1 = BLI_time_now_seconds();
    blender::mmd::mmd_ccd_bake_cpu_reference(buffers, q_cpu);
    cpu_ms = (BLI_time_now_seconds() - t1) * 1000.0;
  }

  const std::vector<float> &q_out = gpu_ok ? q_gpu : q_cpu;

  if (gpu_ok) {
    if (run_cpu_ref) {
      /* q 与 -q 表示同一旋转：逐四元数取 |q1-q2| 与 |q1+q2| 的较小者。 */
      double max_err = 0.0;
      double sum_err = 0.0;
      const size_t quat_count = q_cpu.size() / 4;
      for (size_t qi = 0; qi < quat_count; qi++) {
        double d_neg = 0.0;
        double d_pos = 0.0;
        for (int k = 0; k < 4; k++) {
          const double diff = double(q_cpu[qi * 4 + k]) - double(q_gpu[qi * 4 + k]);
          const double sum = double(q_cpu[qi * 4 + k]) + double(q_gpu[qi * 4 + k]);
          d_neg += diff * diff;
          d_pos += sum * sum;
        }
        const double err = std::sqrt(std::min(d_neg, d_pos));
        max_err = std::max(max_err, err);
        sum_err += err;
      }
      BKE_reportf(op->reports,
                  RPT_INFO,
                  "GPU bake (%s): %d frames x %d bones (%dx sub-frame sampling) in %.1f ms "
                  "(CPU ref %.1f ms, 仅精度对照); "
                  "quat error (sign-aware) max %.6f mean %.6f",
                  blender::mmd::mmd_ccd_bake_gpu_last_backend(),
                  frame_count,
                  bone_count,
                  subframes,
                  gpu_ms,
                  cpu_ms,
                  max_err,
                  sum_err / double(std::max<size_t>(quat_count, 1)));
    }
    else {
      BKE_reportf(op->reports,
                  RPT_INFO,
                  "GPU bake (%s): %d frames x %d bones (%dx sub-frame sampling) in %.1f ms "
                  "(GPU-only; set MMD_BAKE_CPU_REF=1 for the dev CPU precision reference)",
                  blender::mmd::mmd_ccd_bake_gpu_last_backend(),
                  frame_count,
                  bone_count,
                  subframes,
                  gpu_ms);
    }

    /* 调试：MMD_BAKE_DEBUG_DUMP=路径 时写出全部 CPU/GPU 四元数对比。 */
    if (const char *dump_path = BLI_getenv("MMD_BAKE_DEBUG_DUMP")) {
      FILE *f = BLI_fopen(dump_path, "wb");
      if (f != nullptr) {
        std::fprintf(f, "bone_count %d\nframe_count %d\n", bone_count, frame_count);
        for (int i = 0; i < bone_count; i++) {
          std::fprintf(f,
                       "bone %d parent %d flags %d base %g %g %g\n",
                       i,
                       buffers.bones[i].parent,
                       buffers.bones[i].flags,
                       buffers.bones[i].base_pos[0],
                       buffers.bones[i].base_pos[1],
                       buffers.bones[i].base_pos[2]);
        }
        std::fprintf(f, "chain_count %d\n", int(buffers.chains.size()));
        for (size_t ci = 0; ci < buffers.chains.size(); ci++) {
          const MmdCCDBakeBuffers::ChainInfo &ch = buffers.chains[ci];
          std::fprintf(f,
                       "chain %d target %d effector %d offset %d count %d iter %d angle %g\n",
                       int(ci),
                       ch.target_bone,
                       ch.effector_bone,
                       ch.link_offset,
                       ch.link_count,
                       ch.iterations,
                       ch.runtime_angle);
        }
        std::fprintf(f, "link_count %d\n", int(buffers.links.size()));
        for (size_t li = 0; li < buffers.links.size(); li++) {
          const MmdCCDBakeBuffers::LinkInfo &lk = buffers.links[li];
          std::fprintf(f,
                       "link %d bone %d has_limit %d min %g %g %g max %g %g %g\n",
                       int(li),
                       lk.bone,
                       lk.has_limit,
                       lk.limit_min[0],
                       lk.limit_min[1],
                       lk.limit_min[2],
                       lk.limit_max[0],
                       lk.limit_max[1],
                       lk.limit_max[2]);
        }
        /* frame 0 输入（q_base + initial m0）。 */
        for (int i = 0; i < bone_count; i++) {
          const MmdCCDBakeBuffers::FrameBone &fb = buffers.frames[i];
          std::fprintf(f,
                       "in0 %d qb %g %g %g %g m0 %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g %g\n",
                       i,
                       fb.q_base[0],
                       fb.q_base[1],
                       fb.q_base[2],
                       fb.q_base[3],
                       fb.m0[0][0], fb.m0[0][1], fb.m0[0][2], fb.m0[0][3],
                       fb.m0[1][0], fb.m0[1][1], fb.m0[1][2], fb.m0[1][3],
                       fb.m0[2][0], fb.m0[2][1], fb.m0[2][2], fb.m0[2][3],
                       fb.m0[3][0], fb.m0[3][1], fb.m0[3][2], fb.m0[3][3]);
        }
        for (size_t qi = 0; qi < q_cpu.size() / 4; qi++) {
          std::fprintf(f,
                       "q %zu %d %.9g %.9g %.9g %.9g %.9g %.9g %.9g %.9g\n",
                       qi,
                       int(qi) % bone_count,
                       q_cpu[qi * 4 + 0],
                       q_cpu[qi * 4 + 1],
                       q_cpu[qi * 4 + 2],
                       q_cpu[qi * 4 + 3],
                       q_gpu[qi * 4 + 0],
                       q_gpu[qi * 4 + 1],
                       q_gpu[qi * 4 + 2],
                       q_gpu[qi * 4 + 3]);
        }
        std::fclose(f);
      }
    }
  }
  else {
    BKE_reportf(op->reports,
                RPT_INFO,
                "CPU bake (explicit use_gpu=0 dev override): %d frames x %d bones in %.1f ms",
                frame_count,
                bone_count,
                cpu_ms);
  }

  /* 8) 写出烘焙 Action（链骨用求解结果，其余用 FK 姿态）。 */
  if (armature_obj->adt == nullptr || armature_obj->adt->action == nullptr) {
    wm->runtime->is_interface_locked = false;
    BKE_report(op->reports,
               RPT_WARNING,
               "Bake completed but no source Action exists; nothing written");
    return OPERATOR_FINISHED;
  }
  const std::string bake_name = std::string(armature_obj->adt->action->id.name + 2) + " | Baked";
  animrig::Action &bake_action = animrig::action_add(*bmain, bake_name);
  animrig::Slot &slot = bake_action.slot_add_for_id(armature_obj->id);
  bake_action.layer_keystrip_ensure();
  animrig::Strip &strip = *bake_action.layer(0)->strip(0);
  animrig::Channelbag &bag = strip.data<animrig::StripKeyframeData>(bake_action)
                                 .channelbag_for_slot_add(slot);
  const double t_write0 = BLI_time_now_seconds();

  /* 曲线预创建（单线程：channelbag 的曲线列表操作非线程安全）。 */
  struct BoneCurves {
    std::array<FCurve *, 4> rot{};
    std::array<FCurve *, 3> loc{};
    bool has_loc = false;
    float q_conv[4];
  };
  std::vector<BoneCurves> bone_curves(bone_count);
  for (int i = 0; i < bone_count; i++) {
    Bone *b = pool[i];
    BoneCurves &bc = bone_curves[i];
    if (pchans[i] == nullptr) {
      continue;
    }
    /* q_conv（与采集阶段一致，预计算供并行写出使用）。 */
    float arm_rot[3][3];
    copy_m3_m4(arm_rot, b->arm_mat);
    normalize_m3(arm_rot);
    for (int r = 0; r < 3; r++) {
      std::swap(arm_rot[r][1], arm_rot[r][2]);
    }
    float conv_mat[3][3];
    transpose_m3_m3(conv_mat, arm_rot);
    mat3_to_quat(bc.q_conv, conv_mat);
    normalize_qt(bc.q_conv);

    char escaped[128] = {};
    BLI_str_escape(escaped, b->name, sizeof(escaped));
    const std::string rot_path =
        std::string("pose.bones[\"") + escaped + "\"].rotation_quaternion";
    for (int ch = 0; ch < 4; ch++) {
      animrig::FCurveDescriptor d;
      d.rna_path = rot_path;
      d.array_index = ch;
      d.prop_type = PROP_FLOAT;
      d.prop_subtype = PROP_NONE;
      bc.rot[ch] = &bag.fcurve_ensure(nullptr, d);
    }
    bc.has_loc = bones_have_loc[i];
    if (bc.has_loc) {
      const std::string loc_path = std::string("pose.bones[\"") + escaped + "\"].location";
      for (int ch = 0; ch < 3; ch++) {
        animrig::FCurveDescriptor d;
        d.rna_path = loc_path;
        d.array_index = ch;
        d.prop_type = PROP_FLOAT;
        d.prop_subtype = PROP_TRANSLATION;
        bc.loc[ch] = &bag.fcurve_ensure(nullptr, d);
      }
    }
  }

  /* 并行批量填充：每骨独立曲线，直接构造 BezTriple 数组（时序递增，一次
   * 分配 + 每曲线一次句柄重算）。旧实现逐键 insert_vert_fcurve——每次插入
   * 都重算邻键贝塞尔句柄，15M 键 ≈ 200s+，是烘焙 CPU 耗时的大头。 */
  std::vector<float> key_times(frame_count);
  for (int f = 0; f < frame_count; f++) {
    key_times[f] = float(frame_start) + float(f) * frame_step;
  }

  const auto init_bezier = [&](BezTriple *bezt, const int f, const float value) {
    bezt[f].vec[1][0] = key_times[f];
    bezt[f].vec[1][1] = value;
    bezt[f].ipo = BEZT_IPO_LIN;
    bezt[f].h1 = bezt[f].h2 = HD_AUTO_ANIM;
    bezt[f].f1 = bezt[f].f2 = bezt[f].f3 = BEZT_FLAG_SELECT;
  };
  const auto assign_curve = [&](FCurve &curve, BezTriple *bezt) {
    MEM_SAFE_DELETE(curve.bezt);
    curve.bezt = bezt;
    curve.totvert = frame_count;
    BKE_fcurve_handles_recalc(curve);
  };

  blender::threading::parallel_for(IndexRange(0, bone_count), 1, [&](const IndexRange range) {
    for (const int i : range) {
      BoneCurves &bc = bone_curves[i];
      if (bc.rot[0] == nullptr) {
        continue;
      }
      BezTriple *bq[4];
      for (int ch = 0; ch < 4; ch++) {
        bq[ch] = static_cast<BezTriple *>(
            MEM_new_array_zeroed(size_t(frame_count), sizeof(BezTriple), "mmd_bake_rot_bezt"));
      }
      BezTriple *bl[3] = {nullptr, nullptr, nullptr};
      if (bc.has_loc) {
        for (int ch = 0; ch < 3; ch++) {
          bl[ch] = static_cast<BezTriple *>(
              MEM_new_array_zeroed(size_t(frame_count), sizeof(BezTriple), "mmd_bake_loc_bezt"));
        }
      }

      const bool is_ik_bone = (buffers.bones[i].flags & 2) != 0;
      for (int f = 0; f < frame_count; f++) {
        float bl_q[4];
        /* 链骨仅在其所在链该帧"IK 开启"时采用 CCD 求解结果；门控关闭（VMD
         * IK 开关为 0 或纯 FK 链）时写采样到的 FK 旋转 q_base——与原生回放的
         * IK/FK 逐帧切换一致。 */
        bool use_solved = is_ik_bone;
        if (use_solved) {
          use_solved = false;
          for (const int ci : bone_chains[i]) {
            if (chain_gate[size_t(f) * chain_count + ci] != 0) {
              use_solved = true;
              break;
            }
          }
        }
        if (use_solved) {
          const size_t qi = (size_t(f) * bone_count + i) * 4;
          float mmd_q[4] = {q_out[qi + 0], q_out[qi + 1], q_out[qi + 2], q_out[qi + 3]};
          float cq[4], tq[4];
          conjugate_qt_qt(cq, bc.q_conv);
          mul_qt_qtqt(tq, bc.q_conv, mmd_q);
          mul_qt_qtqt(bl_q, tq, cq);
          normalize_qt(bl_q);
        }
        else {
          /* 非链骨或门控关闭的链骨：写逐帧采样的 FK 旋转（MMD 空间 q_base
           * 反变换回 Blender 空间）。此前这里读姿势通道的"当前"旋转，所有帧
           * 都得到同一个常量（烘焙结束时末帧的姿态），导致烘焙动作里整个
           * 非 IK 躯体冻结不动。 */
          const MmdCCDBakeBuffers::FrameBone &fb =
              buffers.frames[size_t(f) * bone_count + i];
          float cq[4], tq[4];
          conjugate_qt_qt(cq, bc.q_conv);
          mul_qt_qtqt(tq, bc.q_conv, fb.q_base);
          mul_qt_qtqt(bl_q, tq, cq);
          normalize_qt(bl_q);
        }
        for (int ch = 0; ch < 4; ch++) {
          init_bezier(bq[ch], f, bl_q[ch]);
        }
        if (bc.has_loc) {
          for (int ch = 0; ch < 3; ch++) {
            init_bezier(bl[ch], f, sampled_loc[(size_t(f) * bone_count + i) * 3 + ch]);
          }
        }
      }

      for (int ch = 0; ch < 4; ch++) {
        assign_curve(*bc.rot[ch], bq[ch]);
      }
      if (bc.has_loc) {
        for (int ch = 0; ch < 3; ch++) {
          assign_curve(*bc.loc[ch], bl[ch]);
        }
      }
    }
  });
  const double write_ms = (BLI_time_now_seconds() - t_write0) * 1000.0;
  std::fprintf(stderr,
               "[BAKETIME] write action %d bones x %d frames: %.0f ms\n",
               bone_count,
               frame_count,
               write_ms);
  DEG_id_tag_update_ex(bmain, &bake_action.id, ID_RECALC_ANIMATION_NO_FLUSH);
  /* 手动烘焙完成后：把烘焙动作设为活动动作并挂起全部 IK（与旧导入时自动
   * 烘焙的收尾一致），烘焙结果立即接管回放。assign_action 不会自己通知
   * 依赖图，需补 ID_RECALC_ANIMATION 让烘焙动作自返回起生效。 */
  animrig::assign_action(&bake_action, armature_obj->id);
  vmd_suspend_all_ik_after_bake(bmain, armature_obj);
  DEG_id_tag_update_ex(bmain, &armature_obj->id, ID_RECALC_ANIMATION | ID_RECALC_GEOMETRY);
  /* 解锁界面：烘焙结果已就绪，视口一次性刷新为最终姿态。 */
  wm->runtime->is_interface_locked = false;
  BKE_reportf(op->reports,
              RPT_INFO,
              "Baked Action '%s' written (collect %.0f ms + write %.0f ms, total %.0f ms)",
              bake_name.c_str(),
              collect_ms,
              write_ms,
              (BLI_time_now_seconds() - t_collect0) * 1000.0);
  return OPERATOR_FINISHED;
}

void WM_OT_mmd_bake_motion(wmOperatorType *ot)
{
  ot->name = "MMD Bake Motion (GPU)";
  ot->description = "GPU-accelerated CCD IK bake of the active VMD action into FK curves";
  ot->idname = "WM_OT_mmd_bake_motion";
  ot->exec = wm_mmd_bake_motion_exec;
  ot->poll = WM_operator_winactive;
  ot->flag = OPTYPE_UNDO | OPTYPE_PRESET;

  RNA_def_int(ot->srna,
              "frame_start",
              0,
              0,
              1000000,
              "Start Frame",
              "First frame to bake",
              0,
              100000);
  RNA_def_int(ot->srna,
              "frame_end",
              250,
              0,
              1000000,
              "End Frame",
              "Last frame to bake",
              0,
              100000);
  RNA_def_float(ot->srna,
                "coordinate_scale",
                0.08f,
                0.000001f,
                1000.0f,
                "Coordinate Scale",
                "Blender units per MMD coordinate unit",
                0.001f,
                1.0f);
  RNA_def_boolean(ot->srna,
                  "use_gpu",
                  true,
                  "Use GPU",
                  "Solve IK chains on the GPU (Vulkan compute, all vendors); "
                  "falls back to the CPU solver when unavailable");
  RNA_def_int(ot->srna,
              "subframes",
              2,
              1,
              16,
              "Subframes",
              "Samples per frame (2 = double frame rate); sub-frame sampling captures "
              "VMD bezier midpoints so baked curves interpolate linearly with high "
              "precision on export (UE5/FBX)",
              1,
              8);
}

}  // namespace blender
