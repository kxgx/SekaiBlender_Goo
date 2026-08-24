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
#include "ANIM_fcurve.hh"
#include "BKE_armature.hh"
#include "BKE_context.hh"
#include "BKE_fcurve.hh"
#include "BKE_lib_id.hh"
#include "BKE_pose.hh"
#include "BKE_report.hh"
#include "BKE_scene.hh"

#include "DEG_depsgraph.hh"

#include "ED_object.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "BLI_math_matrix_c.hh"
#include "BLI_math_rotation_c.hh"
#include "BLI_fileops.hh"
#include "BLI_path_utils.hh"
#include "BLI_string.hh"
#include "BLI_time.hh"

#include "DNA_action_types.h"
#include "DNA_constraint_types.h"
#include "DNA_armature_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "mmd_ccd_ik_bake.hh"
#include "importer/pmx_import_bone_ik.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
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
  const int frame_start = RNA_int_get(op->ptr, "frame_start");
  const int frame_end = RNA_int_get(op->ptr, "frame_end");
  const bool use_gpu = RNA_boolean_get(op->ptr, "use_gpu");
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

  /* 4) 临时关闭 IK 约束（采集纯 FK）。 */
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

  /* 5) 逐帧采集。 */
  Depsgraph *depsgraph = CTX_data_depsgraph_pointer(C);
  const int frame_count = frame_end - frame_start + 1;
  buffers.frame_count = frame_count;
  buffers.frames.resize(size_t(frame_count) * size_t(bone_count));
  for (int f = frame_start; f <= frame_end; f++) {
    scene->r.cfra = f;
    BKE_scene_graph_update_for_newframe(depsgraph);
    const size_t frame_base = size_t(f - frame_start) * bone_count;
    for (int i = 0; i < bone_count; i++) {
      Bone *b = pool[i];
      bPoseChannel *pchan = BKE_pose_channel_find_name(armature_obj->pose, b->name);
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
    }
  }

  /* 6) 恢复约束。 */
  for (auto &[con, value] : saved_enforce) {
    con->enforce = value;
  }

  /* 7) 求解：GPU 为主，CPU 参照用于精度报告。 */
  std::vector<float> q_gpu;
  std::vector<float> q_cpu;
  const double t0 = BLI_time_now_seconds();
  const bool gpu_ok = use_gpu && blender::mmd::mmd_ccd_bake_gpu(buffers, q_gpu);
  const double gpu_ms = (BLI_time_now_seconds() - t0) * 1000.0;
  const double t1 = BLI_time_now_seconds();
  blender::mmd::mmd_ccd_bake_cpu_reference(buffers, q_cpu);
  const double cpu_ms = (BLI_time_now_seconds() - t1) * 1000.0;

  const std::vector<float> &q_out = gpu_ok ? q_gpu : q_cpu;

  if (gpu_ok) {
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
                "GPU bake: %d frames x %d bones in %.1f ms (CPU %.1f ms); "
                "quat error (sign-aware) max %.6f mean %.6f",
                frame_count,
                bone_count,
                gpu_ms,
                cpu_ms,
                max_err,
                sum_err / double(std::max<size_t>(quat_count, 1)));

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
        for (size_t qi = 0; qi < quat_count; qi++) {
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
                "GPU unavailable; CPU bake: %d frames x %d bones in %.1f ms",
                frame_count,
                bone_count,
                cpu_ms);
  }

  /* 8) 写出烘焙 Action（链骨用求解结果，其余用 FK 姿态）。 */
  if (armature_obj->adt == nullptr || armature_obj->adt->action == nullptr) {
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
  const animrig::KeyframeSettings key_settings = {
      BEZT_KEYTYPE_KEYFRAME, HD_AUTO_ANIM, BEZT_IPO_LIN};

  for (int i = 0; i < bone_count; i++) {
    Bone *b = pool[i];
    bPoseChannel *pchan = BKE_pose_channel_find_name(armature_obj->pose, b->name);
    if (pchan == nullptr) {
      continue;
    }
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

    char escaped[128] = {};
    BLI_str_escape(escaped, b->name, sizeof(escaped));
    const std::string rot_path = std::string("pose.bones[\"") + escaped + "\"].rotation_quaternion";

    std::array<FCurve *, 4> rot_curves{};
    for (int ch = 0; ch < 4; ch++) {
      animrig::FCurveDescriptor d;
      d.rna_path = rot_path;
      d.array_index = ch;
      d.prop_type = PROP_FLOAT;
      d.prop_subtype = PROP_NONE;
      rot_curves[ch] = &bag.fcurve_ensure(nullptr, d);
    }
    const bool is_ik_bone = (buffers.bones[i].flags & 2) != 0;
    for (int f = 0; f < frame_count; f++) {
      float bl_q[4];
      if (is_ik_bone) {
        const size_t qi = (size_t(f) * bone_count + i) * 4;
        float mmd_q[4] = {q_out[qi + 0], q_out[qi + 1], q_out[qi + 2], q_out[qi + 3]};
        float cq[4], tq[4];
        conjugate_qt_qt(cq, q_conv);
        mul_qt_qtqt(tq, q_conv, mmd_q);
        mul_qt_qtqt(bl_q, tq, cq);
        normalize_qt(bl_q);
      }
      else {
        const float4 v = BKE_pchan_rot_to_quat(*pchan);
        copy_qt_qt(bl_q, v);
      }
      for (int ch = 0; ch < 4; ch++) {
        animrig::insert_vert_fcurve(
            rot_curves[ch], {float(frame_start + f), bl_q[ch]}, key_settings, INSERTKEY_FAST);
      }
    }
    for (FCurve *curve : rot_curves) {
      BKE_fcurve_handles_recalc(*curve);
    }
  }
  DEG_id_tag_update_ex(bmain, &bake_action.id, ID_RECALC_ANIMATION_NO_FLUSH);
  BKE_reportf(op->reports, RPT_INFO, "Baked Action '%s' written", bake_name.c_str());
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
}

}  // namespace blender
