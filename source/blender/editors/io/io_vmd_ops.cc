/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup editor/io
 */

#include "io_vmd_ops.hh"

#include "BKE_anim_data.hh"
#include "BKE_nla.hh"
#include "BKE_armature.hh"
#include "BKE_context.hh"
#include "BKE_file_handler.hh"
#include "BKE_idprop.hh"
#include "BKE_layer.hh"
#include "BKE_lib_id.hh"
#include "BKE_main.hh"
#include "BKE_report.hh"

#include "mmd_ccd_ik.hh"

#include <cmath>
#include <cstring>
#include <set>
#include <string>

#include "BLI_string.hh"
#include "BLI_string_utf8.hh"
#include "BLI_listbase.hh"
#include "BLI_path_utils.hh"

#include "DEG_depsgraph.hh"
#include "DEG_depsgraph_build.hh"

#include "DNA_action_types.h"
#include "DNA_anim_types.h"
#include "DNA_constraint_types.h"
#include "DNA_key_types.h"
#include "DNA_mesh_types.h"
#include "DNA_object_types.h"
#include "DNA_scene_types.h"

#include "ED_fileselect.hh"
#include "ED_screen.hh"

#include "RNA_access.hh"
#include "RNA_define.hh"

#include "UI_interface.hh"
#include "UI_interface_layout.hh"

#include "WM_api.hh"
#include "WM_types.hh"

#include "vmd_import.hh"
#include "vmd_export.hh"
#include "io_utils.hh"

#include "importer/pmx_import_bone_ik.hh"
#include "ANIM_action.hh"

namespace blender {
namespace {

bool vmd_target_poll(bContext *C)
{
  return WM_operator_winactive(C);
}

Object *vmd_morph_controller_from_operator(Main *bmain, wmOperator *op)
{
  ID *controller_id = WM_operator_properties_id_lookup_from_name_or_session_uid(
      bmain, op->ptr, ID_OB);
  return controller_id ? id_cast<Object *>(controller_id) : nullptr;
}

/* Auto-detect the PMXMorphController mesh that belongs to the target armature.
 * In a PMX import both the armature and the morph controller are parented to
 * the same model root object (see pmx_import_armature.cc and
 * pmx_import_morph_controller.cc), so we walk up from the armature to its parent
 * and scan that parent's children for a mesh named "PMXMorphController". A
 * scene-wide fallback covers single-model scenes. Returns nullptr when no such
 * object exists. */
static Object *vmd_find_morph_controller(Main *bmain, Object *target)
{
  if (target == nullptr) {
    return nullptr;
  }

  /* Anchors: the armature itself, and its parent (the model root). The morph
   * controller is a sibling of the armature under the model root, but we also
   * check the armature as anchor in case the hierarchy differs. */
  Object *anchors[2] = {target, target->parent};
  for (int i = 0; i < 2; i++) {
    Object *anchor = anchors[i];
    if (anchor == nullptr) {
      continue;
    }
    for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
         ob = static_cast<Object *>(ob->id.next))
    {
      if (ob->parent != anchor || ob->type != OB_MESH) {
        continue;
      }
      /* Allow the ".001" style suffix Blender adds for duplicate models. */
      if (std::strncmp(ob->id.name + 2, "PMXMorphController", 18) == 0) {
        return ob;
      }
    }
  }

  /* Fallback: any PMXMorphController mesh in the scene (single-model case). */
  for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
       ob = static_cast<Object *>(ob->id.next))
  {
    if (ob->type == OB_MESH &&
        std::strncmp(ob->id.name + 2, "PMXMorphController", 18) == 0)
    {
      return ob;
    }
  }
  return nullptr;
}

/* Base items for the dynamic "Target Armature" enum. The full list (including
 * every armature in the scene) is produced by vmd_target_armature_itemf. */
static const EnumPropertyItem vmd_target_base_items[] = {
    {0, "ACTIVE", 0, "Active / Auto", "Use the active armature, or the first armature found in the scene"},
    {0, nullptr, 0, nullptr, nullptr},
};

static const EnumPropertyItem *vmd_target_armature_itemf(bContext *C,
                                                         PointerRNA * /*ptr*/,
                                                         PropertyRNA * /*prop*/,
                                                         bool *r_free)
{
  if (C == nullptr) {
    /* Needed for docs and i18n tools. */
    return vmd_target_base_items;
  }

  EnumPropertyItem *item = nullptr;
  int totitem = 0;
  EnumPropertyItem tmp = {0};

  tmp.identifier = "ACTIVE";
  tmp.name = "Active / Auto";
  tmp.description = "Use the active armature, or the first armature found in the scene";
  tmp.value = 0;
  RNA_enum_item_add(&item, &totitem, &tmp);

  Main *bmain = CTX_data_main(C);
  int ordinal = 1;
  for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
       ob = static_cast<Object *>(ob->id.next))
  {
    if (ob->type != OB_ARMATURE) {
      continue;
    }
    /* Identifier reuses the (persistent) object name; resolution is by value,
     * so non-ASCII / spaced MMD names are fine as an opaque key. */
    tmp.identifier = ob->id.name + 2;
    tmp.name = ob->id.name + 2;
    tmp.description = "Armature object";
    tmp.value = ordinal++;
    RNA_enum_item_add(&item, &totitem, &tmp);
  }

  RNA_enum_item_end(&item, &totitem);
  *r_free = true;
  return item;
}

/* Resolve the armature the VMD motion is applied onto. The "Target Armature"
 * enum is used (explicit pick, or ACTIVE -> active / first armature). The
 * WM_operator_properties_id_lookup session_uid is reserved for the morph
 * controller and is NOT consumed here. Returns nullptr (with a report) when no
 * armature exists. */
/* [世界的歌] When a VMD is imported, the motion file usually bakes MMD's
 * IK / append-transform / axis solve into per-bone keyframes. The Blender-side
 * approximations created by PMX auto-apply (or by the manual Apply operators)
 * would fight those baked curves, producing broken motion. So we suspend every
 * MMD_* approximation constraint that double-applies a baked effect (Local/Fixed
 * axis; see vmd_suspend_mmd_approx_constraints below). The IK constraints
 * themselves (MMD_IK_Approx / MMD_IK_Limit) stay active and solve each leg/arm
 * chain toward its IK control bone every frame — exactly what mmd_tools does in
 * Blender 5.0 — with the per-frame IK/FK switch driven by the VMD's IK toggle
 * track (mmd_ik_toggle / influence F-Curves). The native CCD V8 solver is
 * disabled during VMD playback so it never double-solves the same chain. */

/* mmd_tools（Blender 5.0）对齐：VMD 播放期间的 IK 解算完全交给 iTaSC——
 * PMX 导入创建的 MMD_IK_Approx（IK 约束，subtarget=IK 控制骨）与
 * MMD_IK_Limit（LIMIT_ROTATION）始终保持激活，腿链逐帧解向 IK 控制骨
 * （其位置由 VMD 位移轨道驱动），与 mmd_tools 的机制一致。
 *
 * IK/FK 逐帧切换由 apply_vmd_ik_toggle 写入的 mmd_ik_toggle / influence
 * F-Curve 驱动；VMD 没有开关轨道时 influence 保持 1.0（IK 常开）——同样
 * 与 mmd_tools 默认一致。此前"混合链启发式"（按 FK 曲线覆盖分类后挂起
 * 纯 FK 链、给混合链开原生 CCD）偏离 mmd_tools：IK 舞蹈里腿链 FK 轨道是
 * 录制时的残留姿态，播放它们会让腿僵在绑定姿态（"腿找原点"），而原生
 * CCD 与 iTaSC 双重求解又会互相覆盖导致腿部扭曲。原生 CCD V8 是实时
 * 摆姿势路径专用，VMD 播放一律禁用。 */
static int vmd_suspend_mmd_approx_constraints(Main *bmain, Object *ob)
{
  if (ob->pose == nullptr) {
    return 0;
  }

  /* 禁用全部 IK 控制骨的原生 CCD V8（VMD 播放 = iTaSC，与 mmd_tools 一致）。 */
  io::pmx::PMXBoneIKDefinitionSet ik_def;
  if (io::pmx::read_bone_ik_definition(ob->id, ik_def)) {
    bArmature *arm = id_cast<bArmature *>(ob->data);
    if (arm != nullptr) {
      for (const io::pmx::PMXBoneIKDefinition &def : ik_def.ik_bones) {
        Bone *ik_bone = BKE_armature_find_bone_name(arm, def.bone_name.c_str());
        if (ik_bone != nullptr) {
          mmd::mmd_native_ik_set_enabled(*ik_bone, false);
        }
      }
    }
  }

  int suspended = 0;
  int kept = 0;
  for (bPoseChannel *pchan = static_cast<bPoseChannel *>(ob->pose->chanbase.first); pchan != nullptr;
       pchan = pchan->next) {
    for (bConstraint *con = static_cast<bConstraint *>(pchan->constraints.first); con != nullptr;
         con = con->next) {
      /* MMD_Local_Axis_Approx: self-referential TRANSFORM (ADD mode) that
       * ADDs a bone's own local-space rotation to itself. It is needed for
       * real-time posing, but VMD keyframes are already baked in the correct
       * local-axis frame, so ADDing again double-counts the rotation and
       * breaks the pose (e.g. NXDE frame 108 arms twist behind the head).
       * mmd_tools does not create any equivalent constraint on arm bones.
       *
       * Unconditionally suspend this constraint when a VMD is driving the
       * pose — it must NOT be kept active even when the VMD ships an IK
       * toggle track (the IK toggle only governs MMD_IK_Approx/Limit, not
       * local-axis approximation). */
      if (strcmp(con->name, "MMD_Local_Axis_Approx") == 0) {
        if (con->enforce != 0.0f) {
          con->enforce = 0.0f;
          suspended++;
        }
        continue;
      }

      /* MMD_Fixed_Axis_Approx: LIMIT_ROTATION constraint that restricts a
       * bone's rotation to its fixed axis (used by 捩/twist bones like
       * 左腕捩, 左手捩). VMD keyframes are already baked in the correct
       * fixed-axis frame, so applying the limit again double-restricts the
       * rotation and breaks the pose (e.g. NXDE frame 108 左ひじ twists
       * behind the head because 左腕捩's LIMIT_ROTATION clips the baked
       * rotation, propagating wrong parent transform down the arm chain).
       * mmd_tools uses bone.lock_rotation instead of a constraint, which
       * does not conflict with VMD keyframes.
       *
       * Unconditionally suspend when a VMD is driving the pose. D bones use
       * MMD_Append_Rotation (not MMD_Fixed_Axis_Approx), so the D bone
       * chain is unaffected. */
      if (strcmp(con->name, "MMD_Fixed_Axis_Approx") == 0) {
        if (con->enforce != 0.0f) {
          con->enforce = 0.0f;
          suspended++;
        }
        continue;
      }

      /* IK 相关约束（MMD_IK_Approx / MMD_IK_Limit）：始终保留激活。
       * 与 mmd_tools 一致——IK 约束常驻，per-frame 开关走 influence 曲线；
       * 无开关轨道时 influence=1.0（IK 常开，mmd_tools 默认）。
       * MMD_Append_Rotation 保持激活——D 骨链需要它跟随源骨。 */
      if (strcmp(con->name, "MMD_IK_Approx") != 0 &&
          strcmp(con->name, "MMD_IK_Limit") != 0)
      {
        continue;
      }
      if (con->enforce == 0.0f) {
        con->enforce = 1.0f;
      }
      kept++;
    }
  }
  if (suspended > 0 || kept > 0) {
    DEG_id_tag_update_ex(bmain, &ob->id, ID_RECALC_GEOMETRY | ID_RECALC_TRANSFORM);
    DEG_relations_tag_update(bmain);
  }
  return suspended;
}

static bool vmd_is_rigify_bridge_armature(const Object &object){
  if (object.type != OB_ARMATURE) {
    return false;
  }
  if (object.id.properties != nullptr &&
      IDP_GetPropertyFromGroup_null(object.id.properties, "mmd_rigify_mode") != nullptr)
  {
    return true;
  }
  if (object.pose == nullptr) {
    return false;
  }
  for (const bPoseChannel *pchan = static_cast<const bPoseChannel *>(object.pose->chanbase.first);
       pchan != nullptr;
       pchan = pchan->next)
  {
    for (const bConstraint *constraint = static_cast<const bConstraint *>(
             pchan->constraints.first);
         constraint != nullptr;
         constraint = constraint->next)
    {
      if (strncmp(constraint->name, "MMR_", 4) == 0) {
        return true;
      }
    }
  }
  return false;
}

static Object *vmd_find_rigify_source(Main *bmain, const Object *active)
{
  if (active == nullptr || strncmp(active->id.name + 2, "RIG-", 4) != 0) {
    return nullptr;
  }
  for (Object *object = static_cast<Object *>(bmain->objects.first); object != nullptr;
       object = static_cast<Object *>(object->id.next))
  {
    if (object != active && vmd_is_rigify_bridge_armature(*object)) {
      return object;
    }
  }
  return nullptr;
}

static Object *vmd_resolve_target(bContext *C, wmOperator *op, Main *bmain)
{
  int target_value = 0;
  PropertyRNA *target_prop = RNA_struct_find_property(op->ptr, "target");
  if (target_prop != nullptr) {
    target_value = RNA_property_enum_get(op->ptr, target_prop);
  }

  if (target_value <= 0) {
    Object *active = CTX_data_active_object(C);
    if (active != nullptr && active->type == OB_ARMATURE) {
      if (Object *source = vmd_find_rigify_source(bmain, active)) {
        BKE_reportf(op->reports,
                    RPT_INFO,
                    "Active Rigify armature '%s'; using linked MMD VMD target '%s'",
                    active->id.name + 2,
                    source->id.name + 2);
        return source;
      }
      return active;
    }
    for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
         ob = static_cast<Object *>(ob->id.next))
    {
      if (ob->type == OB_ARMATURE) {
        BKE_reportf(op->reports,
                    RPT_INFO,
                    "No active armature; using '%s' as the VMD target",
                    ob->id.name + 2);
        return ob;
      }
    }
    return nullptr;
  }

  /* Resolve by ordinal among armatures (stable across dropdown rebuilds). */
  int ordinal = 1;
  for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
       ob = static_cast<Object *>(ob->id.next))
  {
    if (ob->type != OB_ARMATURE) {
      continue;
    }
    if (ordinal == target_value) {
      return ob;
    }
    ordinal++;
  }
  return nullptr;
}

/* A generated MMD -> Rigify scene keeps the VMD Action on the original MMD
 * armature.  Its MMR_* constraints are the reverse (Rigify -> MMD) direction,
 * so they must yield while that Action is playing.  The Python bridge creates
 * the mode property and drivers; this small editor-side switch makes VMD
 * import enter PLAYBACK automatically. */
static bool vmd_activate_rigify_playback_mode(Main *bmain, Object &target, ReportList *reports)
{
  if (!vmd_is_rigify_bridge_armature(target)) {
    return false;
  }

  IDProperty *properties = IDP_EnsureProperties(&target.id);
  IDProperty *mode = IDP_GetPropertyTypeFromGroup(properties, "mmd_rigify_mode", IDP_FLOAT);
  if (mode != nullptr) {
    IDP_float_set(mode, 1.0f);
  }
  else {
    if (IDProperty *old = IDP_GetPropertyFromGroup_null(properties, "mmd_rigify_mode")) {
      IDP_FreeFromGroup(properties, old);
    }
    IDP_AddToGroup(properties, blender::bke::idprop::create("mmd_rigify_mode", 1.0f).release());
  }

  /* The Rigify bridge disables the native MMD CCD solver in POSE mode.  VMD
   * playback must restore it so IK toggle tracks and mixed FK/IK motions work.
   */
  if (IDProperty *override_prop = IDP_GetPropertyFromGroup_null(
          properties, "mmd_native_ik_override"))
  {
    if (override_prop->type == IDP_BOOLEAN) {
      IDP_bool_set(override_prop, false);
    }
    else if (override_prop->type == IDP_INT) {
      IDP_int_set(override_prop, 0);
    }
  }

  DEG_id_tag_update_ex(bmain, &target.id, ID_RECALC_ANIMATION | ID_RECALC_GEOMETRY);
  BKE_reportf(reports,
              RPT_INFO,
              "Rigify PLAYBACK mode enabled on '%s' for VMD source animation",
              target.id.name + 2);
  return true;
}

wmOperatorStatus wm_vmd_import_exec(bContext *C, wmOperator *op)
{
  const auto paths = ed::io::paths_from_operator_properties(op->ptr);
  if (paths.is_empty()) {
    return OPERATOR_CANCELLED;
  }

  Main *bmain = CTX_data_main(C);

  Object *target = vmd_resolve_target(C, op, bmain);
  if (target == nullptr) {
    BKE_report(op->reports, RPT_ERROR, "VMD import requires an Armature object in the scene");
    return OPERATOR_CANCELLED;
  }

  const bool import_morphs = RNA_boolean_get(op->ptr, "import_morphs");
  Object *morph_controller = vmd_morph_controller_from_operator(bmain, op);
  if (morph_controller == nullptr && target != nullptr) {
    morph_controller = vmd_find_morph_controller(bmain, target);
  }
  if (import_morphs && morph_controller != nullptr) {
    BKE_reportf(op->reports,
                RPT_INFO,
                "Using '%s' as the VMD morph controller",
                morph_controller->id.name + 2);
  }
  if (import_morphs && morph_controller == nullptr) {
    BKE_report(op->reports,
               RPT_ERROR,
               "VMD Morph import requires an explicit PMXMorphController object");
    return OPERATOR_CANCELLED;
  }

  io::vmd::VMDImportOptions options;
  options.frame_offset = RNA_int_get(op->ptr, "frame_offset");
  options.replace_existing_action = RNA_boolean_get(op->ptr, "replace_existing_action");
  options.coordinate_scale = RNA_float_get(op->ptr, "coordinate_scale");
  options.use_mirror = RNA_boolean_get(op->ptr, "use_mirror");
  options.use_pose_mode = RNA_boolean_get(op->ptr, "use_pose_mode");
  options.include_ik = RNA_boolean_get(op->ptr, "include_ik");
  options.update_scene_settings = RNA_boolean_get(op->ptr, "update_scene_settings");
  options.use_nla = RNA_boolean_get(op->ptr, "use_nla");
  options.use_vmd_bezier_interpolation = RNA_boolean_get(op->ptr, "use_vmd_bezier_interpolation");
  if (options.use_vmd_bezier_interpolation) {
    /* C3: VMD-native bezier interpolation implies non-linear bone tracks. */
    options.use_linear_interpolation = false;
  }

  io::vmd::VMDImportReport result;
  const bool success = import_morphs ? io::vmd::import_vmd_action_with_morphs(
                                           bmain,
                                           *target,
                                           *morph_controller,
                                           paths[0],
                                           options,
                                           op->reports,
                                           result) :
                                       io::vmd::import_vmd_action(
                                           bmain, *target, paths[0], options, op->reports, result);
  if (!success) {
    return OPERATOR_CANCELLED;
  }

  vmd_activate_rigify_playback_mode(bmain, *target, op->reports);

  /* Suspend MMD approximate constraints that would double-apply baked effects
   * (Local/Fixed axis). The IK constraints stay active and solve chains toward
   * their IK control bones every frame — mmd_tools parity. */
  const int suspended = vmd_suspend_mmd_approx_constraints(bmain, target);
  if (suspended > 0) {
    BKE_reportf(op->reports,
                RPT_INFO,
                "Suspended %d MMD axis-approximation constraints. "
                "IK chains stay active (iTaSC, mmd_tools parity) with per-frame "
                "IK/FK switch from the VMD toggle track.",
                suspended);
  }

  /* R3-VMD (mmd_tools Use NLA): move the freshly imported Actions onto NLA
   * tracks so the Armature keeps a free active Action slot. */
  if (options.use_nla) {
    const auto move_to_nla = [&](ID &animated_id, const int first_frame, const int last_frame) {
      AnimData *adt = BKE_animdata_from_id(&animated_id);
      if (adt == nullptr || adt->action == nullptr) {
        return;
      }
      bAction *action = adt->action;
      NlaTrack *track = BKE_nlatrack_new_tail(&adt->nla_tracks, false);
      if (track == nullptr) {
        return;
      }
      STRNCPY_UTF8(track->name, action->id.name + 2);
      NlaStrip *strip = BKE_nlastrip_new(action, animated_id);
      if (strip == nullptr) {
        return;
      }
      strip->start = float(first_frame >= 0 ? first_frame : 1);
      strip->end = float(last_frame >= first_frame ? last_frame + 1 : strip->start + 1.0f);
      BKE_nlatrack_add_strip(track, strip, false);
      BKE_animdata_set_action(nullptr, &animated_id, nullptr);
      BKE_reportf(op->reports,
                  RPT_INFO,
                  "Moved VMD Action to NLA track '%s'",
                  track->name);
    };
    move_to_nla(target->id, result.action.first_frame, result.action.last_frame);
    if (import_morphs) {
      Mesh *controller_mesh = id_cast<Mesh *>(morph_controller->data);
      if (controller_mesh != nullptr && controller_mesh->key != nullptr) {
        move_to_nla(controller_mesh->key->id,
                    result.morph_action.first_frame,
                    result.morph_action.last_frame);
      }
    }
  }

  /* 记录 VMD 源动作名到目标 Object 的 ID 属性（GPU 烘焙门控回退引用：
   * 烘焙完成后活动动作会被替换为无 mmd_ik_toggle 曲线的烘焙结果，重复
   * 烘焙必须回到源动作读取开关曲线，保证两次烘焙的门控一致）。 */
  if (AnimData *adt = BKE_animdata_from_id(&target->id)) {
    if (adt->action != nullptr) {
      IDProperty *properties = IDP_EnsureProperties(&target->id);
      if (IDProperty *existing = IDP_GetPropertyFromGroup_null(properties, "mmd_source_action")) {
        IDP_AssignString(existing, adt->action->id.name + 2);
      }
      else {
        IDP_AddToGroup(properties, IDP_NewString(adt->action->id.name + 2, "mmd_source_action"));
      }
    }
  }

  if (AnimData *anim_data = BKE_animdata_from_id(&target->id)) {
    DEG_id_tag_update_ex(bmain, &target->id, ID_RECALC_ANIMATION);
    if (anim_data->action != nullptr) {
      DEG_id_tag_update_ex(bmain, &anim_data->action->id, ID_RECALC_ANIMATION_NO_FLUSH);
    }
  }
  if (import_morphs) {
    DEG_id_tag_update_ex(bmain,
                         &morph_controller->id,
                         ID_RECALC_ANIMATION | ID_RECALC_GEOMETRY);
    Mesh *controller_mesh = id_cast<Mesh *>(morph_controller->data);
    if (controller_mesh != nullptr) {
      DEG_id_tag_update_ex(bmain,
                           &controller_mesh->id,
                           ID_RECALC_ANIMATION | ID_RECALC_GEOMETRY);
    }
    Key *controller_key = controller_mesh ? controller_mesh->key : nullptr;
    if (controller_key != nullptr) {
      DEG_id_tag_update_ex(bmain,
                           &controller_key->id,
                           ID_RECALC_ANIMATION | ID_RECALC_GEOMETRY);
      if (AnimData *anim_data = BKE_animdata_from_id(&controller_key->id)) {
        if (anim_data->action != nullptr) {
          DEG_id_tag_update_ex(
              bmain, &anim_data->action->id, ID_RECALC_ANIMATION_NO_FLUSH);
        }
      }
    }
  }
  DEG_relations_tag_update(bmain);
  if (Depsgraph *depsgraph = CTX_data_depsgraph_pointer(C)) {
    ED_update_for_newframe(bmain, depsgraph);
  }

  Scene *scene = CTX_data_scene(C);
  /* Update scene frame range to match the imported VMD data. */
  if (options.update_scene_settings) {
    int vmd_start = result.action.first_frame;
    int vmd_end = result.action.last_frame;
    if (!result.morph_action.skipped) {
      if (result.morph_action.first_frame >= 0) {
        vmd_start = vmd_start < 0 ? result.morph_action.first_frame :
                                    (result.morph_action.first_frame < vmd_start ?
                                         result.morph_action.first_frame :
                                         vmd_start);
      }
      if (result.morph_action.last_frame >= 0) {
        vmd_end = vmd_end < 0 ? result.morph_action.last_frame :
                                (result.morph_action.last_frame > vmd_end ?
                                     result.morph_action.last_frame :
                                     vmd_end);
      }
    }
    if (vmd_start >= 0 && vmd_end >= 0 && vmd_end > vmd_start) {
      scene->r.sfra = vmd_start;
      scene->r.efra = vmd_end;
      BKE_reportf(op->reports,
                  RPT_INFO,
                  "Scene frame range set to match VMD: %d–%d",
                  vmd_start,
                  vmd_end);
    }
  }
  /* MMD motions are authored at 30 FPS and VMD carries no FPS field. Keeping
   * the scene at another frame rate would play the motion at the wrong speed,
   * so switch the scene to the MMD standard rate and report the change. */
  if (options.update_scene_settings && scene->r.frs_sec != 30.0f) {
    scene->r.frs_sec = 30.0f;
    BKE_report(op->reports,
               RPT_INFO,
               "Scene FPS set to 30 (MMD standard; VMD has no FPS field)");
  }
  WM_event_add_notifier(C, NC_ANIMATION | ND_NLA_ACTCHANGE, nullptr);
  WM_event_add_notifier(C, NC_ANIMATION | ND_ANIMCHAN | NA_EDITED, nullptr);
  WM_event_add_notifier(C, NC_SCENE | ND_OB_ACTIVE, scene);

  /* 导入不再自动 GPU 烘焙（mmd_tools 一致：导入即回放源动作，IK 约束
   * 常驻解算）。需要 FK 烘焙时手动运行 WM_OT_mmd_bake_motion
   * （渲染菜单的 GPU 烘焙按钮 / 脚本调用）。 */
  return OPERATOR_FINISHED;
}

wmOperatorStatus wm_vmd_camera_import_exec(bContext *C, wmOperator *op)
{
  const auto paths = ed::io::paths_from_operator_properties(op->ptr);
  if (paths.is_empty()) {
    return OPERATOR_CANCELLED;
  }

  Main *bmain = CTX_data_main(C);
  Scene *scene = CTX_data_scene(C);
  ViewLayer *view_layer = CTX_data_view_layer(C);
  LayerCollection *active_collection = BKE_layer_collection_get_active_editable(view_layer);
  Collection *target_collection = active_collection ? active_collection->collection :
                                                       scene->master_collection;
  if (target_collection == nullptr) {
    BKE_report(op->reports, RPT_ERROR, "VMD camera import requires an editable collection");
    return OPERATOR_CANCELLED;
  }

  io::vmd::VMDImportOptions options;
  options.frame_offset = RNA_int_get(op->ptr, "frame_offset");
  options.coordinate_scale = RNA_float_get(op->ptr, "coordinate_scale");
  options.replace_existing_action = RNA_boolean_get(op->ptr, "replace_existing_action");
  options.use_vmd_bezier_interpolation = RNA_boolean_get(op->ptr,
                                                         "use_vmd_bezier_interpolation");
  options.use_linear_interpolation = !options.use_vmd_bezier_interpolation;
  options.detect_camera_changes = RNA_boolean_get(op->ptr, "detect_camera_changes");

  Object *target_camera = nullptr;
  Object *active_object = CTX_data_active_object(C);
  if (active_object != nullptr && active_object->type == OB_CAMERA) {
    target_camera = active_object;
  }
  else if (scene->camera != nullptr && scene->camera->type == OB_CAMERA) {
    target_camera = scene->camera;
  }
  if (target_camera != nullptr) {
    BKE_reportf(op->reports,
                RPT_INFO,
                "Using existing Camera '%s' for VMD camera import",
                target_camera->id.name + 2);
  }

  io::vmd::VMDImportReport result;
  if (!io::vmd::import_vmd_camera(
          bmain, *target_collection, paths[0], options, op->reports, result, target_camera))
  {
    return OPERATOR_CANCELLED;
  }

  Object *imported_camera = nullptr;
  for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
       ob = static_cast<Object *>(ob->id.next))
  {
    if (result.camera_action.target_camera_name == std::string(ob->id.name + 2)) {
      imported_camera = ob;
      break;
    }
  }
  if (imported_camera != nullptr) {
    scene->camera = imported_camera;
    DEG_id_tag_update_ex(bmain, &imported_camera->id, ID_RECALC_ANIMATION | ID_RECALC_TRANSFORM);
    if (imported_camera->data != nullptr) {
      DEG_id_tag_update_ex(
          bmain, imported_camera->data, ID_RECALC_ANIMATION | ID_RECALC_GEOMETRY);
    }
  }

  if (result.camera_action.first_frame >= 0 && result.camera_action.last_frame >= 0 &&
      result.camera_action.last_frame > result.camera_action.first_frame)
  {
    scene->r.sfra = result.camera_action.first_frame;
    scene->r.efra = result.camera_action.last_frame;
    BKE_reportf(op->reports,
                RPT_INFO,
                "Scene frame range set to match VMD camera: %d-%d",
                scene->r.sfra,
                scene->r.efra);
  }

  DEG_relations_tag_update(bmain);
  if (Depsgraph *depsgraph = CTX_data_depsgraph_pointer(C)) {
    ED_update_for_newframe(bmain, depsgraph);
  }
  WM_event_add_notifier(C, NC_SCENE | ND_OB_ACTIVE, scene);
  WM_event_add_notifier(C, NC_ANIMATION | ND_NLA_ACTCHANGE, nullptr);
  BKE_reportf(op->reports,
              RPT_INFO,
              "VMD camera import complete: %d keyframes, target '%s', camera '%s'",
              result.camera_action.keyframe_count / 10,
              result.camera_action.target_empty_name.c_str(),
              result.camera_action.target_camera_name.c_str());
  return OPERATOR_FINISHED;
}

void wm_vmd_import_draw(bContext * /*C*/, wmOperator *op)
{
  ui::Layout &layout = *op->layout;
  layout.use_property_split_set(true);
  layout.use_property_decorate_set(false);
  layout.prop(op->ptr, "target", UI_ITEM_NONE, "Target Armature", ICON_NONE);
  layout.prop(op->ptr, "coordinate_scale", UI_ITEM_NONE, "Coordinate Scale", ICON_NONE);
  layout.prop(op->ptr, "frame_offset", UI_ITEM_NONE, "Frame Offset", ICON_NONE);
  layout.prop(op->ptr,
              "replace_existing_action",
              UI_ITEM_NONE,
              "Replace Existing Action",
              ICON_NONE);
  layout.prop(op->ptr, "use_mirror", UI_ITEM_NONE, "Mirror Motion", ICON_NONE);
  layout.prop(op->ptr,
              "use_pose_mode",
              UI_ITEM_NONE,
              "Treat Current Pose as Rest Pose",
              ICON_NONE);
  layout.prop(op->ptr, "include_ik", UI_ITEM_NONE, "Include IK", ICON_NONE);
  layout.prop(op->ptr,
              "update_scene_settings",
              UI_ITEM_NONE,
              "Update Scene Settings",
              ICON_NONE);
  layout.prop(op->ptr, "use_nla", UI_ITEM_NONE, "Use NLA", ICON_NONE);
  layout.prop(op->ptr, "import_morphs", UI_ITEM_NONE, "Import Morphs", ICON_NONE);
  layout.prop(op->ptr,
              "use_vmd_bezier_interpolation",
              UI_ITEM_NONE,
              "VMD Native Bezier",
              ICON_NONE);
}

void wm_vmd_camera_import_draw(bContext * /*C*/, wmOperator *op)
{
  ui::Layout &layout = *op->layout;
  layout.use_property_split_set(true);
  layout.use_property_decorate_set(false);
  layout.prop(op->ptr, "frame_offset", UI_ITEM_NONE, "Frame Offset", ICON_NONE);
  layout.prop(op->ptr, "coordinate_scale", UI_ITEM_NONE, "Coordinate Scale", ICON_NONE);
  layout.prop(op->ptr,
              "replace_existing_action",
              UI_ITEM_NONE,
              "Replace Existing Action",
              ICON_NONE);
  layout.prop(op->ptr,
              "use_vmd_bezier_interpolation",
              UI_ITEM_NONE,
              "VMD Native Bezier",
              ICON_NONE);
  layout.prop(op->ptr,
              "detect_camera_changes",
              UI_ITEM_NONE,
              "Detect Camera Cut",
              ICON_NONE);
}

wmOperatorStatus wm_vmd_export_invoke(bContext *C, wmOperator *op, const wmEvent * /*event*/)
{
  Scene *scene = CTX_data_scene(C);
  if (!RNA_struct_property_is_set(op->ptr, "frame_start")) {
    RNA_int_set(op->ptr, "frame_start", scene->r.sfra);
  }
  if (!RNA_struct_property_is_set(op->ptr, "frame_end")) {
    RNA_int_set(op->ptr, "frame_end", scene->r.efra);
  }
  ED_fileselect_ensure_default_filepath(C, op, ".vmd");
  WM_event_add_fileselect(C, op);
  return OPERATOR_RUNNING_MODAL;
}

wmOperatorStatus wm_vmd_export_exec(bContext *C, wmOperator *op)
{
  /* mmd_tools 风格目标解析：骨架自身、骨架的子孙（网格）、模型根、骨架的兄弟
   * （morph 控制器）都能作为活动对象导出，最后回退到场景里的第一个骨架。 */
  Main *bmain = CTX_data_main(C);
  Object *active = CTX_data_active_object(C);
  Object *armature = nullptr;
  for (Object *walk = active; walk != nullptr; walk = walk->parent) {
    if (walk->type == OB_ARMATURE) {
      armature = walk;
      break;
    }
  }
  if (armature == nullptr) {
    /* 活动对象不是骨架后代：从其根节点下找骨架兄弟（模型根导出场景）。 */
    Object *root = active;
    while (root != nullptr && root->parent != nullptr) {
      root = root->parent;
    }
    for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
         ob = static_cast<Object *>(ob->id.next))
    {
      if (ob->type != OB_ARMATURE) {
        continue;
      }
      if (root != nullptr && (ob->parent == root || ob->parent == active)) {
        armature = ob;
        break;
      }
    }
  }
  if (armature == nullptr) {
    /* 场景回退：第一个骨架（与导入端一致）。 */
    for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
         ob = static_cast<Object *>(ob->id.next))
    {
      if (ob->type == OB_ARMATURE) {
        armature = ob;
        break;
      }
    }
  }
  if (armature != nullptr) {
    BKE_reportf(op->reports,
                RPT_INFO,
                "VMD export using Armature '%s'",
                armature->id.name + 2);
  }

  char filepath[FILE_MAX];
  RNA_string_get(op->ptr, "filepath", filepath);
  io::vmd::VMDExportOptions options;
  options.frame_start = RNA_int_get(op->ptr, "frame_start");
  options.frame_end = RNA_int_get(op->ptr, "frame_end");
  options.coordinate_scale = RNA_float_get(op->ptr, "coordinate_scale");
  char model_name[256];
  RNA_string_get(op->ptr, "model_name", model_name);
  options.model_name = model_name;
  if (RNA_boolean_get(op->ptr, "export_morphs")) {
    if (armature != nullptr) {
      options.morph_controller = vmd_find_morph_controller(bmain, armature);
    }
    /* 活动对象本身是带 Shape Keys 的网格（morph 控制器）时直接采用。 */
    if (options.morph_controller == nullptr && active != nullptr && active->type == OB_MESH) {
      const Mesh *mesh = id_cast<Mesh *>(active->data);
      if (mesh != nullptr && mesh->key != nullptr) {
        options.morph_controller = active;
      }
    }
  }
  if (armature == nullptr && options.morph_controller == nullptr) {
    BKE_report(op->reports,
               RPT_ERROR,
               "VMD export requires an Armature, the model root, or a Morph controller "
               "as the active object");
    return OPERATOR_CANCELLED;
  }
  io::vmd::VMDExportReport report;
  if (!io::vmd::export_vmd_action(armature, filepath, options, op->reports, report)) {
    return OPERATOR_CANCELLED;
  }
  BKE_reportf(op->reports,
              RPT_INFO,
              "VMD export complete: %d bones, %d bone frames, %d morphs, %d morph frames",
              report.bone_count,
              report.bone_frame_count,
              report.morph_count,
              report.morph_frame_count);
  return OPERATOR_FINISHED;
}

void wm_vmd_export_draw(bContext * /*C*/, wmOperator *op)
{
  ui::Layout &layout = *op->layout;
  layout.use_property_split_set(true);
  layout.use_property_decorate_set(false);
  layout.prop(op->ptr, "model_name", UI_ITEM_NONE, "Model Name", ICON_NONE);
  layout.prop(op->ptr, "frame_start", UI_ITEM_NONE, "Start Frame", ICON_NONE);
  layout.prop(op->ptr, "frame_end", UI_ITEM_NONE, "End Frame", ICON_NONE);
  layout.prop(op->ptr, "coordinate_scale", UI_ITEM_NONE, "Coordinate Scale", ICON_NONE);
  layout.prop(op->ptr, "export_morphs", UI_ITEM_NONE, "Export Morphs", ICON_NONE);
}

bool wm_vmd_export_check(bContext * /*C*/, wmOperator *op)
{
  char filepath[FILE_MAX];
  RNA_string_get(op->ptr, "filepath", filepath);
  if (BLI_path_extension_check(filepath, ".vmd")) {
    return false;
  }
  BLI_path_extension_ensure(filepath, FILE_MAX, ".vmd");
  RNA_string_set(op->ptr, "filepath", filepath);
  return true;
}

/* Resolve the Camera whose Action is exported. Selecting either half of a VMD camera rig is
 * accepted because the importer leaves the Empty active, and the Scene camera covers the common
 * case of exporting the shot that is currently being rendered. */
static Object *vmd_resolve_export_camera(bContext *C, Main *bmain)
{
  Object *active = CTX_data_active_object(C);
  if (active != nullptr) {
    if (active->type == OB_CAMERA) {
      return active;
    }
    if (active->type == OB_EMPTY) {
      for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
           ob = static_cast<Object *>(ob->id.next))
      {
        if (ob->parent == active && ob->type == OB_CAMERA) {
          return ob;
        }
      }
    }
  }
  Scene *scene = CTX_data_scene(C);
  if (scene != nullptr && scene->camera != nullptr && scene->camera->type == OB_CAMERA) {
    return scene->camera;
  }
  return nullptr;
}

wmOperatorStatus wm_vmd_camera_export_exec(bContext *C, wmOperator *op)
{
  Main *bmain = CTX_data_main(C);
  Object *camera = vmd_resolve_export_camera(C, bmain);
  if (camera == nullptr) {
    BKE_report(op->reports,
               RPT_ERROR,
               "VMD camera export requires an active Camera, a VMD camera rig Empty, or a Scene "
               "camera");
    return OPERATOR_CANCELLED;
  }
  char filepath[FILE_MAX];
  RNA_string_get(op->ptr, "filepath", filepath);
  io::vmd::VMDCameraExportOptions options;
  options.frame_start = RNA_int_get(op->ptr, "frame_start");
  options.frame_end = RNA_int_get(op->ptr, "frame_end");
  options.coordinate_scale = RNA_float_get(op->ptr, "coordinate_scale");
  char model_name[256];
  RNA_string_get(op->ptr, "model_name", model_name);
  options.model_name = model_name;

  io::vmd::VMDCameraExportReport report;
  if (!io::vmd::export_vmd_camera(*camera, filepath, options, op->reports, report)) {
    return OPERATOR_CANCELLED;
  }
  BKE_reportf(op->reports,
              RPT_INFO,
              "VMD camera export complete: %d frames from '%s' (%s)",
              report.camera_frame_count,
              camera->id.name + 2,
              report.used_camera_rig ? "camera rig" : "standalone camera");
  return OPERATOR_FINISHED;
}

void wm_vmd_camera_export_draw(bContext * /*C*/, wmOperator *op)
{
  ui::Layout &layout = *op->layout;
  layout.use_property_split_set(true);
  layout.use_property_decorate_set(false);
  layout.prop(op->ptr, "model_name", UI_ITEM_NONE, "Model Name", ICON_NONE);
  layout.prop(op->ptr, "frame_start", UI_ITEM_NONE, "Start Frame", ICON_NONE);
  layout.prop(op->ptr, "frame_end", UI_ITEM_NONE, "End Frame", ICON_NONE);
  layout.prop(op->ptr, "coordinate_scale", UI_ITEM_NONE, "Coordinate Scale", ICON_NONE);
}

}  // namespace

/* 烘焙后 FK 动作覆盖全部链骨，无条件挂起所有 MMD 近似 IK 与原生 CCD，
 * 避免 iTaSC 继续把腿解向 足IK 目标、与烘焙动作的纯 FK 结果互相覆盖。
 * 手动 GPU 烘焙（WM_OT_mmd_bake_motion）完成后调用。 */
int vmd_suspend_all_ik_after_bake(Main *bmain, Object *ob)
{
  if (ob->pose == nullptr) {
    return 0;
  }
  int suspended = 0;
  for (bPoseChannel *pchan = static_cast<bPoseChannel *>(ob->pose->chanbase.first);
       pchan != nullptr;
       pchan = pchan->next)
  {
    for (bConstraint *con = static_cast<bConstraint *>(pchan->constraints.first);
         con != nullptr;
         con = con->next)
    {
      if (strcmp(con->name, "MMD_IK_Approx") != 0 && strcmp(con->name, "MMD_IK_Limit") != 0) {
        continue;
      }
      if (con->enforce != 0.0f) {
        con->enforce = 0.0f;
        suspended++;
      }
    }
  }

  /* 原生 CCD V8/V2 一并禁用。 */
  io::pmx::PMXBoneIKDefinitionSet ik_def;
  if (io::pmx::read_bone_ik_definition(ob->id, ik_def)) {
    bArmature *arm = id_cast<bArmature *>(ob->data);
    if (arm != nullptr) {
      for (const io::pmx::PMXBoneIKDefinition &def : ik_def.ik_bones) {
        Bone *ik_bone = BKE_armature_find_bone_name(arm, def.bone_name.c_str());
        if (ik_bone != nullptr) {
          mmd::mmd_native_ik_set_enabled(*ik_bone, false);
        }
      }
    }
  }

  if (suspended > 0) {
    DEG_id_tag_update_ex(bmain, &ob->id, ID_RECALC_GEOMETRY | ID_RECALC_TRANSFORM);
    DEG_relations_tag_update(bmain);
  }
  return suspended;
}

void WM_OT_vmd_import(wmOperatorType *ot)
{
  ot->name = "Import VMD";
  ot->description = "Import VMD bone animation onto a target Armature";
  ot->idname = "WM_OT_vmd_import";
  ot->invoke = ed::io::filesel_drop_import_invoke;
  ot->exec = wm_vmd_import_exec;
  ot->poll = vmd_target_poll;
  ot->ui = wm_vmd_import_draw;
  ot->flag = OPTYPE_UNDO | OPTYPE_PRESET;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_FOLDER,
                                 FILE_BLENDER,
                                 FILE_OPENFILE,
                                 WM_FILESEL_FILEPATH | WM_FILESEL_FILES | WM_FILESEL_DIRECTORY |
                                     WM_FILESEL_SHOW_PROPS,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_DEFAULT);

  PropertyRNA *target_prop = RNA_def_enum(ot->srna,
                                          "target",
                                          vmd_target_base_items,
                                          0,
                                          "Target Armature",
                                          "Armature to apply the VMD motion onto");
  RNA_def_property_enum_funcs_runtime(
      target_prop, nullptr, nullptr, vmd_target_armature_itemf, nullptr, nullptr);

  RNA_def_int(ot->srna,
              "frame_offset",
              30,
              -1000000,
              1000000,
              "Frame Offset",
              "Add this value to every imported VMD frame",
              -1000,
              1000);
  RNA_def_float(ot->srna,
                "coordinate_scale",
                0.08f,
                0.000001f,
                1000.0f,
                "Coordinate Scale",
                "Blender units per MMD coordinate unit (mmd_tools: Scale)",
                0.001f,
                1.0f);
  RNA_def_boolean(ot->srna,
                  "replace_existing_action",
                  true,
                  "Replace Existing Action",
                  "Replace the Armature's existing Action; when disabled, new keyframes "
                  "are written into the existing Action (UPDATE mode)");
  RNA_def_boolean(ot->srna,
                  "use_mirror",
                  false,
                  "Mirror Motion",
                  "Mirror the whole motion across the X axis (side-flipped bone targets "
                  "and mirrored values)");
  RNA_def_boolean(ot->srna,
                  "use_pose_mode",
                  false,
                  "Treat Current Pose as Rest Pose",
                  "Use the model's current pose as the motion base instead of the rest "
                  "pose (for T-Pose / A-Pose mismatches)");
  RNA_def_boolean(ot->srna,
                  "include_ik",
                  true,
                  "Include IK",
                  "Import VMD IK toggle tracks that drive the native CCD solver per frame");
  RNA_def_boolean(ot->srna,
                  "update_scene_settings",
                  true,
                  "Update Scene Settings",
                  "Set the scene frame rate to 30 FPS and the frame range to the VMD range");
  RNA_def_boolean(ot->srna,
                  "use_nla",
                  false,
                  "Use NLA",
                  "Import the motion as NLA strips instead of replacing the Armature's "
                  "active Action");
  RNA_def_boolean(ot->srna,
                  "import_morphs",
                  true,
                  "Import Morphs",
                  "Import Vertex Morph animation into an explicit PMX Morph Controller");
  RNA_def_boolean(ot->srna,
                  "use_vmd_bezier_interpolation",
                  true,
                  "VMD Native Bezier",
                  "Use VMD-embedded bezier interpolation for bone tracks instead of linear");
  WM_operator_properties_id_lookup(ot, false);

  PropertyRNA *prop = RNA_def_string(ot->srna, "filter_glob", "*.vmd", 0, "Extension Filter", "");
  RNA_def_property_flag(prop, PROP_HIDDEN);
}

void WM_OT_vmd_camera_import(wmOperatorType *ot)
{
  ot->name = "Import VMD Camera";
  ot->description = "Import VMD camera animation into a native Empty and Camera rig";
  ot->idname = "WM_OT_vmd_camera_import";
  ot->invoke = ed::io::filesel_drop_import_invoke;
  ot->exec = wm_vmd_camera_import_exec;
  ot->poll = vmd_target_poll;
  ot->ui = wm_vmd_camera_import_draw;
  ot->flag = OPTYPE_UNDO | OPTYPE_PRESET;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_FOLDER,
                                 FILE_BLENDER,
                                 FILE_OPENFILE,
                                 WM_FILESEL_FILEPATH | WM_FILESEL_FILES | WM_FILESEL_DIRECTORY |
                                     WM_FILESEL_SHOW_PROPS,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_DEFAULT);

  RNA_def_int(ot->srna,
              "frame_offset",
              30,
              -1000000,
              1000000,
              "Frame Offset",
              "Add this value to every imported VMD camera frame",
              -1000,
              1000);
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
                  "replace_existing_action",
                  true,
                  "Replace Existing Action",
                  "Replace actions on the imported camera rig when applicable");
  RNA_def_boolean(ot->srna,
                  "detect_camera_changes",
                  true,
                  "Detect Camera Cut",
                  "When consecutive camera keyframes are at most 1 frame apart, use "
                  "CONSTANT interpolation (hard cut) instead of smoothing");
  RNA_def_boolean(ot->srna,
                  "use_vmd_bezier_interpolation",
                  true,
                  "VMD Native Bezier",
                  "Use VMD-embedded Bezier interpolation for camera tracks instead of linear");
  PropertyRNA *prop = RNA_def_string(ot->srna, "filter_glob", "*.vmd", 0, "Extension Filter", "");
  RNA_def_property_flag(prop, PROP_HIDDEN);
}

void WM_OT_vmd_export(wmOperatorType *ot)
{
  ot->name = "Export VMD";
  ot->description = "Export the active Armature Action as VMD bone motion";
  ot->idname = "WM_OT_vmd_export";
  ot->invoke = wm_vmd_export_invoke;
  ot->exec = wm_vmd_export_exec;
  ot->poll = vmd_target_poll;
  ot->ui = wm_vmd_export_draw;
  ot->check = wm_vmd_export_check;
  ot->flag = OPTYPE_PRESET;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_FOLDER,
                                 FILE_BLENDER,
                                 FILE_SAVE,
                                 WM_FILESEL_FILEPATH | WM_FILESEL_SHOW_PROPS,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_DEFAULT);
  RNA_def_string(ot->srna,
                 "model_name",
                 "Model",
                 255,
                 "Model Name",
                 "VMD model name stored in the fixed header");
  RNA_def_int(ot->srna,
              "frame_start",
              0,
              0,
              MAXFRAME,
              "Start Frame",
              "First Action frame to export; it becomes VMD frame zero",
              0,
              MAXFRAME);
  RNA_def_int(ot->srna,
              "frame_end",
              250,
              0,
              MAXFRAME,
              "End Frame",
              "Last Action frame to export",
              0,
              MAXFRAME);
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
                  "export_morphs",
                  true,
                  "Export Morphs",
                  "Export the same model's PMXMorphController Shape Key Action when available");
  PropertyRNA *prop = RNA_def_string(ot->srna, "filter_glob", "*.vmd", 0, "Extension Filter", "");
  RNA_def_property_flag(prop, PROP_HIDDEN);
}

void WM_OT_vmd_camera_export(wmOperatorType *ot)
{
  ot->name = "Export VMD Camera";
  ot->description = "Export a Camera Action as VMD camera motion";
  ot->idname = "WM_OT_vmd_camera_export";
  ot->invoke = wm_vmd_export_invoke;
  ot->exec = wm_vmd_camera_export_exec;
  ot->poll = vmd_target_poll;
  ot->ui = wm_vmd_camera_export_draw;
  ot->check = wm_vmd_export_check;
  ot->flag = OPTYPE_PRESET;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_FOLDER,
                                 FILE_BLENDER,
                                 FILE_SAVE,
                                 WM_FILESEL_FILEPATH | WM_FILESEL_SHOW_PROPS,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_DEFAULT);
  RNA_def_string(ot->srna,
                 "model_name",
                 io::vmd::VMD_CAMERA_MODEL_NAME,
                 255,
                 "Model Name",
                 "VMD header name; MikuMikuDance expects the camera/lighting name here");
  RNA_def_int(ot->srna,
              "frame_start",
              0,
              0,
              MAXFRAME,
              "Start Frame",
              "First Action frame to export; it becomes VMD frame zero",
              0,
              MAXFRAME);
  RNA_def_int(ot->srna,
              "frame_end",
              250,
              0,
              MAXFRAME,
              "End Frame",
              "Last Action frame to export",
              0,
              MAXFRAME);
  RNA_def_float(ot->srna,
                "coordinate_scale",
                0.08f,
                0.000001f,
                1000.0f,
                "Coordinate Scale",
                "Blender units per MMD coordinate unit",
                0.001f,
                1.0f);
  PropertyRNA *prop = RNA_def_string(ot->srna, "filter_glob", "*.vmd", 0, "Extension Filter", "");
  RNA_def_property_flag(prop, PROP_HIDDEN);
}

namespace ed::io {

void vmd_file_handler_add()
{
  auto fh = std::make_unique<bke::FileHandlerType>();
  STRNCPY_UTF8(fh->idname, "IO_FH_vmd");
  STRNCPY_UTF8(fh->import_operator, "WM_OT_vmd_import");
  STRNCPY_UTF8(fh->export_operator, "WM_OT_vmd_export");
  STRNCPY_UTF8(fh->label, "VMD");
  STRNCPY_UTF8(fh->file_extensions_str, ".vmd");
  fh->poll_drop = poll_file_object_drop;
  bke::file_handler_add(std::move(fh));
}

}  // namespace ed::io

}  // namespace blender
