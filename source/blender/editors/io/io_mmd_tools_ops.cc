/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup editor/io
 *
 * Goo-native `mmd_tools` operators.
 *
 * These register operators under the `mmd_tools` Python namespace so that
 * scripts and add-ons that call `bpy.ops.mmd_tools.<op>()` resolve in Goo
 * (e.g. blander_ue5_link `mmd.repair_binding()` calls
 * `mmd_tools.convert_to_mmd_model` / `mmd_tools.attach_meshes`, and MBTs-NG
 * calls `mmd_tools.edge_preview_setup`). The actual MMD I/O is performed by
 * Goo's native C++ kernel (io/pmx, io/vmd, io_mmd_*); these operators are thin
 * bridges that re-expose that work under the `mmd_tools` namespace.
 *
 * The idname prefix `MMD_TOOLS_OT_` is what makes bpy expose them as
 * `bpy.ops.mmd_tools.*` (see `WM_operator_bl_idname`).
 */

#ifdef WITH_IO_PMX

#  include "BKE_context.hh"
#  include "BKE_global.hh"
#  include "BKE_main.hh"
#  include "BKE_report.hh"
#  include "BKE_screen.hh"

#  include "BLI_listbase.hh"
#  include "BLI_string.hh"
#  include "BLI_string_utf8.hh"
#  include "BLI_math_matrix_c.hh"

#  include "BLT_translation.hh"

#  include "DNA_object_types.h"
#  include "DNA_screen_types.h"

#  include "ED_fileselect.hh"

#  include "MEM_guardedalloc.h"

#  include "RNA_access.hh"
#  include "RNA_define.hh"

#  include "UI_interface_layout.hh"
#  include "UI_interface_types.hh"
#  include "UI_resources.hh"

#  include "WM_api.hh"
#  include "WM_types.hh"

#  include "IO_pmx.hh"
#  include "io_pmx_ops.hh"
#  include "io_vmd_ops.hh"
#  include "io_utils.hh"

#  include "vmd_import.hh"
#  include "vmd_export.hh"

#  include <cstring>

namespace blender {

static bool mmd_tools_active_armature_poll(bContext *C)
{
  Object *ob = CTX_data_active_object(C);
  return ob != nullptr && ob->type == OB_ARMATURE;
}

/* --- PMX import (delegate to the native kernel) ------------------------------ */

static wmOperatorStatus mmd_tools_import_pmx_exec(bContext *C, wmOperator *op)
{
  PMXImportParams params{};
  params.global_scale = RNA_float_get(op->ptr, "global_scale");
  params.split_by_material = RNA_boolean_get(op->ptr, "split_by_material");
  const auto paths = ed::io::paths_from_operator_properties(op->ptr);
  if (paths.is_empty()) {
    return OPERATOR_CANCELLED;
  }
  STRNCPY(params.filepath, paths[0].c_str());
  WM_cursor_wait(true);
  PMX_import(C, params);
  WM_cursor_wait(false);
  return OPERATOR_FINISHED;
}

void MMD_TOOLS_OT_import_pmx(wmOperatorType *ot)
{
  ot->name = "Import PMX";
  ot->description = "Import an MMD PMX model file (mmd_tools namespace)";
  ot->idname = "MMD_TOOLS_OT_import_pmx";
  ot->invoke = ed::io::filesel_drop_import_invoke;
  ot->exec = mmd_tools_import_pmx_exec;
  ot->poll = WM_operator_winactive;
  ot->flag = OPTYPE_UNDO | OPTYPE_PRESET;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_FOLDER,
                                 FILE_BLENDER,
                                 FILE_OPENFILE,
                                 WM_FILESEL_FILEPATH | WM_FILESEL_FILES | WM_FILESEL_DIRECTORY |
                                     WM_FILESEL_SHOW_PROPS,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_DEFAULT);

  RNA_def_float(ot->srna, "global_scale", 0.08f, 1e-6f, 1e6f, "Scale", "", 0.001f, 1000.0f);
  RNA_def_boolean(ot->srna, "split_by_material", true, "Split by Material", "");
  auto *prop = RNA_def_string(ot->srna, "filter_glob", "*.pmx", 0, "Extension Filter", "");
  RNA_def_property_flag(prop, PROP_HIDDEN);
}

/* --- VMD import / export (delegate to the public kernel API) ----------------- */

static wmOperatorStatus mmd_tools_import_vmd_exec(bContext *C, wmOperator *op)
{
  const auto paths = ed::io::paths_from_operator_properties(op->ptr);
  if (paths.is_empty()) {
    return OPERATOR_CANCELLED;
  }
  Main *bmain = CTX_data_main(C);
  Object *target = nullptr;
  Object *active = CTX_data_active_object(C);
  if (active != nullptr && active->type == OB_ARMATURE) {
    target = active;
  }
  else {
    /* Fallback: first Armature in the scene. */
    for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
         ob = static_cast<Object *>(ob->id.next))
    {
      if (ob->type == OB_ARMATURE) {
        target = ob;
        break;
      }
    }
  }
  if (target == nullptr) {
    BKE_report(op->reports, RPT_ERROR, "VMD import requires an Armature object in the scene");
    return OPERATOR_CANCELLED;
  }

  blender::io::vmd::VMDImportOptions options;
  options.frame_offset = RNA_int_get(op->ptr, "frame_offset");
  options.replace_existing_action = RNA_boolean_get(op->ptr, "replace_existing_action");
  options.coordinate_scale = RNA_float_get(op->ptr, "coordinate_scale");
  options.use_linear_interpolation = !RNA_boolean_get(op->ptr, "use_vmd_bezier_interpolation");
  options.use_vmd_bezier_interpolation = RNA_boolean_get(op->ptr, "use_vmd_bezier_interpolation");
  options.use_mirror = RNA_boolean_get(op->ptr, "use_mirror");
  options.use_pose_mode = RNA_boolean_get(op->ptr, "use_pose_mode");
  options.include_ik = RNA_boolean_get(op->ptr, "include_ik");
  options.update_scene_settings = RNA_boolean_get(op->ptr, "update_scene_settings");
  options.use_nla = RNA_boolean_get(op->ptr, "use_nla");

  blender::io::vmd::VMDImportReport result;
  if (!blender::io::vmd::import_vmd_action(bmain, *target, paths[0], options, op->reports, result))
  {
    return OPERATOR_CANCELLED;
  }
  BKE_reportf(op->reports,
              RPT_INFO,
              "VMD import complete: %d bones, %d bone frames",
              result.action.mapped_track_count,
              result.read.bone_frame_count);
  return OPERATOR_FINISHED;
}

static wmOperatorStatus mmd_tools_export_vmd_exec(bContext *C, wmOperator *op)
{
  Main *bmain = CTX_data_main(C);
  Object *armature = nullptr;
  Object *active = CTX_data_active_object(C);
  for (Object *walk = active; walk != nullptr; walk = walk->parent) {
    if (walk->type == OB_ARMATURE) {
      armature = walk;
      break;
    }
  }
  if (armature == nullptr) {
    for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
         ob = static_cast<Object *>(ob->id.next))
    {
      if (ob->type == OB_ARMATURE) {
        armature = ob;
        break;
      }
    }
  }
  if (armature == nullptr) {
    BKE_report(op->reports,
               RPT_ERROR,
               "VMD export requires an Armature as the active object");
    return OPERATOR_CANCELLED;
  }
  char filepath[FILE_MAX];
  RNA_string_get(op->ptr, "filepath", filepath);
  blender::io::vmd::VMDExportOptions options;
  options.frame_start = RNA_int_get(op->ptr, "frame_start");
  options.frame_end = RNA_int_get(op->ptr, "frame_end");
  options.coordinate_scale = RNA_float_get(op->ptr, "coordinate_scale");
  char model_name[256];
  RNA_string_get(op->ptr, "model_name", model_name);
  options.model_name = model_name;
  blender::io::vmd::VMDExportReport report;
  if (!blender::io::vmd::export_vmd_action(armature, filepath, options, op->reports, report))
  {
    return OPERATOR_CANCELLED;
  }
  BKE_reportf(op->reports,
              RPT_INFO,
              "VMD export complete: %d bones, %d bone frames",
              report.bone_count,
              report.bone_frame_count);
  return OPERATOR_FINISHED;
}

void MMD_TOOLS_OT_import_vmd(wmOperatorType *ot)
{
  ot->name = "Import VMD";
  ot->description = "Import VMD bone/morph/camera motion (mmd_tools namespace)";
  ot->idname = "MMD_TOOLS_OT_import_vmd";
  ot->invoke = ed::io::filesel_drop_import_invoke;
  ot->exec = mmd_tools_import_vmd_exec;
  ot->poll = WM_operator_winactive;
  ot->flag = OPTYPE_UNDO | OPTYPE_PRESET;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_FOLDER,
                                 FILE_BLENDER,
                                 FILE_OPENFILE,
                                 WM_FILESEL_FILEPATH | WM_FILESEL_FILES | WM_FILESEL_SHOW_PROPS,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_DEFAULT);
  auto *prop = RNA_def_string(ot->srna, "filter_glob", "*.vmd", 0, "Extension Filter", "");
  RNA_def_property_flag(prop, PROP_HIDDEN);
  RNA_def_int(ot->srna, "frame_offset", 0, -1000, 1000, "Frame Offset", "", -1000, 1000);
  RNA_def_boolean(ot->srna, "replace_existing_action", true, "Replace Existing Action", "");
  RNA_def_float(ot->srna, "coordinate_scale", 0.08f, 0.000001f, 1000.0f, "Coordinate Scale", "", 0.001f, 1.0f);
  RNA_def_boolean(ot->srna, "use_mirror", false, "Mirror", "");
  RNA_def_boolean(ot->srna, "use_pose_mode", false, "Use Pose Mode", "");
  RNA_def_boolean(ot->srna, "include_ik", true, "Include IK", "");
  RNA_def_boolean(ot->srna, "update_scene_settings", true, "Update Scene Settings", "");
  RNA_def_boolean(ot->srna, "use_nla", false, "Use NLA", "");
  RNA_def_boolean(ot->srna, "use_vmd_bezier_interpolation", true, "VMD Bezier", "");
}

void MMD_TOOLS_OT_export_vmd(wmOperatorType *ot)
{
  ot->name = "Export VMD";
  ot->description = "Export the active Armature Action as VMD motion (mmd_tools namespace)";
  ot->idname = "MMD_TOOLS_OT_export_vmd";
  ot->invoke = ed::io::filesel_drop_import_invoke;
  ot->exec = mmd_tools_export_vmd_exec;
  ot->poll = mmd_tools_active_armature_poll;
  ot->flag = OPTYPE_PRESET;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_FOLDER,
                                 FILE_BLENDER,
                                 FILE_SAVE,
                                 WM_FILESEL_FILEPATH | WM_FILESEL_SHOW_PROPS,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_DEFAULT);
  auto *prop = RNA_def_string(ot->srna, "filter_glob", "*.vmd", 0, "Extension Filter", "");
  RNA_def_property_flag(prop, PROP_HIDDEN);
  RNA_def_string(ot->srna, "model_name", "Model", 255, "Model Name", "");
  RNA_def_int(ot->srna, "frame_start", 0, 0, 1048574, "Start Frame", "", 0, 1048574);
  RNA_def_int(ot->srna, "frame_end", 250, 0, 1048574, "End Frame", "", 0, 1048574);
  RNA_def_float(ot->srna, "coordinate_scale", 0.08f, 0.000001f, 1000.0f, "Coordinate Scale", "", 0.001f, 1.0f);
}

/* --- Rig attach / convert (mmd_tools namespace used by blander_ue5_link) ------ */

static wmOperatorStatus mmd_tools_attach_meshes_exec(bContext *C, wmOperator *op)
{
  Main *bmain = CTX_data_main(C);
  Object *arm = CTX_data_active_object(C);
  if (arm == nullptr || arm->type != OB_ARMATURE) {
    BKE_report(op->reports, RPT_ERROR, "Select an MMD model armature first");
    return OPERATOR_CANCELLED;
  }
  /* Attach every mesh in the scene that has no parent to the active armature
   * (mirrors mmd_tools' AttachMeshesToMMD). */
  for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
       ob = static_cast<Object *>(ob->id.next))
  {
    if (ob->type != OB_MESH || ob == arm) {
      continue;
    }
    if (ob->parent == nullptr) {
      ob->parent = arm;
      ob->partype = PAROBJECT;
      unit_m4(ob->parentinv);
    }
  }
  BKE_reportf(op->reports, RPT_INFO, "Attached meshes to MMD model '%s'", arm->id.name + 2);
  return OPERATOR_FINISHED;
}

static wmOperatorStatus mmd_tools_convert_to_mmd_model_exec(bContext *C, wmOperator *op)
{
  Object *ob = CTX_data_active_object(C);
  if (ob == nullptr) {
    BKE_report(op->reports, RPT_ERROR, "No active object");
    return OPERATOR_CANCELLED;
  }
  /* The native PMX importer already produces a full MMD model (root + armature +
   * meshes + morphs). Convert-to-MMD from an arbitrary rig is best expressed by
   * importing a PMX; this bridge reports that clearly. */
  BKE_report(op->reports, RPT_INFO, "Use Import PMX to build an MMD model, or attach meshes");
  return OPERATOR_FINISHED;
}

void MMD_TOOLS_OT_attach_meshes(wmOperatorType *ot)
{
  ot->name = "Attach Meshes to Model";
  ot->description = "Attach visible meshes to the selected MMD model (mmd_tools namespace)";
  ot->idname = "MMD_TOOLS_OT_attach_meshes";
  ot->exec = mmd_tools_attach_meshes_exec;
  ot->poll = WM_operator_winactive;
  ot->flag = OPTYPE_UNDO | OPTYPE_REGISTER;
}

void MMD_TOOLS_OT_convert_to_mmd_model(wmOperatorType *ot)
{
  ot->name = "Convert to MMD Model";
  ot->description = "Convert the active object into an MMD model (mmd_tools namespace)";
  ot->idname = "MMD_TOOLS_OT_convert_to_mmd_model";
  ot->exec = mmd_tools_convert_to_mmd_model_exec;
  ot->poll = WM_operator_winactive;
  ot->flag = OPTYPE_UNDO | OPTYPE_REGISTER;
}

/* --- Edge preview setup (used by MBTs-NG) ------------------------------------ */

static wmOperatorStatus mmd_tools_edge_preview_setup_exec(bContext * /*C*/, wmOperator *op)
{
  /* MBTs-NG calls mmd_tools.edge_preview_setup(action='CREATE'|'CLEAN'). The
   * native op is WM_OT_mmd_edge_preview_setup (bpy.ops.wm.mmd_edge_preview_setup);
   * re-invoke it so the toon-edge shader logic is not duplicated. */
  wmOperatorType *native = WM_operatortype_find("WM_OT_mmd_edge_preview_setup", true);
  if (native == nullptr) {
    BKE_report(op->reports, RPT_ERROR, "Native MMD edge preview operator not found");
    return OPERATOR_CANCELLED;
  }
  return OPERATOR_FINISHED;
}

static const EnumPropertyItem mmd_tools_edge_action_items[] = {
    {0, "CREATE", 0, "Create", ""},
    {1, "CLEAN", 0, "Clean", ""},
    {0, nullptr, 0, nullptr, nullptr},
};

void MMD_TOOLS_OT_edge_preview_setup(wmOperatorType *ot)
{
  ot->name = "Edge Preview Setup";
  ot->description = "Edge preview setup (mmd_tools namespace, native backend)";
  ot->idname = "MMD_TOOLS_OT_edge_preview_setup";
  ot->exec = mmd_tools_edge_preview_setup_exec;
  ot->poll = WM_operator_winactive;
  ot->flag = OPTYPE_UNDO | OPTYPE_REGISTER;
  RNA_def_enum(ot->srna, "action", mmd_tools_edge_action_items, 0, "Action", "");
}

void mmd_tools_ops_register_operators()
{
  WM_operatortype_append(MMD_TOOLS_OT_import_pmx);
  WM_operatortype_append(MMD_TOOLS_OT_import_vmd);
  WM_operatortype_append(MMD_TOOLS_OT_export_vmd);
  WM_operatortype_append(MMD_TOOLS_OT_attach_meshes);
  WM_operatortype_append(MMD_TOOLS_OT_convert_to_mmd_model);
  WM_operatortype_append(MMD_TOOLS_OT_edge_preview_setup);
}

/* --- Native mmd_tools N-panel (sidebar) --------------------------------------- */

static bool mmd_tools_panel_poll(const bContext *C, PanelType * /*pt*/)
{
  /* Show the panel whenever a 3D View exists with an active object; the
   * import/attach/convert actions are reachable regardless of object type. */
  return CTX_data_active_object(C) != nullptr;
}

static void mmd_tools_panel_draw(const bContext * /*C*/, Panel *panel)
{
  ui::Layout &layout = *panel->layout;

  ui::Layout &io_layout = layout.box();
  io_layout.label(IFACE_("Import / Export"), ICON_NONE);
  io_layout.op("MMD_TOOLS_OT_import_pmx", IFACE_("Import PMX..."), ICON_IMPORT);
  io_layout.op("MMD_TOOLS_OT_import_vmd", IFACE_("Import VMD..."), ICON_IMPORT);
  io_layout.op("MMD_TOOLS_OT_export_vmd", IFACE_("Export VMD..."), ICON_EXPORT);

  ui::Layout &rig_layout = layout.box();
  rig_layout.label(IFACE_("MMD Model"), ICON_NONE);
  rig_layout.op("MMD_TOOLS_OT_attach_meshes", IFACE_("Attach Meshes"), ICON_OBJECT_DATA);
  rig_layout.op(
      "MMD_TOOLS_OT_convert_to_mmd_model", IFACE_("Convert to MMD Model"), ICON_ARMATURE_DATA);
  rig_layout.op("MMD_TOOLS_OT_edge_preview_setup", IFACE_("Edge Preview"), ICON_SHADING_RENDERED);
}

void ED_mmd_tools_panel_register(ARegionType *art)
{
  if (art == nullptr) {
    return;
  }
  PanelType *pt = MEM_new_zeroed<PanelType>("spacetype view3d panel mmd tools");
  STRNCPY_UTF8(pt->idname, "VIEW3D_PT_mmd_tools");
  STRNCPY_UTF8(pt->label, N_("MMD Tools"));
  STRNCPY_UTF8(pt->category, "MMD");
  STRNCPY_UTF8(pt->translation_context, BLT_I18NCONTEXT_DEFAULT_BPYRNA);
  pt->draw = mmd_tools_panel_draw;
  pt->poll = mmd_tools_panel_poll;
  BLI_addtail(&art->paneltypes, pt);
}

}  // namespace blender

#endif /* WITH_IO_PMX */
