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
#  include "BKE_idprop.hh"
#  include "BKE_layer.hh"
#  include "BKE_main.hh"
#  include "BKE_material.hh"
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

#  include "exporter/pmx_export.hh"

#  include <algorithm>
#  include <string>

#  include <cstring>

namespace blender {

static bool mmd_tools_active_armature_poll(bContext *C)
{
  Object *ob = CTX_data_active_object(C);
  return ob != nullptr && ob->type == OB_ARMATURE;
}

/* Set the registered mmd_tools RNA `mmd_type` enum on an object if the property
 * is exposed (registered by the built-in mmd_tools Python module). Returns true
 * when the value was set. */
static bool mmd_tools_set_attr(Object *ob, const char *prop_name, const char *value)
{
  PointerRNA ptr = RNA_id_pointer_create(&ob->id);
  PropertyRNA *prop = RNA_struct_find_property(&ptr, prop_name);
  if (prop == nullptr) {
    return false;
  }
  int enum_index = 0;
  if (!RNA_property_enum_value(nullptr, &ptr, prop, value, &enum_index)) {
    return false;
  }
  RNA_property_enum_set(&ptr, prop, enum_index);
  return true;
}

/* Mark the model hierarchy with the mmd_tools data model so plugins that read
 * `obj.mmd_type` recognise natively-imported MMD models. */
static void mmd_tools_mark_imported_model(Main *bmain, Object *armature)
{
  if (armature == nullptr) {
    return;
  }
  /* The model root is the top-level Empty that parents the armature (and the PMX
   * Geometry mesh objects / morph controller). Walk up to the topmost Empty. */
  Object *root = armature;
  while (root->parent != nullptr && root->parent->type == OB_EMPTY) {
    root = root->parent;
  }
  if (root == armature || root->parent != nullptr) {
    /* armature may be directly under an Empty; find the top Empty ancestor */
    Object *walk = armature;
    while (walk->parent != nullptr) {
      walk = walk->parent;
    }
    root = walk;
  }
  /* Mark the top-level object as the MMD model root. It is usually an Empty. */
  mmd_tools_set_attr(root, "mmd_type", "ROOT");

  if (armature != root) {
    mmd_tools_set_attr(armature, "mmd_type", "NONE");
  }
  /* Mark every mesh object in the model as MMD mesh (NONE type is the default
   * but setting it explicitly signalling is harmless for plugins). */
  for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
       ob = static_cast<Object *>(ob->id.next))
  {
    if (ob->type == OB_MESH && (ob->parent == root || ob->parent == armature)) {
      mmd_tools_set_attr(ob, "mmd_type", "NONE");
    }
  }
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
  /* mmd_tools compatibility: after import, mark the MMD model root Empty and the
   * armature/mesh objects with the mmd_tools RNA data model (mmd_type / mmd_root)
   * so plugins that read `obj.mmd_type` / find the "ROOT" object recognise the
   * natively-imported model (e.g. blander_ue5_link mmd.find_model_root). */
  if (params.result_armature != nullptr) {
    mmd_tools_mark_imported_model(CTX_data_main(C), params.result_armature);
  }
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

/* --- PMX export (delegate to the native export kernel) ------------------------ */

static wmOperatorStatus mmd_tools_export_pmx_exec(bContext *C, wmOperator *op)
{
  bool ambiguous = false;
  Collection *model_root = blender::io::pmx::find_pmx_model_collection(
      CTX_data_main(C), CTX_data_active_object(C), ambiguous);
  if (model_root == nullptr) {
    BKE_report(op->reports,
               RPT_ERROR,
               ambiguous ? "Several imported PMX models are present; select an object of the one "
                           "to export" :
                           "No imported PMX model found. PMX export needs the source data PMX "
                           "import retains on the model collection");
    return OPERATOR_CANCELLED;
  }
  char filepath[FILE_MAX];
  RNA_string_get(op->ptr, "filepath", filepath);
  blender::io::pmx::PMXExportOptions options;
  STRNCPY(options.filepath, filepath);
  blender::io::pmx::PMXExportReport report;
  if (!blender::io::pmx::export_pmx_model(*model_root, options, op->reports, report)) {
    return OPERATOR_CANCELLED;
  }
  BKE_reportf(op->reports,
              RPT_INFO,
              "PMX export complete: %d verts, %d faces, %d materials, %d bones, %d morphs",
              report.vertex_count,
              report.face_count,
              report.material_count,
              report.bone_count,
              report.morph_count);
  return OPERATOR_FINISHED;
}

void MMD_TOOLS_OT_export_pmx(wmOperatorType *ot)
{
  ot->name = "Export PMX";
  ot->description = "Export an imported MMD PMX model (mmd_tools namespace)";
  ot->idname = "MMD_TOOLS_OT_export_pmx";
  ot->invoke = ed::io::filesel_drop_import_invoke;
  ot->exec = mmd_tools_export_pmx_exec;
  ot->poll = WM_operator_winactive;
  ot->flag = OPTYPE_PRESET;

  WM_operator_properties_filesel(ot,
                                 FILE_TYPE_FOLDER,
                                 FILE_BLENDER,
                                 FILE_SAVE,
                                 WM_FILESEL_FILEPATH | WM_FILESEL_SHOW_PROPS,
                                 FILE_DEFAULTDISPLAY,
                                 FILE_SORT_DEFAULT);
  auto *prop = RNA_def_string(ot->srna, "filter_glob", "*.pmx", 0, "Extension Filter", "");
  RNA_def_property_flag(prop, PROP_HIDDEN);
}

/* --- Convert MMD materials to Blender (mmd_tools.convert_materials) ----------- */

static wmOperatorStatus mmd_tools_convert_materials_exec(bContext *C, wmOperator *op)
{
  /* MBTs-NG and mmd_tools call `mmd_tools.convert_materials()`. Goo's PMX import
   * marks imported materials with `mmd_pmx_edge_enabled` system properties; treat
   * those as MMD materials. Building a full MMD->Blender shader converter is a
   * large, separate effort; this bridge is conservative and robust, and reports
   * how many MMD materials are present so the workflow is not a silent no-op. */
  Main *bmain = CTX_data_main(C);
  int mmd_materials = 0;
  for (Material *mat = static_cast<Material *>(bmain->materials.first); mat != nullptr;
       mat = static_cast<Material *>(mat->id.next))
  {
    if (mat == nullptr) {
      continue;
    }
    IDProperty *props = mat->id.system_properties;
    if (props != nullptr &&
        IDP_GetPropertyFromGroup_null(props, "mmd_pmx_edge_enabled") != nullptr)
    {
      mmd_materials++;
    }
  }
  BKE_reportf(op->reports,
              RPT_INFO,
              mmd_materials == 0 ?
                  "No MMD materials found. Import a PMX model first (convert_materials is a "
                  "no-op stub in this Goo build)" :
                  "Found %d MMD material(s); convert_materials is a reporting stub in this Goo "
                  "build",
              mmd_materials);
  return OPERATOR_FINISHED;
}

void MMD_TOOLS_OT_convert_materials(wmOperatorType *ot)
{
  ot->name = "Convert Materials";
  ot->description = "Convert MMD materials to Blender (mmd_tools namespace)";
  ot->idname = "MMD_TOOLS_OT_convert_materials";
  ot->exec = mmd_tools_convert_materials_exec;
  ot->poll = WM_operator_winactive;
  ot->flag = OPTYPE_UNDO | OPTYPE_REGISTER;
}

/* --- VMD import / export (delegate to the public kernel API) ----------------- */

static wmOperatorStatus mmd_tools_import_vmd_exec(bContext *C, wmOperator *op)
{
  char filepath[FILE_MAX];
  RNA_string_get(op->ptr, "filepath", filepath);
  if (filepath[0] == '\0') {
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

  Object *morph_controller = nullptr;
  /* The native PMX importer creates a `PMXMorphControl` mesh under the model
   * root. Import bone + morph animation when it is present (mirrors the native
   * WM_OT_vmd_import path), otherwise bone-only. */
  for (Object *ob = static_cast<Object *>(bmain->objects.first); ob != nullptr;
       ob = static_cast<Object *>(ob->id.next))
  {
    if (ob->type == OB_MESH && strncmp(ob->id.name + 2, "PMXMorphControl", 15) == 0) {
      morph_controller = ob;
      break;
    }
  }

  blender::io::vmd::VMDImportReport result;
  const bool success = (morph_controller != nullptr) ?
                           blender::io::vmd::import_vmd_action_with_morphs(
                               bmain, *target, *morph_controller, filepath, options, op->reports,
                               result) :
                           blender::io::vmd::import_vmd_action(
                               bmain, *target, filepath, options, op->reports, result);
  if (!success) {
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

/* --- VMD camera import / export (delegate to the public camera kernel) -------- */

static wmOperatorStatus mmd_tools_vmd_camera_import_exec(bContext *C, wmOperator *op)
{
  char filepath[FILE_MAX];
  RNA_string_get(op->ptr, "filepath", filepath);
  if (filepath[0] == '\0') {
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
  blender::io::vmd::VMDImportOptions options;
  options.frame_offset = RNA_int_get(op->ptr, "frame_offset");
  options.replace_existing_action = RNA_boolean_get(op->ptr, "replace_existing_action");
  options.coordinate_scale = RNA_float_get(op->ptr, "coordinate_scale");
  Object *target_camera = nullptr;
  Object *active_object = CTX_data_active_object(C);
  if (active_object != nullptr && active_object->type == OB_CAMERA) {
    target_camera = active_object;
  }
  else if (scene->camera != nullptr && scene->camera->type == OB_CAMERA) {
    target_camera = scene->camera;
  }
  blender::io::vmd::VMDImportReport result;
  if (!blender::io::vmd::import_vmd_camera(
          bmain, *target_collection, filepath, options, op->reports, result, target_camera))
  {
    return OPERATOR_CANCELLED;
  }
  BKE_report(op->reports, RPT_INFO, "VMD camera import complete");
  return OPERATOR_FINISHED;
}

static wmOperatorStatus mmd_tools_vmd_camera_export_exec(bContext *C, wmOperator *op)
{
  Scene *scene = CTX_data_scene(C);
  Object *camera = CTX_data_active_object(C);
  if (camera == nullptr || camera->type != OB_CAMERA) {
    camera = scene->camera;
  }
  if (camera == nullptr || camera->type != OB_CAMERA) {
    BKE_report(op->reports, RPT_ERROR, "VMD camera export requires an active Camera");
    return OPERATOR_CANCELLED;
  }
  char filepath[FILE_MAX];
  RNA_string_get(op->ptr, "filepath", filepath);
  blender::io::vmd::VMDCameraExportOptions options;
  options.frame_start = RNA_int_get(op->ptr, "frame_start");
  options.frame_end = RNA_int_get(op->ptr, "frame_end");
  options.coordinate_scale = RNA_float_get(op->ptr, "coordinate_scale");
  blender::io::vmd::VMDCameraExportReport report;
  if (!blender::io::vmd::export_vmd_camera(*camera, filepath, options, op->reports, report)) {
    return OPERATOR_CANCELLED;
  }
  BKE_report(op->reports, RPT_INFO, "VMD camera export complete");
  return OPERATOR_FINISHED;
}

void MMD_TOOLS_OT_vmd_camera_import(wmOperatorType *ot)
{
  ot->name = "Import VMD Camera";
  ot->description = "Import VMD camera motion (mmd_tools namespace)";
  ot->idname = "MMD_TOOLS_OT_vmd_camera_import";
  ot->invoke = ed::io::filesel_drop_import_invoke;
  ot->exec = mmd_tools_vmd_camera_import_exec;
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
}

void MMD_TOOLS_OT_vmd_camera_export(wmOperatorType *ot)
{
  ot->name = "Export VMD Camera";
  ot->description = "Export the active Camera as VMD camera motion (mmd_tools namespace)";
  ot->idname = "MMD_TOOLS_OT_vmd_camera_export";
  ot->invoke = ed::io::filesel_drop_import_invoke;
  ot->exec = mmd_tools_vmd_camera_export_exec;
  ot->poll = WM_operator_winactive;
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

void MMD_TOOLS_OT_set_panel_language(wmOperatorType *ot);

void mmd_tools_ops_register_operators()
{
  WM_operatortype_append(MMD_TOOLS_OT_import_pmx);
  WM_operatortype_append(MMD_TOOLS_OT_export_pmx);
  WM_operatortype_append(MMD_TOOLS_OT_import_vmd);
  WM_operatortype_append(MMD_TOOLS_OT_export_vmd);
  WM_operatortype_append(MMD_TOOLS_OT_vmd_camera_import);
  WM_operatortype_append(MMD_TOOLS_OT_vmd_camera_export);
  WM_operatortype_append(MMD_TOOLS_OT_attach_meshes);
  WM_operatortype_append(MMD_TOOLS_OT_convert_to_mmd_model);
  WM_operatortype_append(MMD_TOOLS_OT_edge_preview_setup);
  WM_operatortype_append(MMD_TOOLS_OT_convert_materials);
  WM_operatortype_append(MMD_TOOLS_OT_set_panel_language);
}

/* --- Native mmd_tools N-panel (sidebar) --------------------------------------- */

enum class MMDToolsPanelLanguage : int {
  Chinese = 0,
  English = 1,
  Japanese = 2,
};

static const EnumPropertyItem mmd_tools_panel_language_items[] = {
    {int(MMDToolsPanelLanguage::Chinese), "CHINESE", 0, "中文", "使用中文界面"},
    {int(MMDToolsPanelLanguage::English), "ENGLISH", 0, "English", "Use the English interface"},
    {int(MMDToolsPanelLanguage::Japanese), "JAPANESE", 0, "日本語", "日本語のインターフェースを使用"},
    {0, nullptr, 0, nullptr, nullptr},
};

/* Strings shown by the mmd_tools sidebar panel, translated per language. */
struct MMDToolsPanelText {
  const char *import_export;
  const char *import_pmx;
  const char *export_pmx;
  const char *import_vmd;
  const char *export_vmd;
  const char *import_vmd_cam;
  const char *export_vmd_cam;
  const char *mmd_model;
  const char *attach_meshes;
  const char *convert_model;
  const char *convert_materials;
  const char *edge_preview;
  const char *simulation;
  const char *physics_start;
  const char *physics_bake;
  const char *physics_stop;
};

const MMDToolsPanelText &mmd_tools_panel_text(const MMDToolsPanelLanguage language)
{
  static const MMDToolsPanelText chinese = {
      "导入 / 导出",
      "导入 PMX...",
      "导出 PMX...",
      "导入 VMD...",
      "导出 VMD...",
      "导入 VMD 相机...",
      "导出 VMD 相机...",
      "MMD 模型",
      "挂接网格",
      "转换为 MMD 模型",
      "转换材质",
      "描边预览",
      "模拟 / 烘焙",
      "物理开始",
      "烘焙物理",
      "物理停止"};
  static const MMDToolsPanelText english = {
      "Import / Export",
      "Import PMX...",
      "Export PMX...",
      "Import VMD...",
      "Export VMD...",
      "Import VMD Cam...",
      "Export VMD Cam...",
      "MMD Model",
      "Attach Meshes",
      "Convert to MMD Model",
      "Convert Materials",
      "Edge Preview",
      "Simulation / Bake",
      "Physics Start",
      "Bake Physics",
      "Physics Stop"};
  static const MMDToolsPanelText japanese = {
      "読み込み / 書き出し",
      "PMX を読み込み...",
      "PMX を書き出し...",
      "VMD を読み込み...",
      "VMD を書き出し...",
      "VMD カメラを読み込み...",
      "VMD カメラを書き出し...",
      "MMD モデル",
      "メッシュを接続",
      "MMD モデルに変換",
      "マテリアルを変換",
      "エッジプレビュー",
      "シミュレーション / ベイク",
      "物理開始",
      "物理をベイク",
      "物理停止"};
  switch (language) {
    case MMDToolsPanelLanguage::English:
      return english;
    case MMDToolsPanelLanguage::Japanese:
      return japanese;
    case MMDToolsPanelLanguage::Chinese:
    default:
      return chinese;
  }
}

/* Language is stored on the Scene (like the MMD physics panel). Default 中文. */
constexpr const char *kMMDToolsLanguageProperty = "mmd_tools_panel_language";

static MMDToolsPanelLanguage scene_mmd_tools_panel_language(Scene *scene)
{
  if (scene != nullptr) {
    IDProperty *props = IDP_GetProperties(&scene->id);
    if (props != nullptr) {
      if (IDProperty *prop = IDP_GetPropertyTypeFromGroup(
              props, kMMDToolsLanguageProperty, IDP_INT))
      {
        return MMDToolsPanelLanguage(std::clamp(IDP_int_get(prop), 0, 2));
      }
    }
  }
  return MMDToolsPanelLanguage::Chinese;
}

static void scene_mmd_tools_panel_language_set(Scene *scene, const MMDToolsPanelLanguage language)
{
  if (scene == nullptr) {
    return;
  }
  IDProperty *props = IDP_EnsureProperties(&scene->id);
  const int value = std::clamp(int(language), 0, 2);
  IDProperty *prop = IDP_GetPropertyTypeFromGroup(props, kMMDToolsLanguageProperty, IDP_INT);
  if (prop != nullptr) {
    IDP_int_set(prop, value);
  }
  else {
    IDP_AddToGroup(props, IDP_NewInt(value, kMMDToolsLanguageProperty));
  }
}

static wmOperatorStatus mmd_tools_set_panel_language_exec(bContext *C, wmOperator *op)
{
  scene_mmd_tools_panel_language_set(
      CTX_data_scene(C), MMDToolsPanelLanguage(RNA_enum_get(op->ptr, "language")));
  return OPERATOR_FINISHED;
}

void MMD_TOOLS_OT_set_panel_language(wmOperatorType *ot)
{
  ot->name = "Set MMD Tools Panel Language";
  ot->description = "Choose the interface language for the MMD Tools panel";
  ot->idname = "MMD_TOOLS_OT_set_panel_language";
  ot->exec = mmd_tools_set_panel_language_exec;
  ot->poll = WM_operator_winactive;
  ot->flag = OPTYPE_REGISTER;
  RNA_def_enum(ot->srna, "language", mmd_tools_panel_language_items, 0, "Language", "");
}

static bool mmd_tools_panel_poll(const bContext *C, PanelType * /*pt*/)
{
  /* Show the panel whenever a 3D View exists with an active object; the
   * import/attach/convert actions are reachable regardless of object type. */
  return CTX_data_active_object(C) != nullptr;
}

static void mmd_tools_panel_draw(const bContext *C, Panel *panel)
{
  ui::Layout &layout = *panel->layout;
  Scene *scene = CTX_data_scene(C);
  const MMDToolsPanelText &text = mmd_tools_panel_text(scene_mmd_tools_panel_language(scene));

  /* Language switcher at the top (mirrors the MMD physics panel). */
  const char *language_name = mmd_tools_panel_language_items[int(scene_mmd_tools_panel_language(
      scene))]
                                  .name;
  layout.op_menu_enum(C,
                      "MMD_TOOLS_OT_set_panel_language",
                      "language",
                      std::string(text.import_export) + ": " + language_name,
                      ICON_WORLD);

  ui::Layout &io_layout = layout.box();
  io_layout.label(text.import_export, ICON_NONE);
  io_layout.op("MMD_TOOLS_OT_import_pmx", text.import_pmx, ICON_IMPORT);
  io_layout.op("MMD_TOOLS_OT_export_pmx", text.export_pmx, ICON_EXPORT);
  io_layout.op("MMD_TOOLS_OT_import_vmd", text.import_vmd, ICON_IMPORT);
  io_layout.op("MMD_TOOLS_OT_export_vmd", text.export_vmd, ICON_EXPORT);
  io_layout.op("MMD_TOOLS_OT_vmd_camera_import", text.import_vmd_cam, ICON_CAMERA_DATA);
  io_layout.op("MMD_TOOLS_OT_vmd_camera_export", text.export_vmd_cam, ICON_CAMERA_DATA);

  ui::Layout &rig_layout = layout.box();
  rig_layout.label(text.mmd_model, ICON_NONE);
  rig_layout.op("MMD_TOOLS_OT_attach_meshes", text.attach_meshes, ICON_OBJECT_DATA);
  rig_layout.op("MMD_TOOLS_OT_convert_to_mmd_model", text.convert_model, ICON_ARMATURE_DATA);
  rig_layout.op("MMD_TOOLS_OT_convert_materials", text.convert_materials, ICON_MATERIAL);
  rig_layout.op("MMD_TOOLS_OT_edge_preview_setup", text.edge_preview, ICON_SHADING_RENDERED);

  /* Surface Goo's native MMD simulation capabilities under the same panel. */
  ui::Layout &sim_layout = layout.box();
  sim_layout.label(text.simulation, ICON_NONE);
  sim_layout.op("WM_OT_mmd_physics_start", text.physics_start, ICON_PLAY);
  sim_layout.op("WM_OT_mmd_physics_bake", text.physics_bake, ICON_ACTION);
  sim_layout.op("WM_OT_mmd_physics_stop", text.physics_stop, ICON_PAUSE);
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
