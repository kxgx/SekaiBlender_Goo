/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup io_vmd
 */

#pragma once

#include "vmd_action.hh"
#include "vmd_camera_action.hh"
#include "vmd_morph_action.hh"
#include "vmd_morph_mapping.hh"

#include <string>
#include <vector>

namespace blender {
struct Main;
struct Collection;
struct Object;
struct ReportList;

namespace io::vmd {

struct VMDImportOptions {
  int frame_offset = 0;
  bool replace_existing_action = false;
  float coordinate_scale = 0.08f;
  bool use_linear_interpolation = true;
  bool use_vmd_bezier_interpolation = false;
  /* mmd_tools 选项移植（R2-VMD）：镜像整段动作（左右翻转目标 + X 镜像值）。 */
  bool use_mirror = false;
  /* 把当前姿势当作基准姿势（Treat Current Pose as Rest Pose），
   * 适配 T-Pose/A-Pose 与模型当前姿势不一致导致的位置漂移。 */
  bool use_pose_mode = false;
  /* 导入 VMD IK 开关轨道（默认开启，配合原生 CCD 求解器）。 */
  bool include_ik = true;
  /* 导入后自动设置场景帧率(30fps)与帧范围（默认开启）。 */
  bool update_scene_settings = true;
  /* R3-VMD (mmd_tools parity): import the Action as an NLA strip instead of
   * binding it as the Armature's active Action. */
  bool use_nla = false;
  /* R3-VMD: hard-cut detection for camera tracks. */
  bool detect_camera_changes = true;
};

struct VMDImportReport {
  bool success = false;
  VMDReadReport read;
  VMDMappingReport mapping;
  VMDActionReport action;
  VMDCameraActionReport camera_action;
  VMDMorphMappingReport morph_mapping;
  VMDMorphActionReport morph_action;
  std::vector<std::string> warnings;
  std::vector<std::string> errors;
};

/**
 * Import the bone animation from one VMD file into one explicit Armature Object.
 *
 * This is the only orchestration entry point for the editor operator. It keeps the
 * Reader, mapping, and Action stages in the VMD module and never searches for a
 * target object globally.
 */
bool import_vmd_action(Main *bmain,
                       Object &target_armature,
                       const std::string &filepath,
                       const VMDImportOptions &options,
                       ReportList *reports,
                       VMDImportReport &r_result);

/**
 * Import bone and vertex-morph animation into explicit PMX model targets.
 *
 * Unlike the legacy bone-only entry point, this function requires the caller to
 * provide the PMXMorphController object. It never searches Blender data globally.
 */
bool import_vmd_action_with_morphs(Main *bmain,
                                   Object &target_armature,
                                   Object &target_morph_controller,
                                   const std::string &filepath,
                                   const VMDImportOptions &options,
                                   ReportList *reports,
                                   VMDImportReport &r_result);

/** Import the camera section of one VMD file into a new native MMD camera rig. */
bool import_vmd_camera(Main *bmain,
                       Collection &target_collection,
                       const std::string &filepath,
                       const VMDImportOptions &options,
                       ReportList *reports,
                       VMDImportReport &r_result,
                       Object *target_camera = nullptr);

}  // namespace io::vmd
}  // namespace blender
