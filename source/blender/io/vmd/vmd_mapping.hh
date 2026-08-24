/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup io_vmd
 */

#pragma once

#include "IO_vmd.hh"

#include <cstdint>
#include <string>
#include <vector>

namespace blender::io::vmd {

struct VMDMappedBoneTrack {
  std::string vmd_bone_name;
  std::string armature_bone_name;
  int target_bone_index = -1;
  int keyframe_count = 0;
  uint32_t first_frame = 0;
  uint32_t last_frame = 0;
  std::vector<size_t> keyframe_indices;
  /* Empty when matched by exact name; otherwise records the adaptation path
   * used ("alias", "normalized", "d_bone", "mirror") for the import report. */
  std::string matched_via;
  /* True when the motion mirror option flips this track across X: the keyframe
   * values must be mirrored (translation X negated, rotation Y/Z negated). */
  bool use_mirror = false;
};

struct VMDMissingBoneTrack {
  std::string vmd_bone_name;
  int keyframe_count = 0;
  uint32_t first_frame = 0;
  uint32_t last_frame = 0;
  uint64_t first_source_offset = 0;
};

struct VMDMappingIssue {
  enum class Severity : uint8_t {
    Info = 0,
    Warning = 1,
    Error = 2,
  };

  Severity severity = Severity::Warning;
  std::string path;
  std::string message;
  uint64_t source_offset = 0;
};

struct VMDMappingReport {
  bool target_valid = false;
  bool mapping_valid = false;
  int target_bone_count = 0;
  int vmd_track_count = 0;
  int mapped_track_count = 0;
  int missing_track_count = 0;
  /* Number of tracks matched through name adaptation (alias, normalization,
   * or D-bone fallback) instead of an exact name match. */
  int adapted_track_count = 0;
  int mapped_keyframe_count = 0;
  int ignored_keyframe_count = 0;
  int duplicate_track_frame_count = 0;
  int empty_name_track_count = 0;
  int first_frame = -1;
  int last_frame = -1;
  std::vector<std::string> target_bone_names;
  std::vector<VMDMappedBoneTrack> mapped_tracks;
  std::vector<VMDMissingBoneTrack> missing_tracks;
  std::vector<VMDMappingIssue> issues;
};

/**
 * Map VMD bone tracks to one explicit Armature name snapshot.
 *
 * The function only compares UTF-8 names and returns a pure in-memory report.
 * It does not access or modify Blender data, create Actions, or create F-Curves.
 */
VMDMappingReport map_bone_tracks(const VMDModel &model,
                                 const std::vector<std::string> &target_bone_names,
                                 const bool mirror = false);

}  // namespace blender::io::vmd
