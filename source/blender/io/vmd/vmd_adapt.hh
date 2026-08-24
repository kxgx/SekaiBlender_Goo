/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup io_vmd
 *
 * MMD motion adaptation helpers.
 *
 * VMD motions are authored against one specific model's naming convention.
 * Applying a motion to a model that uses a different convention (Japanese
 * variants such as 左腕捩/左腕捩り, full-width ＩＫ characters, half-width
 * kana, older D-bone names such as 左ひじD, or normalized English bone
 * names) previously failed with "bone not found". These helpers normalize
 * both sides and resolve names through a curated alias table so native
 * import adapts motions automatically. Exact matches always win; adapted
 * matches are reported so callers can surface them in the import report.
 */

#pragma once

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace blender::io::vmd {

/**
 * Normalize an MMD bone/morph name for comparison:
 * - trims leading/trailing whitespace (ASCII space and ideographic space)
 * - full-width ASCII (０-９, Ａ-Ｚ, ！-～) to half-width
 * - half-width katakana (ｱ-ﾝ) to katakana
 * - katakana to hiragana (MMD morph names are canonically hiragana)
 * - small kana (ぁぃぅぇぉゃゅょゎっ...) to normal kana
 * - ASCII upper case to lower case
 *
 * The result is for comparison only; original names are preserved elsewhere.
 */
std::string normalize_mmd_name(std::string_view name);

/**
 * Mirror an MMD bone/morph name across the X axis: 左↔右, L↔R suffixes
 * (".L"/"_L", full-width Ｌ/Ｒ), and the English left↔right words.
 * Names without a side marker are returned unchanged.
 */
std::string mirror_mmd_name(const std::string &name);

struct VMDNameResolution {
  /** Actual target name that was matched; empty when not found. */
  std::string target_name;
  /** Index into the caller's target name list; -1 when not found. */
  int target_index = -1;
  /** True when matched by the exact (unmodified) name. */
  bool exact = false;
  /** Adaptation path used, e.g. "alias", "normalized", "d_bone"; empty for exact. */
  std::string via;
};

/**
 * Resolve a VMD bone name against a target Armature's bone list.
 *
 * Resolution order:
 *  1. exact name
 *  2. curated alias table (Japanese variants, English names)
 *  3. normalized name comparison
 *  4. normalized alias comparison
 *  5. MMD D-bone fallback: a trailing 'D'/'Ｄ' is stripped and the base name
 *     is resolved the same way (older motions key 左ひじD while many models
 *     only carry 左ひじ).
 */
VMDNameResolution resolve_bone_name(const std::string &vmd_name,
                                    const std::unordered_map<std::string, int> &target_exact,
                                    const std::unordered_map<std::string, int> &target_normalized);

/**
 * Resolve a VMD morph name against a target controller's Shape Key list.
 * Same order as #resolve_bone_name, but with the (conservative) morph alias
 * table and without the D-bone fallback: a wrong morph mapping is worse than
 * a missing one, so morphs rely mostly on normalization.
 */
VMDNameResolution resolve_morph_name(const std::string &vmd_name,
                                     const std::unordered_map<std::string, int> &target_exact,
                                     const std::unordered_map<std::string, int> &target_normalized);

}  // namespace blender::io::vmd
