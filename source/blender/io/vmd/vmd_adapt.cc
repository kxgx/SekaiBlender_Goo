/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup io_vmd
 */

#include "vmd_adapt.hh"

#include "BLI_string_utf8.hh"

#include <array>
#include <initializer_list>
#include <string_view>

namespace blender::io::vmd {
namespace {

/* -------------------------------------------------------------------- */
/* Unicode helpers. */

constexpr uint32_t cp_ideographic_space = 0x3000;
constexpr uint32_t cp_fullwidth_exclam = 0xFF01;
constexpr uint32_t cp_fullwidth_tilde = 0xFF5E;
constexpr uint32_t cp_halfwidth_kana_first = 0xFF61;
constexpr uint32_t cp_halfwidth_kana_last = 0xFF9F;
constexpr uint32_t cp_katakana_first = 0x30A1;
constexpr uint32_t cp_katakana_last = 0x30F6;

/* Half-width katakana to full-width katakana. Sourced from the Unicode
 * half-width/full-width folding table (excludes punctuation and the
 * combining dakuten/handakuten marks, which are dropped). */
constexpr uint32_t halfwidth_kana_to_katakana(const uint32_t cp)
{
  switch (cp) {
    case 0xFF67: return 0x30A1; /* ｧ ァ */
    case 0xFF68: return 0x30A3; /* ｨ ィ */
    case 0xFF69: return 0x30A5; /* ｩ ゥ */
    case 0xFF6A: return 0x30A7; /* ｪ ェ */
    case 0xFF6B: return 0x30A9; /* ｫ ォ */
    case 0xFF6C: return 0x30E3; /* ｬ ャ */
    case 0xFF6D: return 0x30E5; /* ｭ ュ */
    case 0xFF6E: return 0x30E7; /* ｮ ョ */
    case 0xFF6F: return 0x30C3; /* ｯ ッ */
    case 0xFF70: return 0x30FC; /* ｰ ー */
    case 0xFF71: return 0x30A2; /* ｱ ア */
    case 0xFF72: return 0x30A4; /* ｲ イ */
    case 0xFF73: return 0x30A6; /* ｳ ウ */
    case 0xFF74: return 0x30A8; /* ｴ エ */
    case 0xFF75: return 0x30AA; /* ｵ オ */
    case 0xFF76: return 0x30AB; /* ｶ カ */
    case 0xFF77: return 0x30AD; /* ｷ キ */
    case 0xFF78: return 0x30AF; /* ｸ ク */
    case 0xFF79: return 0x30B1; /* ｹ ケ */
    case 0xFF7A: return 0x30B3; /* ｺ コ */
    case 0xFF7B: return 0x30B5; /* ｻ サ */
    case 0xFF7C: return 0x30B7; /* ｼ シ */
    case 0xFF7D: return 0x30B9; /* ｽ ス */
    case 0xFF7E: return 0x30BB; /* ｾ セ */
    case 0xFF7F: return 0x30BD; /* ｿ ソ */
    case 0xFF80: return 0x30BF; /* ﾀ タ */
    case 0xFF81: return 0x30C1; /* ﾁ チ */
    case 0xFF82: return 0x30C4; /* ﾂ ツ */
    case 0xFF83: return 0x30C6; /* ﾃ テ */
    case 0xFF84: return 0x30C8; /* ﾄ ト */
    case 0xFF85: return 0x30CA; /* ﾅ ナ */
    case 0xFF86: return 0x30CB; /* ﾆ ニ */
    case 0xFF87: return 0x30CC; /* ﾇ ヌ */
    case 0xFF88: return 0x30CD; /* ﾈ ネ */
    case 0xFF89: return 0x30CE; /* ﾉ ノ */
    case 0xFF8A: return 0x30CF; /* ﾊ ハ */
    case 0xFF8B: return 0x30D2; /* ﾋ ヒ */
    case 0xFF8C: return 0x30D5; /* ﾌ フ */
    case 0xFF8D: return 0x30D8; /* ﾍ ヘ */
    case 0xFF8E: return 0x30DB; /* ﾎ ホ */
    case 0xFF8F: return 0x30DE; /* ﾏ マ */
    case 0xFF90: return 0x30DF; /* ﾐ ミ */
    case 0xFF91: return 0x30E0; /* ﾑ ム */
    case 0xFF92: return 0x30E1; /* ﾒ メ */
    case 0xFF93: return 0x30E2; /* ﾓ モ */
    case 0xFF94: return 0x30E4; /* ﾔ ヤ */
    case 0xFF95: return 0x30E6; /* ﾕ ユ */
    case 0xFF96: return 0x30E8; /* ﾖ ヨ */
    case 0xFF97: return 0x30E9; /* ﾗ ラ */
    case 0xFF98: return 0x30EA; /* ﾘ リ */
    case 0xFF99: return 0x30EB; /* ﾙ ル */
    case 0xFF9A: return 0x30EC; /* ﾚ レ */
    case 0xFF9B: return 0x30ED; /* ﾛ ロ */
    case 0xFF9C: return 0x30EF; /* ﾜ ワ */
    case 0xFF9D: return 0x30F3; /* ﾝ ン */
    default: return 0;
  }
}

/* Small kana to normal kana (hiragana). */
constexpr uint32_t small_kana_to_normal(const uint32_t cp)
{
  switch (cp) {
    case 0x3041: return 0x3042; /* ぁ あ */
    case 0x3043: return 0x3044; /* ぃ い */
    case 0x3045: return 0x3046; /* ぅ う */
    case 0x3047: return 0x3048; /* ぇ え */
    case 0x3049: return 0x304A; /* ぉ お */
    case 0x3063: return 0x3064; /* っ つ */
    case 0x3083: return 0x3084; /* ゃ や */
    case 0x3085: return 0x3086; /* ゅ ゆ */
    case 0x3087: return 0x3088; /* ょ よ */
    case 0x308E: return 0x308F; /* ゎ わ */
    case 0x3095: return 0x304B; /* ゕ か */
    case 0x3096: return 0x3051; /* ゖ け */
    default: return 0;
  }
}

/* -------------------------------------------------------------------- */
/* Alias tables.
 *
 * Each entry lists one canonical name and every known synonym. Resolution
 * expands in both directions: any synonym of the VMD name is looked up in
 * the target list. Japanese variants cover common model authoring
 * differences; English names cover models normalized during MMD↔engine
 * exchange workflows. */

struct AliasEntry {
  std::string_view canonical;
  std::array<std::string_view, 8> aliases;
  int alias_count;
};

struct AliasTable {
  const AliasEntry *entries;
  size_t count;
};

AliasEntry make_alias_entry(const std::string_view canonical,
                            std::initializer_list<const char *> aliases)
{
  AliasEntry entry = {};
  entry.canonical = canonical;
  int i = 0;
  for (const char *alias : aliases) {
    entry.aliases[size_t(i++)] = alias;
  }
  entry.alias_count = i;
  return entry;
}

const AliasTable &bone_alias_table()
{
  static const std::vector<AliasEntry> table = []() {
    std::vector<AliasEntry> t;
    auto add = [&](const char *canonical, std::initializer_list<const char *> aliases) {
      t.push_back(make_alias_entry(canonical, aliases));
    };
    add("全ての親", {"all parents", "root"});
    add("センター", {"センタ", "center"});
    add("グルーブ", {"groove"});
    add("腰", {"waist"});
    add("上半身", {"upper body", "upper_body"});
    add("上半身2", {"上半身２", "upper body 2", "upper_body_2"});
    add("下半身", {"lower body", "lower_body"});
    add("胸", {"chest"});
    add("首", {"ネック", "neck"});
    add("首下", {"首元", "neck_lower"});
    add("頭", {"ヘッド", "head"});
    add("頭先", {"head_tip"});
    add("左肩", {"肩.L", "shoulder_L"});
    add("右肩", {"肩.R", "shoulder_R"});
    add("左腕", {"腕.L", "arm_L", "upper_arm_L"});
    add("右腕", {"腕.R", "arm_R", "upper_arm_R"});
    add("左ひじ", {"肘.L", "ひじ.L", "elbow_L"});
    add("右ひじ", {"肘.R", "ひじ.R", "elbow_R"});
    add("左手首", {"手首.L", "wrist_L"});
    add("右手首", {"手首.R", "wrist_R"});
    add("左腕捩", {"左腕捩り", "左腕捩れ", "左腕捻り", "左腕ねじり", "左腕ひねり", "左腕捻", "腕捩.L", "arm_twist_L"});
    add("右腕捩", {"右腕捩り", "右腕捩れ", "右腕捻り", "右腕ねじり", "右腕ひねり", "右腕捻", "腕捩.R", "arm_twist_R"});
    add("左手捩", {"左手捩り", "左手捻り", "左手ねじり", "左手捻", "手捩.L", "wrist_twist_L"});
    add("右手捩", {"右手捩り", "右手捻り", "右手ねじり", "右手捻", "手捩.R", "wrist_twist_R"});
    add("左足", {"足.L", "leg_L", "thigh_L"});
    add("右足", {"足.R", "leg_R", "thigh_R"});
    add("左ひざ", {"膝.L", "ひざ.L", "knee_L"});
    add("右ひざ", {"膝.R", "ひざ.R", "knee_R"});
    add("左足首", {"足首.L", "ankle_L"});
    add("右足首", {"足首.R", "ankle_R"});
    add("左つま先", {"左つまさき", "つま先.L", "toe_L"});
    add("右つま先", {"右つまさき", "つま先.R", "toe_R"});
    add("左肩P", {"肩P.L", "shoulderP_L"});
    add("右肩P", {"肩P.R", "shoulderP_R"});
    add("左肩C", {"肩C.L", "shoulderC_L"});
    add("右肩C", {"肩C.R", "shoulderC_R"});
    add("左足IK親", {"足IK親.L", "足ＩＫ親.L"});
    add("右足IK親", {"足IK親.R", "足ＩＫ親.R"});
    add("左手先", {"手先.L", "hand_tip_L"});
    add("右手先", {"手先.R", "hand_tip_R"});
    add("左足補", {"足補.L"});
    add("右足補", {"足補.R"});
    add("左ひじ+", {"ひじ+.L", "肘+.L"});
    add("右ひじ+", {"ひじ+.R", "肘+.R"});
    add("左足先EX", {"足先EX.L"});
    add("右足先EX", {"足先EX.R"});
    add("左ひざ補助", {"ひざ補助.L"});
    add("右ひざ補助", {"ひざ補助.R"});
    add("左足D", {"足D.L"});
    add("右足D", {"足D.R"});
    add("左ひざD", {"ひざD.L", "膝D.L"});
    add("右ひざD", {"ひざD.R", "膝D.R"});
    add("左足首D", {"足首D.L"});
    add("右足首D", {"足首D.R"});
    add("左ひじD", {"ひじD.L", "肘D.L"});
    add("右ひじD", {"ひじD.R", "肘D.R"});
    add("左腕捩1", {"腕捩1.L"});
    add("右腕捩1", {"腕捩1.R"});
    add("左腕捩2", {"腕捩2.L"});
    add("右腕捩2", {"腕捩2.R"});
    add("左腕捩3", {"腕捩3.L"});
    add("右腕捩3", {"腕捩3.R"});
    add("左手捩1", {"手捩1.L"});
    add("右手捩1", {"手捩1.R"});
    add("左手捩2", {"手捩2.L"});
    add("右手捩2", {"手捩2.R"});
    add("左手捩3", {"手捩3.L"});
    add("右手捩3", {"手捩3.R"});
    add("左足IK", {"足ＩＫ.L", "leg_IK_L", "leg_ik_L"});
    add("右足IK", {"足ＩＫ.R", "leg_IK_R", "leg_ik_R"});
    add("左つま先IK", {"左つまさきIK", "つま先ＩＫ.L", "toe_IK_L"});
    add("右つま先IK", {"右つまさきIK", "つま先ＩＫ.R", "toe_IK_R"});
    add("左目", {"目.L", "eye_L"});
    add("右目", {"目.R", "eye_R"});
    add("両目", {"eyes"});
    add("左親指０", {"thumb0_L"});
    add("左親指１", {"thumb1_L"});
    add("左親指２", {"thumb2_L"});
    add("右親指０", {"thumb0_R"});
    add("右親指１", {"thumb1_R"});
    add("右親指２", {"thumb2_R"});
    add("左人指１", {"fore1_L"});
    add("左人指２", {"fore2_L"});
    add("左人指３", {"fore3_L"});
    add("右人指１", {"fore1_R"});
    add("右人指２", {"fore2_R"});
    add("右人指３", {"fore3_R"});
    add("左中指１", {"middle1_L"});
    add("左中指２", {"middle2_L"});
    add("左中指３", {"middle3_L"});
    add("右中指１", {"middle1_R"});
    add("右中指２", {"middle2_R"});
    add("右中指３", {"middle3_R"});
    add("左薬指１", {"third1_L"});
    add("左薬指２", {"third2_L"});
    add("左薬指３", {"third3_L"});
    add("右薬指１", {"third1_R"});
    add("右薬指２", {"third2_R"});
    add("右薬指３", {"third3_R"});
    add("左小指１", {"little1_L"});
    add("左小指２", {"little2_L"});
    add("左小指３", {"little3_L"});
    add("右小指１", {"little1_R"});
    add("右小指２", {"little2_R"});
    add("右小指３", {"little3_R"});
    return t;
  }();
  static const AliasTable result = {table.data(), table.size()};
  return result;
}

/* Conservative morph synonyms: only widely used renamings of the same morph.
 * Wrong morph matches are worse than missing ones, so this table stays small;
 * normalization (kana unify, full-width digits) handles the rest. */
const AliasTable &morph_alias_table()
{
  static const std::vector<AliasEntry> table = []() {
    std::vector<AliasEntry> t;
    auto add = [&](const char *canonical, std::initializer_list<const char *> aliases) {
      t.push_back(make_alias_entry(canonical, aliases));
    };
    add("まばたき", {"瞬き", "blink"});
    add("ウィンク", {"ウインク", "wink"});
    add("ウィンク右", {"ウインク右", "wink_R"});
    add("ウィンク左", {"ウインク左", "wink_L"});
    add("ウィンク２", {"ウインク２", "ウインク2", "wink 2"});
    add("笑い", {"smile"});
    add("あ", {"a"});
    add("い", {"i"});
    add("う", {"u"});
    add("え", {"e"});
    add("お", {"o"});
    add("∧", {"^"});
    add("はぅ", {"はう"});
    add("ん", {"n"});
    return t;
  }();
  static const AliasTable result = {table.data(), table.size()};
  return result;
}

/* Expand a name into all synonyms of its alias group. Group membership is
 * decided by exact OR normalized comparison, so full-width/half-width
 * variants of the same key (e.g. 右足ＩＫ vs 右足IK) resolve to one group. */
std::vector<std::string_view> synonyms_of(const std::string &name, const AliasTable &table)
{
  const std::string normalized_name = normalize_mmd_name(name);
  auto member_matches = [&](const std::string_view member) {
    return member == name ||
           (!normalized_name.empty() && normalize_mmd_name(member) == normalized_name);
  };
  for (size_t i = 0; i < table.count; i++) {
    const AliasEntry &entry = table.entries[i];
    if (member_matches(entry.canonical)) {
      std::vector<std::string_view> group;
      group.reserve(size_t(entry.alias_count));
      for (int a = 0; a < entry.alias_count; a++) {
        group.push_back(entry.aliases[a]);
      }
      return group;
    }
    for (int a = 0; a < entry.alias_count; a++) {
      if (member_matches(entry.aliases[a])) {
        std::vector<std::string_view> group;
        group.reserve(size_t(entry.alias_count) + 1);
        group.push_back(entry.canonical);
        for (int b = 0; b < entry.alias_count; b++) {
          if (b != a) {
            group.push_back(entry.aliases[b]);
          }
        }
        return group;
      }
    }
  }
  return {};
}

/* Strip a trailing MMD D-bone marker: ASCII 'D' or full-width 'Ｄ'.
 * Restricting to upper case avoids false positives on English names such
 * as "hand" (lowercase final 'd'). */
bool d_bone_base(const std::string &name, std::string &r_base)
{
  if (name.size() < 2) {
    return false;
  }
  if (name.size() >= 3 && std::string_view(name).substr(name.size() - 3) == "Ｄ") {
    r_base = name.substr(0, name.size() - 3);
    return true;
  }
  if (name.back() == 'D') {
    r_base = name.substr(0, name.size() - 1);
    return true;
  }
  return false;
}

bool lookup_exact(const std::string &name,
                  const std::unordered_map<std::string, int> &target_exact,
                  int &r_index)
{
  const auto it = target_exact.find(name);
  if (it == target_exact.end()) {
    return false;
  }
  r_index = it->second;
  return true;
}

bool lookup_normalized(const std::string &name,
                       const std::unordered_map<std::string, int> &target_normalized,
                       int &r_index)
{
  const auto it = target_normalized.find(normalize_mmd_name(name));
  if (it == target_normalized.end()) {
    return false;
  }
  r_index = it->second;
  return true;
}

VMDNameResolution resolve_common(const std::string &vmd_name,
                                 const std::unordered_map<std::string, int> &target_exact,
                                 const std::unordered_map<std::string, int> &target_normalized,
                                 const AliasTable &alias_table,
                                 const bool use_d_bone)
{
  VMDNameResolution result;
  int index = -1;

  /* 1. Exact. */
  if (lookup_exact(vmd_name, target_exact, index)) {
    result.target_index = index;
    result.exact = true;
    return result;
  }

  /* 2. Alias synonyms against exact names. */
  for (const std::string_view synonym : synonyms_of(vmd_name, alias_table)) {
    if (lookup_exact(std::string(synonym), target_exact, index)) {
      result.target_index = index;
      result.via = "alias";
      return result;
    }
  }

  /* 3. Normalized comparison. */
  if (lookup_normalized(vmd_name, target_normalized, index)) {
    result.target_index = index;
    result.via = "normalized";
    return result;
  }

  /* 4. Normalized synonyms. */
  for (const std::string_view synonym : synonyms_of(vmd_name, alias_table)) {
    if (lookup_normalized(std::string(synonym), target_normalized, index)) {
      result.target_index = index;
      result.via = "alias";
      return result;
    }
  }

  /* 5. MMD D-bone fallback. */
  if (use_d_bone) {
    std::string base;
    if (d_bone_base(vmd_name, base)) {
      if (lookup_exact(base, target_exact, index)) {
        result.target_index = index;
        result.via = "d_bone";
        return result;
      }
      for (const std::string_view synonym : synonyms_of(base, alias_table)) {
        if (lookup_exact(std::string(synonym), target_exact, index)) {
          result.target_index = index;
          result.via = "d_bone";
          return result;
        }
      }
      if (lookup_normalized(base, target_normalized, index)) {
        result.target_index = index;
        result.via = "d_bone";
        return result;
      }
      for (const std::string_view synonym : synonyms_of(base, alias_table)) {
        if (lookup_normalized(std::string(synonym), target_normalized, index)) {
          result.target_index = index;
          result.via = "d_bone";
          return result;
        }
      }
    }
  }

  return result;
}

}  // namespace

std::string normalize_mmd_name(const std::string_view name)
{
  std::vector<uint32_t> codepoints;
  codepoints.reserve(name.size());

  const char *p = name.data();
  const size_t len = name.size();
  size_t offset = 0;
  while (offset < len) {
    size_t consumed = 0;
    const uint32_t cp = BLI_str_utf8_as_unicode_step_or_error(p + offset, len - offset, &consumed);
    if (cp == BLI_UTF8_ERR || consumed == 0) {
      break;
    }
    offset += consumed;

    uint32_t folded = cp;
    if (folded == cp_ideographic_space) {
      folded = ' ';
    }
    else if (folded >= cp_fullwidth_exclam && folded <= cp_fullwidth_tilde) {
      folded -= 0xFEE0;
    }
    else if (folded >= cp_halfwidth_kana_first && folded <= cp_halfwidth_kana_last) {
      const uint32_t mapped = halfwidth_kana_to_katakana(folded);
      if (mapped != 0) {
        folded = mapped;
      }
      else {
        /* ｡｢｣､･ and the combining dakuten marks: drop. */
        continue;
      }
    }

    if (folded >= cp_katakana_first && folded <= cp_katakana_last) {
      folded -= 0x60; /* Katakana → hiragana. */
    }
    else if (folded == 0x30F7) {
      folded = 0x3094; /* ヷ → ゔ */
    }

    const uint32_t small = small_kana_to_normal(folded);
    if (small != 0) {
      folded = small;
    }

    if (folded >= 'A' && folded <= 'Z') {
      folded += 32;
    }

    codepoints.push_back(folded);
  }

  /* Trim ASCII spaces (ideographic spaces were folded above). */
  size_t begin = 0;
  while (begin < codepoints.size() && codepoints[begin] == ' ') {
    begin++;
  }
  size_t end = codepoints.size();
  while (end > begin && codepoints[end - 1] == ' ') {
    end--;
  }

  std::string result;
  result.reserve(end - begin);
  for (size_t i = begin; i < end; i++) {
    char buffer[8] = {};
    const size_t written = BLI_str_utf8_from_unicode(codepoints[i], buffer, sizeof(buffer));
    if (written > 0 && written <= sizeof(buffer)) {
      result.append(buffer, written);
    }
  }
  return result;
}

std::string mirror_mmd_name(const std::string &name)
{
  std::string result = name;

  /* 左↔右 (each occurrence; MMD names carry a single side marker).
   * 3-byte UTF-8: 左 = E5 B7 A6, 右 = E5 8F B3。
   * 用 U+FFFD 占位避免两次循环相互抵消。 */
  bool swapped_jp = false;
  const std::string left_jp = "\xE5\xB7\xA6";
  const std::string right_jp = "\xE5\x8F\xB3";
  const std::string placeholder = "\xEF\xBF\xBD";
  auto replace_all = [&](const std::string &from, const std::string &to) {
    size_t pos = 0;
    while ((pos = result.find(from, pos)) != std::string::npos) {
      result.replace(pos, from.size(), to);
      swapped_jp = true;
      pos += to.size();
    }
  };
  replace_all(left_jp, placeholder);
  replace_all(right_jp, left_jp);
  replace_all(placeholder, right_jp);
  if (swapped_jp) {
    return result;
  }

  /* English left/right words (case-insensitive, whole words only). */
  auto replace_word = [&](const std::string &from, const std::string &to) -> bool {
    const std::string lower = [&]() {
      std::string s = result;
      for (char &c : s) {
        if (c >= 'A' && c <= 'Z') {
          c = char(c + 32);
        }
      }
      return s;
    }();
    size_t p = lower.find(from);
    if (p == std::string::npos) {
      return false;
    }
    /* Whole-word check: boundaries are start/end or non-alphanumeric. */
    auto boundary = [&](size_t i) {
      if (i == 0 || i == result.size()) {
        return true;
      }
      const char c = result[i];
      return !((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9'));
    };
    while (p != std::string::npos) {
      if (boundary(p) && boundary(p + from.size())) {
        result.replace(p, from.size(), to);
        return true;
      }
      p = lower.find(from, p + 1);
    }
    return false;
  };
  if (replace_word("left", "right") || replace_word("right", "left")) {
    return result;
  }

  /* L/R suffixes: ".L"↔".R" and "_L"↔"_R" (and lower case), full-width Ｌ/Ｒ. */
  const std::array<std::string, 8> from_suffixes = {".L", ".R", "_L", "_R",
                                                    ".l", ".r", "_l", "_r"};
  const std::array<std::string, 8> to_suffixes = {".R", ".L", "_R", "_L",
                                                  ".r", ".l", "_r", "_l"};
  for (size_t i = 0; i < from_suffixes.size(); i++) {
    if (result.size() >= from_suffixes[i].size() &&
        result.compare(result.size() - from_suffixes[i].size(),
                       from_suffixes[i].size(),
                       from_suffixes[i]) == 0)
    {
      result.replace(result.size() - from_suffixes[i].size(),
                     from_suffixes[i].size(),
                     to_suffixes[i]);
      return result;
    }
  }

  /* Full-width Ｌ (EF BC AC) / Ｒ (EF BC B2). */
  const std::string fw_l = "\xEF\xBC\xAC";
  const std::string fw_r = "\xEF\xBC\xB2";
  if (result.size() >= 3 && result.compare(result.size() - 3, 3, fw_l) == 0) {
    result.replace(result.size() - 3, 3, fw_r);
  }
  else if (result.size() >= 3 && result.compare(result.size() - 3, 3, fw_r) == 0) {
    result.replace(result.size() - 3, 3, fw_l);
  }

  return result;
}

VMDNameResolution resolve_bone_name(const std::string &vmd_name,
                                    const std::unordered_map<std::string, int> &target_exact,
                                    const std::unordered_map<std::string, int> &target_normalized)
{
  return resolve_common(vmd_name, target_exact, target_normalized, bone_alias_table(), true);
}

VMDNameResolution resolve_morph_name(const std::string &vmd_name,
                                     const std::unordered_map<std::string, int> &target_exact,
                                     const std::unordered_map<std::string, int> &target_normalized)
{
  return resolve_common(vmd_name, target_exact, target_normalized, morph_alias_table(), false);
}

}  // namespace blender::io::vmd
