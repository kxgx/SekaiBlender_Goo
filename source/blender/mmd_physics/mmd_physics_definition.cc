/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "mmd_physics_definition.hh"

#include "BKE_armature.hh"
#include "BKE_collection.hh"
#include "BKE_constraint.h"
#include "BKE_idprop.hh"
#include "BKE_lib_id.hh"
#include "BKE_main.hh"
#include "BKE_mesh.h"
#include "BKE_object.hh"
#include "BKE_report.hh"
#include "BKE_rigidbody.h"
#include "BKE_scene.hh"
#include "BKE_pose.hh"
#include "BKE_anim_data.hh"
#include "BKE_fcurve.hh"

#include "DEG_depsgraph.hh"
#include "DEG_depsgraph_build.hh"
#include "DEG_depsgraph_query.hh"

#include "ANIM_action.hh"
#include "ANIM_fcurve.hh"
#include "ANIM_keyframing.hh"

#include "BLI_fileops.hh"
#include "BLI_math_matrix_c.hh"
#include "BLI_math_rotation_c.hh"
#include "BLI_math_vector.hh"
#include "BLI_math_vector_c.hh"
#include "BLI_string.hh"
#include "BLI_vector.hh"
#include "MEM_guardedalloc.h"
#include "DNA_anim_types.h"
#include "DNA_collection_types.h"
#include "DNA_constraint_types.h"
#include "DNA_mesh_types.h"
#include "DNA_object_types.h"
#include "DNA_rigidbody_types.h"
#include "DNA_scene_types.h"

#include "../io/pmx/intern/pmx_types.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>
#include <system_error>
#include <utility>

#include "BLI_string_ref.hh"

namespace blender::mmd_physics {
namespace {

bool finite_vec3(const float value[3])
{
  return std::isfinite(value[0]) && std::isfinite(value[1]) && std::isfinite(value[2]);
}

void add_error(MMDPhysicsBuildResult &result,
               const char *section,
               const int index,
               const char *field,
               const std::string &detail)
{
  std::ostringstream message;
  message << section << "[" << index << "]." << field << ": " << detail;
  result.errors.push_back(message.str());
  result.definition.validation.valid = false;
  result.definition.validation.total_errors++;
}

bool check_finite_vec3(MMDPhysicsBuildResult &result,
                        const char *section,
                        const int index,
                        const char *field,
                        const float value[3])
{
  const bool valid = finite_vec3(value);
  if (!valid) {
    add_error(result, section, index, field, "contains NaN or Inf");
  }
  return valid;
}

void transform_position(std::array<float, 3> &dst, const float src[3], const float scale)
{
  /* PMX -> Blender uses the same axis conversion as the importer.
   * mmd_tools: loc = Vector(pmx.pos).xzy * scale
   * PMX (X, Y, Z) -> Blender (X, Z, Y) * scale */
  dst[0] = src[0] * scale;
  dst[1] = src[2] * scale;
  dst[2] = src[1] * scale;
}

void transform_rotation(std::array<float, 3> &dst, const float src[3])
{
  /* mmd_tools PMX importer: rot = Vector(pmx.rot).xzy * -1
   * PMX (X, Y, Z) -> Blender (-X, -Z, -Y)
   * This is NOT a standard coordinate-system rotation; it is mmd_tools'
   * convention. We replicate it to align Bullet world body transforms with
   * MikuMikuPhysics reference. */
  dst[0] = -src[0];
  dst[1] = -src[2];
  dst[2] = -src[1];
}

void transform_vec3_yz_swap(std::array<float, 3> &dst, const float src[3])
{
  /* mmd_tools: vec.xzy (swap Y/Z, no negate, no scale).
   * Used for joint spring_linear/angular. */
  dst[0] = src[0];
  dst[1] = src[2];
  dst[2] = src[1];
}

void transform_vec3_yz_swap_negate(std::array<float, 3> &dst, const float src[3])
{
  /* mmd_tools: vec.xzy * -1 (swap Y/Z + negate).
   * Used for joint rotation limits. */
  dst[0] = -src[0];
  dst[1] = -src[2];
  dst[2] = -src[1];
}

bool positive_shape_value(const float value)
{
  return std::isfinite(value) && value > 0.0f;
}

}  // namespace

namespace {

constexpr char kDefinitionProperty[] = "mmd_physics_definition";
constexpr int kMaxPersistedItems = 100000;

IDProperty *new_int_array(const char *name, const int *values, const int length)
{
  blender::Vector<int32_t> copied(length);
  for (int i = 0; i < length; i++) {
    copied[i] = int32_t(values[i]);
  }
  return blender::bke::idprop::create(name, copied.as_span()).release();
}

IDProperty *new_float_array(const char *name, const std::array<float, 3> &values)
{
  return blender::bke::idprop::create(name, blender::Span<float>(values.data(), 3)).release();
}

void add_property(IDProperty *group, IDProperty *property)
{
  if (!property || !IDP_AddToGroup(group, property)) {
    if (property) {
      IDP_FreeProperty(property);
    }
  }
}

void add_int(IDProperty *group, const char *name, const int value)
{
  add_property(group, IDP_NewInt(value, name));
}

void add_float(IDProperty *group, const char *name, const float value)
{
  add_property(group, blender::bke::idprop::create(name, value).release());
}

void add_bool(IDProperty *group, const char *name, const bool value)
{
  add_property(group, blender::bke::idprop::create_bool(name, value).release());
}

void add_string(IDProperty *group, const char *name, const std::string &value)
{
  add_property(group, IDP_NewString(value.c_str(), name));
}

void add_vec3(IDProperty *group, const char *name, const std::array<float, 3> &values)
{
  add_property(group, new_float_array(name, values));
}

void report_error(ReportList *reports, const std::string &message)
{
  if (reports) {
    BKE_report(reports, RPT_WARNING, message.c_str());
  }
}

IDProperty *get_group(IDProperty *group, const char *name)
{
  return IDP_GetPropertyTypeFromGroup(group, name, IDP_GROUP);
}

IDProperty *get_array(IDProperty *group, const char *name, const int length, const char subtype)
{
  IDProperty *property = IDP_GetPropertyTypeFromGroup(group, name, IDP_ARRAY);
  return property && property->len == length && property->subtype == subtype ? property : nullptr;
}

bool read_int(IDProperty *group, const char *name, int &value)
{
  IDProperty *property = IDP_GetPropertyTypeFromGroup(group, name, IDP_INT);
  if (!property) {
    return false;
  }
  value = IDP_int_get(property);
  return true;
}

bool read_float(IDProperty *group, const char *name, float &value)
{
  IDProperty *property = IDP_GetPropertyTypeFromGroup(group, name, IDP_FLOAT);
  if (!property || !std::isfinite(IDP_float_get(property))) {
    return false;
  }
  value = IDP_float_get(property);
  return true;
}

bool read_bool(IDProperty *group, const char *name, bool &value)
{
  IDProperty *property = IDP_GetPropertyTypeFromGroup(group, name, IDP_BOOLEAN);
  if (!property) {
    return false;
  }
  value = IDP_bool_get(property);
  return true;
}

bool read_string(IDProperty *group, const char *name, std::string &value)
{
  IDProperty *property = IDP_GetPropertyTypeFromGroup(group, name, IDP_STRING);
  if (!property) {
    return false;
  }
  value = IDP_string_get(property);
  return true;
}

bool read_vec3(IDProperty *group, const char *name, std::array<float, 3> &value)
{
  IDProperty *property = get_array(group, name, 3, IDP_FLOAT);
  if (!property) {
    return false;
  }
  const float *values = IDP_array_float_get(property);
  if (!std::isfinite(values[0]) || !std::isfinite(values[1]) || !std::isfinite(values[2])) {
    return false;
  }
  std::copy(values, values + 3, value.begin());
  return true;
}

void append_group(IDProperty *array, IDProperty *item)
{
  IDP_ResizeIDPArray(array, array->len + 1);
  IDP_SetIndexArray(array, array->len - 1, item);
  /* IDP_SetIndexArray makes a shallow copy of the item. */
  MEM_delete(item);
}

void write_json_string(FILE *file, const std::string &value)
{
  static constexpr char hex[] = "0123456789abcdef";
  fputc('"', file);
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"': fputs("\\\"", file); break;
      case '\\': fputs("\\\\", file); break;
      case '\b': fputs("\\b", file); break;
      case '\f': fputs("\\f", file); break;
      case '\n': fputs("\\n", file); break;
      case '\r': fputs("\\r", file); break;
      case '\t': fputs("\\t", file); break;
      default:
        if (byte < 0x20) {
          const char escaped[7] = {'\\', 'u', '0', '0', hex[byte >> 4], hex[byte & 0x0f], '\0'};
          fputs(escaped, file);
        }
        else {
          fputc(byte, file);
        }
        break;
    }
  }
  fputc('"', file);
}

void write_json_float(FILE *file, const float value)
{
  if (!std::isfinite(value)) {
    fputs("null", file);
    return;
  }
  char buffer[32];
  const auto result = std::to_chars(
      buffer, buffer + sizeof(buffer), value, std::chars_format::general, 9);
  if (result.ec == std::errc()) {
    fwrite(buffer, 1, size_t(result.ptr - buffer), file);
  }
  else {
    fputs("null", file);
  }
}

void write_float3(FILE *file, const std::array<float, 3> &value)
{
  fputc('[', file);
  for (int axis = 0; axis < 3; axis++) {
    if (axis != 0) {
      fputc(',', file);
    }
    write_json_float(file, value[axis]);
  }
  fputc(']', file);
}

void write_limit_modes(FILE *file, const std::array<MMDJointAxisLimitMode, 3> &modes)
{
  fputc('[', file);
  for (int axis = 0; axis < 3; axis++) {
    if (axis != 0) {
      fputc(',', file);
    }
    write_json_string(file, modes[axis] == MMDJointAxisLimitMode::Free ? "free" : "limited");
  }
  fputc(']', file);
}

}  // namespace

MMDPhysicsBuildResult build_physics_definition(const PMXModel &model,
                                               const Vector<std::string> &bone_names,
                                               const char *model_name,
                                               const float coordinate_scale)
{
  MMDPhysicsBuildResult result;
  MMDPhysicsDefinition &definition = result.definition;
  definition.source_model_name = model_name ? std::string(model_name) : model.name_local;
  definition.source_pmx_version = model.header.version;
  definition.coordinate_scale = coordinate_scale;
  definition.source_file_size = model.file_size;
  definition.source_parse_end_offset = model.parse_end_offset;

  definition.bone_mapping.reserve(model.bones.size());
  for (int i = 0; i < int(model.bones.size()); i++) {
    const PMXBone &pmx_bone = model.bones[i];
    MMDBoneMapping mapping;
    mapping.pmx_index = i;
    mapping.pmx_name_local = pmx_bone.name_local;
    mapping.pmx_name_universal = pmx_bone.name_universal;
    if (i < bone_names.size()) {
      mapping.blender_bone_name = bone_names[i];
      mapping.resolved = !mapping.blender_bone_name.empty();
    }
    if (!mapping.resolved) {
      definition.validation.unresolved_bones++;
      add_error(result, "bones", i, "blender_bone_name", "no Blender bone mapping");
    }
    definition.bone_mapping.push_back(std::move(mapping));
  }

  definition.rigid_bodies.reserve(model.rigid_bodies.size());
  for (int i = 0; i < int(model.rigid_bodies.size()); i++) {
    const PMXRigidBody &pmx_rigid = model.rigid_bodies[i];
    MMDRigidBodyDefinition rigid;
    rigid.pmx_index = i;
    rigid.name_local = pmx_rigid.name_local;
    rigid.name_universal = pmx_rigid.name_universal;
    rigid.pmx_bone_index = pmx_rigid.bone_index;
    rigid.collision_group = pmx_rigid.collision_group;
    rigid.no_collision_group = pmx_rigid.no_collision_group;
    rigid.shape_type = pmx_rigid.shape_type;
    rigid.physics_type = pmx_rigid.physics_type;
    rigid.mass = pmx_rigid.mass;
    rigid.linear_damping = pmx_rigid.linear_damping;
    rigid.angular_damping = pmx_rigid.angular_damping;
    rigid.restitution = pmx_rigid.restitution;
    rigid.friction = pmx_rigid.friction;
    transform_position(rigid.position, pmx_rigid.pos, coordinate_scale);
    transform_rotation(rigid.rotation, pmx_rigid.rot);
    rigid.shape_size[0] = pmx_rigid.shape_size[0] * coordinate_scale;
    rigid.shape_size[1] = pmx_rigid.shape_size[rigid.shape_type == 1 ? 2 : 1] * coordinate_scale;
    rigid.shape_size[2] = pmx_rigid.shape_size[rigid.shape_type == 1 ? 1 : 2] * coordinate_scale;

    bool rigid_valid = true;
    if (pmx_rigid.bone_index == -1) {
      rigid.bone_resolved = true;
    }
    else if (pmx_rigid.bone_index >= 0 &&
             pmx_rigid.bone_index < int(definition.bone_mapping.size()) &&
             definition.bone_mapping[pmx_rigid.bone_index].resolved)
    {
      rigid.blender_bone_name = definition.bone_mapping[pmx_rigid.bone_index].blender_bone_name;
      rigid.bone_resolved = true;
    }
    else {
      add_error(result, "rigid_bodies", i, "bone_index", "does not resolve to a Blender bone");
      rigid_valid = false;
    }

    if (rigid.collision_group > 15) {
      add_error(result, "rigid_bodies", i, "collision_group", "must be in range 0..15");
      rigid_valid = false;
    }
    if (rigid.shape_type > 2) {
      add_error(result, "rigid_bodies", i, "shape_type", "unsupported shape type");
      rigid_valid = false;
    }
    else {
      const int required_axes = rigid.shape_type == 0 ? 1 : 2;
      const int axis_count = rigid.shape_type == 1 ? 3 : required_axes;
      for (int axis = 0; axis < axis_count; axis++) {
        if (!positive_shape_value(rigid.shape_size[axis])) {
          add_error(result,
                    "rigid_bodies",
                    i,
                    "shape_size",
                    rigid.shape_type == 1 ? "box dimensions must be positive and finite" :
                                            "required dimensions must be positive and finite");
          rigid_valid = false;
          break;
        }
      }
    }
    if (!check_finite_vec3(result, "rigid_bodies", i, "pos", pmx_rigid.pos)) {
      rigid_valid = false;
    }
    if (!check_finite_vec3(result, "rigid_bodies", i, "rot", pmx_rigid.rot)) {
      rigid_valid = false;
    }
    if (!std::isfinite(rigid.mass) || !std::isfinite(rigid.linear_damping) ||
        !std::isfinite(rigid.angular_damping) || !std::isfinite(rigid.restitution) ||
        !std::isfinite(rigid.friction))
    {
      add_error(result, "rigid_bodies", i, "physical_parameters", "contains NaN or Inf");
      rigid_valid = false;
    }
    if (pmx_rigid.physics_type > 2) {
      add_error(result, "rigid_bodies", i, "physics_type", "unsupported physics type");
      rigid_valid = false;
    }
    if (!rigid_valid) {
      definition.validation.invalid_rigid_bodies++;
    }
    definition.rigid_bodies.push_back(std::move(rigid));
  }

  definition.joints.reserve(model.joints.size());
  for (int i = 0; i < int(model.joints.size()); i++) {
    const PMXJoint &pmx_joint = model.joints[i];
    MMDJointDefinition joint;
    joint.pmx_index = i;
    joint.name_local = pmx_joint.name_local;
    joint.name_universal = pmx_joint.name_universal;
    joint.type = pmx_joint.type;
    joint.rigid_a_index = pmx_joint.rigid_a_index;
    joint.rigid_b_index = pmx_joint.rigid_b_index;
    transform_position(joint.position, pmx_joint.pos, coordinate_scale);
    transform_rotation(joint.rotation, pmx_joint.rot);
    /* mmd_tools: maximum_location = pmx.max.xzy * scale,
     *            minimum_location = pmx.min.xzy * scale.
     * Both min and max are Y/Z swapped and scaled (no min/max exchange). */
    transform_position(joint.translation_min, pmx_joint.translation_limit_min, coordinate_scale);
    transform_position(joint.translation_max, pmx_joint.translation_limit_max, coordinate_scale);
    /* mmd_tools: maximum_rotation = pmx.min_rotation.xzy * -1,
     *            minimum_rotation = pmx.max_rotation.xzy * -1.
     * Note the min/max exchange! After exchange, our rotation_min holds
     * pmx.max_rotation transformed, and rotation_max holds pmx.min_rotation
     * transformed. This mirrors mmd_tools' createJoint argument order. */
    transform_vec3_yz_swap_negate(joint.rotation_min, pmx_joint.rotation_limit_max);
    transform_vec3_yz_swap_negate(joint.rotation_max, pmx_joint.rotation_limit_min);
    /* mmd_tools: spring_linear = pmx.spring_constant.xzy,
     *            spring_angular = pmx.spring_rotation_constant.xzy.
     * Y/Z swap only, no negate, no scale. */
    transform_vec3_yz_swap(joint.spring_translation, pmx_joint.spring_translation);
    transform_vec3_yz_swap(joint.spring_rotation, pmx_joint.spring_rotation);

    bool joint_valid = true;
    if (joint.type != 0) {
      add_error(result, "joints", i, "type", "unsupported joint type");
      joint_valid = false;
    }
    if (joint.rigid_a_index < 0 || joint.rigid_a_index >= int(model.rigid_bodies.size())) {
      add_error(result, "joints", i, "rigid_a_index", "out of range");
      joint_valid = false;
    }
    if (joint.rigid_b_index < 0 || joint.rigid_b_index >= int(model.rigid_bodies.size())) {
      add_error(result, "joints", i, "rigid_b_index", "out of range");
      joint_valid = false;
    }
    if (!check_finite_vec3(result, "joints", i, "pos", pmx_joint.pos)) {
      joint_valid = false;
    }
    if (!check_finite_vec3(result, "joints", i, "rot", pmx_joint.rot)) {
      joint_valid = false;
    }
    if (!check_finite_vec3(
            result, "joints", i, "translation_limit_min", pmx_joint.translation_limit_min))
    {
      joint_valid = false;
    }
    if (!check_finite_vec3(
            result, "joints", i, "translation_limit_max", pmx_joint.translation_limit_max))
    {
      joint_valid = false;
    }
    if (!check_finite_vec3(result, "joints", i, "rotation_limit_min", pmx_joint.rotation_limit_min))
    {
      joint_valid = false;
    }
    if (!check_finite_vec3(result, "joints", i, "rotation_limit_max", pmx_joint.rotation_limit_max))
    {
      joint_valid = false;
    }
    if (!check_finite_vec3(result, "joints", i, "spring_translation", pmx_joint.spring_translation))
    {
      joint_valid = false;
    }
    if (!check_finite_vec3(result, "joints", i, "spring_rotation", pmx_joint.spring_rotation))
    {
      joint_valid = false;
    }
    for (int axis = 0; axis < 3; axis++) {
      if (joint.translation_min[axis] > joint.translation_max[axis]) {
        add_error(result, "joints", i, "translation_limits", "minimum exceeds maximum");
        joint_valid = false;
      }
      /* PMX/MMD uses an inverted angular interval to represent a free axis. */
      joint.rotation_limit_mode[axis] = joint.rotation_min[axis] > joint.rotation_max[axis] ?
                                            MMDJointAxisLimitMode::Free :
                                            MMDJointAxisLimitMode::Limited;
    }
    if (!joint_valid) {
      definition.validation.invalid_joints++;
    }
    definition.joints.push_back(std::move(joint));
  }

  return result;
}

bool serialize_physics_definition(Collection &model_root,
                                  const MMDPhysicsDefinition &definition,
                                  ReportList *reports)
{
  if (!definition.validation.valid) {
    report_error(reports, "MMD physics definition: refusing to persist invalid definition");
    return false;
  }
  if (definition.bone_mapping.size() > kMaxPersistedItems ||
      definition.rigid_bodies.size() > kMaxPersistedItems ||
      definition.joints.size() > kMaxPersistedItems)
  {
    report_error(reports, "MMD physics definition: item count exceeds persistence limit");
    return false;
  }

  IDProperty *root = blender::bke::idprop::create_group(kDefinitionProperty).release();
  if (!root) {
    report_error(reports, "MMD physics definition: failed to allocate root property");
    return false;
  }
  add_int(root, "schema_version", definition.schema_version);
  add_string(root, "source_model_name", definition.source_model_name);
  add_float(root, "source_pmx_version", definition.source_pmx_version);
  add_float(root, "coordinate_scale", definition.coordinate_scale);
  add_string(root, "coordinate_space", definition.coordinate_space);
  add_string(root, "rotation_order", definition.rotation_order);
  add_string(root, "joint_limits_space", definition.joint_limits_space);
  if (definition.source_file_size > size_t(INT32_MAX) ||
      definition.source_parse_end_offset > size_t(INT32_MAX))
  {
    IDP_FreeProperty(root);
    report_error(reports, "MMD physics definition: source diagnostics exceed IDProperty range");
    return false;
  }
  add_int(root, "source_file_size", int(definition.source_file_size));
  add_int(root, "source_parse_end_offset", int(definition.source_parse_end_offset));
  add_int(root, "bone_count", int(definition.bone_mapping.size()));
  add_int(root, "rigid_body_count", int(definition.rigid_bodies.size()));
  add_int(root, "joint_count", int(definition.joints.size()));
  add_bool(root, "validation_valid", definition.validation.valid);
  add_int(root, "validation_total_errors", definition.validation.total_errors);

  IDProperty *bones = IDP_NewIDPArray("bones");
  for (const MMDBoneMapping &bone : definition.bone_mapping) {
    IDProperty *item = blender::bke::idprop::create_group("bone").release();
    add_int(item, "pmx_index", bone.pmx_index);
    add_string(item, "pmx_name_local", bone.pmx_name_local);
    add_string(item, "pmx_name_universal", bone.pmx_name_universal);
    add_string(item, "blender_bone_name", bone.blender_bone_name);
    add_bool(item, "resolved", bone.resolved);
    append_group(bones, item);
  }
  add_property(root, bones);

  IDProperty *rigids = IDP_NewIDPArray("rigid_bodies");
  for (const MMDRigidBodyDefinition &rigid : definition.rigid_bodies) {
    IDProperty *item = blender::bke::idprop::create_group("rigid_body").release();
    add_int(item, "pmx_index", rigid.pmx_index);
    add_string(item, "name_local", rigid.name_local);
    add_string(item, "name_universal", rigid.name_universal);
    add_int(item, "pmx_bone_index", rigid.pmx_bone_index);
    add_string(item, "blender_bone_name", rigid.blender_bone_name);
    add_bool(item, "bone_resolved", rigid.bone_resolved);
    add_int(item, "collision_group", rigid.collision_group);
    add_int(item, "no_collision_group", rigid.no_collision_group);
    add_int(item, "shape_type", rigid.shape_type);
    add_vec3(item, "shape_size", rigid.shape_size);
    add_vec3(item, "position", rigid.position);
    add_vec3(item, "rotation", rigid.rotation);
    add_float(item, "mass", rigid.mass);
    add_float(item, "linear_damping", rigid.linear_damping);
    add_float(item, "angular_damping", rigid.angular_damping);
    add_float(item, "restitution", rigid.restitution);
    add_float(item, "friction", rigid.friction);
    add_int(item, "physics_type", rigid.physics_type);
    append_group(rigids, item);
  }
  add_property(root, rigids);

  IDProperty *joints = IDP_NewIDPArray("joints");
  for (const MMDJointDefinition &joint : definition.joints) {
    IDProperty *item = blender::bke::idprop::create_group("joint").release();
    add_int(item, "pmx_index", joint.pmx_index);
    add_string(item, "name_local", joint.name_local);
    add_string(item, "name_universal", joint.name_universal);
    add_int(item, "type", joint.type);
    add_int(item, "rigid_a_index", joint.rigid_a_index);
    add_int(item, "rigid_b_index", joint.rigid_b_index);
    add_vec3(item, "position", joint.position);
    add_vec3(item, "rotation", joint.rotation);
    add_vec3(item, "translation_min", joint.translation_min);
    add_vec3(item, "translation_max", joint.translation_max);
    add_vec3(item, "rotation_min", joint.rotation_min);
    add_vec3(item, "rotation_max", joint.rotation_max);
    int modes[3] = {int(joint.rotation_limit_mode[0]),
                    int(joint.rotation_limit_mode[1]),
                    int(joint.rotation_limit_mode[2])};
    add_property(item, new_int_array("rotation_limit_mode", modes, 3));
    add_vec3(item, "spring_translation", joint.spring_translation);
    add_vec3(item, "spring_rotation", joint.spring_rotation);
    append_group(joints, item);
  }
  add_property(root, joints);

  IDProperty *system = IDP_ID_system_properties_ensure(&model_root.id);
  IDP_ReplaceInGroup(system, root);
  return true;
}

bool deserialize_physics_definition(const Collection &model_root,
                                    MMDPhysicsDefinition &definition,
                                    ReportList *reports)
{
  IDProperty *system = model_root.id.system_properties;
  IDProperty *root = system ? IDP_GetPropertyTypeFromGroup(system, kDefinitionProperty, IDP_GROUP) : nullptr;
  if (!root) {
    report_error(reports, "MMD physics definition: property is missing or not a group");
    return false;
  }
  int schema = 0;
  int bone_count = 0;
  int rigid_count = 0;
  int joint_count = 0;
  if (!read_int(root, "schema_version", schema) || schema != 2 ||
      !read_int(root, "bone_count", bone_count) || !read_int(root, "rigid_body_count", rigid_count) ||
      !read_int(root, "joint_count", joint_count) || bone_count < 0 || rigid_count < 0 ||
      joint_count < 0 || bone_count > kMaxPersistedItems || rigid_count > kMaxPersistedItems ||
      joint_count > kMaxPersistedItems)
  {
    report_error(reports, "MMD physics definition: invalid schema or item counts");
    return false;
  }
  IDProperty *bones = IDP_GetPropertyTypeFromGroup(root, "bones", IDP_IDPARRAY);
  IDProperty *rigids = IDP_GetPropertyTypeFromGroup(root, "rigid_bodies", IDP_IDPARRAY);
  IDProperty *joints = IDP_GetPropertyTypeFromGroup(root, "joints", IDP_IDPARRAY);
  if (!bones || !rigids || !joints || bones->len != bone_count || rigids->len != rigid_count ||
      joints->len != joint_count)
  {
    report_error(reports, "MMD physics definition: array lengths do not match metadata");
    return false;
  }
  definition = MMDPhysicsDefinition{};
  definition.schema_version = schema;
  if (!read_string(root, "source_model_name", definition.source_model_name) ||
      !read_float(root, "source_pmx_version", definition.source_pmx_version) ||
      !read_float(root, "coordinate_scale", definition.coordinate_scale) ||
      !read_string(root, "coordinate_space", definition.coordinate_space) ||
      !read_string(root, "rotation_order", definition.rotation_order) ||
      definition.rotation_order != "YXZ")
  {
    report_error(reports, "MMD physics definition: invalid metadata types or values");
    return false;
  }
  if (!read_string(root, "joint_limits_space", definition.joint_limits_space) ||
      definition.joint_limits_space != "blender_import_space")
  {
    report_error(reports, "MMD physics definition: unsupported joint coordinate space");
    return false;
  }
  int source_size = 0;
  int parse_end = 0;
  if (!read_int(root, "source_file_size", source_size) || !read_int(root, "source_parse_end_offset", parse_end) ||
      source_size < 0 || parse_end < 0)
  {
    report_error(reports, "MMD physics definition: invalid source file diagnostics");
    return false;
  }
  definition.source_file_size = size_t(source_size);
  definition.source_parse_end_offset = size_t(parse_end);
  definition.bone_mapping.reserve(bone_count);
  for (int i = 0; i < bone_count; i++) {
    IDProperty *item = IDP_GetIndexArray(bones, i);
    int index = -1;
    MMDBoneMapping bone;
    if (!item || item->type != IDP_GROUP || !read_int(item, "pmx_index", index) || index != i ||
        !read_string(item, "pmx_name_local", bone.pmx_name_local) ||
        !read_string(item, "pmx_name_universal", bone.pmx_name_universal) ||
        !read_string(item, "blender_bone_name", bone.blender_bone_name) ||
        !read_bool(item, "resolved", bone.resolved))
    {
      report_error(reports, "MMD physics definition: invalid bone entry");
      return false;
    }
    bone.pmx_index = index;
    definition.bone_mapping.push_back(std::move(bone));
  }
  definition.rigid_bodies.reserve(rigid_count);
  for (int i = 0; i < rigid_count; i++) {
    IDProperty *item = IDP_GetIndexArray(rigids, i);
    MMDRigidBodyDefinition rigid;
    int value = 0;
    if (!item || item->type != IDP_GROUP || !read_int(item, "pmx_index", value) || value != i ||
        !read_string(item, "name_local", rigid.name_local) ||
        !read_string(item, "name_universal", rigid.name_universal) ||
        !read_int(item, "pmx_bone_index", rigid.pmx_bone_index) ||
        !read_string(item, "blender_bone_name", rigid.blender_bone_name) ||
        !read_bool(item, "bone_resolved", rigid.bone_resolved) ||
        !read_int(item, "collision_group", value))
    {
      report_error(reports, "MMD physics definition: invalid rigid body entry");
      return false;
    }
    rigid.pmx_index = i;
    rigid.collision_group = uint8_t(value);
    if (!read_int(item, "no_collision_group", value)) return false;
    rigid.no_collision_group = uint16_t(value);
    if (!read_int(item, "shape_type", value)) return false;
    rigid.shape_type = uint8_t(value);
    if (!read_vec3(item, "shape_size", rigid.shape_size) || !read_vec3(item, "position", rigid.position) ||
        !read_vec3(item, "rotation", rigid.rotation) || !read_float(item, "mass", rigid.mass) ||
        !read_float(item, "linear_damping", rigid.linear_damping) ||
        !read_float(item, "angular_damping", rigid.angular_damping) ||
        !read_float(item, "restitution", rigid.restitution) || !read_float(item, "friction", rigid.friction) ||
        !read_int(item, "physics_type", value))
    {
      report_error(reports, "MMD physics definition: invalid rigid body fields");
      return false;
    }
    rigid.physics_type = uint8_t(value);
    definition.rigid_bodies.push_back(std::move(rigid));
  }
  definition.joints.reserve(joint_count);
  for (int i = 0; i < joint_count; i++) {
    IDProperty *item = IDP_GetIndexArray(joints, i);
    MMDJointDefinition joint;
    int value = 0;
    if (!item || item->type != IDP_GROUP || !read_int(item, "pmx_index", value) || value != i ||
        !read_string(item, "name_local", joint.name_local) ||
        !read_string(item, "name_universal", joint.name_universal) || !read_int(item, "type", value))
    {
      report_error(reports, "MMD physics definition: invalid joint entry");
      return false;
    }
    joint.pmx_index = i;
    joint.type = uint8_t(value);
    if (!read_int(item, "rigid_a_index", joint.rigid_a_index) ||
        !read_int(item, "rigid_b_index", joint.rigid_b_index) ||
        !read_vec3(item, "position", joint.position) || !read_vec3(item, "rotation", joint.rotation) ||
        !read_vec3(item, "translation_min", joint.translation_min) ||
        !read_vec3(item, "translation_max", joint.translation_max) ||
        !read_vec3(item, "rotation_min", joint.rotation_min) ||
        !read_vec3(item, "rotation_max", joint.rotation_max) ||
        !read_vec3(item, "spring_translation", joint.spring_translation) ||
        !read_vec3(item, "spring_rotation", joint.spring_rotation))
    {
      report_error(reports, "MMD physics definition: invalid joint fields");
      return false;
    }
    IDProperty *modes = get_array(item, "rotation_limit_mode", 3, IDP_INT);
    if (!modes) {
      report_error(reports, "MMD physics definition: invalid joint rotation limit modes");
      return false;
    }
    const int *mode_values = IDP_array_int_get(modes);
    for (int axis = 0; axis < 3; axis++) {
      if (mode_values[axis] < 0 || mode_values[axis] > 1) {
        report_error(reports, "MMD physics definition: unsupported joint rotation limit mode");
        return false;
      }
      joint.rotation_limit_mode[axis] = MMDJointAxisLimitMode(mode_values[axis]);
      if (joint.translation_min[axis] > joint.translation_max[axis] ||
          (joint.rotation_limit_mode[axis] == MMDJointAxisLimitMode::Limited &&
           joint.rotation_min[axis] > joint.rotation_max[axis]))
      {
        report_error(reports, "MMD physics definition: invalid persisted joint limits");
        return false;
      }
    }
    definition.joints.push_back(std::move(joint));
  }
  bool valid = true;
  if (!read_bool(root, "validation_valid", valid) || !valid) {
    report_error(reports, "MMD physics definition: persisted definition is not valid");
    return false;
  }
  definition.validation.valid = true;
  return true;
}

bool write_physics_definition_json(const MMDPhysicsDefinition &definition,
                                   const char *filepath,
                                   ReportList *reports)
{
  if (filepath == nullptr || filepath[0] == '\0') {
    report_error(reports, "MMD physics definition export: filepath is empty");
    return false;
  }
  if (!definition.validation.valid) {
    report_error(reports, "MMD physics definition export: refusing to export invalid definition");
    return false;
  }

  const std::string output_path(filepath);
  const std::string temporary_path = output_path + ".tmp";
  if (!BLI_file_ensure_parent_dir_exists(output_path.c_str())) {
    report_error(reports, "MMD physics definition export: failed to create output directory");
    return false;
  }
  FILE *file = BLI_fopen(temporary_path.c_str(), "wb");
  if (file == nullptr) {
    report_error(reports, "MMD physics definition export: failed to open temporary JSON output");
    return false;
  }

  fprintf(file,
          "{\"schema_version\":%d,\"export_kind\":\"mmd_physics_definition\","
          "\"producer\":\"blender_mmdworld\",\"source\":{\"model_name\":",
          definition.schema_version);
  write_json_string(file, definition.source_model_name);
  fputs(",\"pmx_version\":", file);
  write_json_float(file, definition.source_pmx_version);
  fputs(",\"coordinate_scale\":", file);
  write_json_float(file, definition.coordinate_scale);
  fputs(",\"coordinate_space\":", file);
  write_json_string(file, definition.coordinate_space);
  fputs(",\"rotation_order\":", file);
  write_json_string(file, definition.rotation_order);
  fputs(",\"joint_limits_space\":", file);
  write_json_string(file, definition.joint_limits_space);
  fprintf(file,
          ",\"source_file_size\":%zu,\"source_parse_end_offset\":%zu},\"validation\":{\"valid\":true,\"total_errors\":%d},\"bones\":[",
          definition.source_file_size,
          definition.source_parse_end_offset,
          definition.validation.total_errors);
  for (size_t index = 0; index < definition.bone_mapping.size(); index++) {
    const MMDBoneMapping &bone = definition.bone_mapping[index];
    if (index != 0) {
      fputc(',', file);
    }
    fprintf(file, "{\"pmx_index\":%d,\"pmx_name_local\":", bone.pmx_index);
    write_json_string(file, bone.pmx_name_local);
    fputs(",\"pmx_name_universal\":", file);
    write_json_string(file, bone.pmx_name_universal);
    fputs(",\"blender_bone_name\":", file);
    write_json_string(file, bone.blender_bone_name);
    fputs(bone.resolved ? ",\"resolved\":true}" : ",\"resolved\":false}", file);
  }
  fputs("],\"rigid_bodies\":[", file);
  for (size_t index = 0; index < definition.rigid_bodies.size(); index++) {
    const MMDRigidBodyDefinition &rigid = definition.rigid_bodies[index];
    if (index != 0) {
      fputc(',', file);
    }
    fprintf(file, "{\"pmx_index\":%d,\"name_local\":", rigid.pmx_index);
    write_json_string(file, rigid.name_local);
    fputs(",\"name_universal\":", file);
    write_json_string(file, rigid.name_universal);
    fprintf(file, ",\"pmx_bone_index\":%d,\"blender_bone_name\":", rigid.pmx_bone_index);
    write_json_string(file, rigid.blender_bone_name);
    fputs(rigid.bone_resolved ? ",\"bone_resolved\":true" : ",\"bone_resolved\":false", file);
    fprintf(file,
            ",\"collision_group\":%u,\"no_collision_group\":%u,\"shape_type\":%u,\"shape_size\":",
            unsigned(rigid.collision_group),
            unsigned(rigid.no_collision_group),
            unsigned(rigid.shape_type));
    write_float3(file, rigid.shape_size);
    fputs(",\"position\":", file);
    write_float3(file, rigid.position);
    fputs(",\"rotation\":", file);
    write_float3(file, rigid.rotation);
    fputs(",\"mass\":", file);
    write_json_float(file, rigid.mass);
    fputs(",\"linear_damping\":", file);
    write_json_float(file, rigid.linear_damping);
    fputs(",\"angular_damping\":", file);
    write_json_float(file, rigid.angular_damping);
    fputs(",\"restitution\":", file);
    write_json_float(file, rigid.restitution);
    fputs(",\"friction\":", file);
    write_json_float(file, rigid.friction);
    fprintf(file, ",\"physics_type\":%u}", unsigned(rigid.physics_type));
  }
  fputs("],\"joints\":[", file);
  for (size_t index = 0; index < definition.joints.size(); index++) {
    const MMDJointDefinition &joint = definition.joints[index];
    if (index != 0) {
      fputc(',', file);
    }
    fprintf(file, "{\"pmx_index\":%d,\"name_local\":", joint.pmx_index);
    write_json_string(file, joint.name_local);
    fputs(",\"name_universal\":", file);
    write_json_string(file, joint.name_universal);
    fprintf(file,
            ",\"type\":%u,\"rigid_a_index\":%d,\"rigid_b_index\":%d,\"position\":",
            unsigned(joint.type),
            joint.rigid_a_index,
            joint.rigid_b_index);
    write_float3(file, joint.position);
    fputs(",\"rotation\":", file);
    write_float3(file, joint.rotation);
    fputs(",\"translation_min\":", file);
    write_float3(file, joint.translation_min);
    fputs(",\"translation_max\":", file);
    write_float3(file, joint.translation_max);
    fputs(",\"rotation_min\":", file);
    write_float3(file, joint.rotation_min);
    fputs(",\"rotation_max\":", file);
    write_float3(file, joint.rotation_max);
    fputs(",\"rotation_limit_mode\":", file);
    write_limit_modes(file, joint.rotation_limit_mode);
    fputs(",\"spring_translation\":", file);
    write_float3(file, joint.spring_translation);
    fputs(",\"spring_rotation\":", file);
    write_float3(file, joint.spring_rotation);
    fputc('}', file);
  }
  fputs("]}\n", file);
  const bool write_ok = ferror(file) == 0;
  const bool close_ok = fclose(file) == 0;
  if (!write_ok || !close_ok) {
    BLI_delete(temporary_path.c_str(), false, false);
    report_error(reports, "MMD physics definition export: failed while writing JSON");
    return false;
  }
  if (BLI_rename_overwrite(temporary_path.c_str(), output_path.c_str()) != 0) {
    BLI_delete(temporary_path.c_str(), false, false);
    report_error(reports, "MMD physics definition export: failed to publish JSON");
    return false;
  }
  return true;
}

namespace {

void add_mapping_issue(MMDPhysicsMappingReport &report,
                       const MMDPhysicsMappingIssueSeverity severity,
                       const std::string &path,
                       const std::string &message)
{
  report.issues.push_back({severity, path, message});
  report.total_issues++;
  if (severity == MMDPhysicsMappingIssueSeverity::Error) {
    report.mapping_valid = false;
  }
}

std::string indexed_path(const char *section, const int index, const char *field)
{
  std::ostringstream path;
  path << section << "[" << index << "]";
  if (field[0] != '\0') {
    path << "." << field;
  }
  return path.str();
}

bool finite_array(const std::array<float, 3> &values)
{
  return std::all_of(values.begin(), values.end(), [](const float value) {
    return std::isfinite(value);
  });
}

bool validate_rigid_body(const MMDRigidBodyDefinition &rigid,
                         const int index,
                         const MMDPhysicsDefinition &definition,
                         MMDPhysicsMappingReport &report)
{
  bool valid = true;
  const std::string prefix = indexed_path("rigid_bodies", index, "") + ".";
  if (rigid.pmx_index != index) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "pmx_index",
                      "does not match array index");
    valid = false;
  }
  if (rigid.collision_group > 15) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "collision_group",
                      "must be in range 0..15");
    valid = false;
  }
  if (rigid.shape_type > 2) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "shape_type",
                      "unsupported shape type");
    valid = false;
  }
  else {
    const int required_axes = rigid.shape_type == 0 ? 1 : 2;
    const int axis_count = rigid.shape_type == 1 ? 3 : required_axes;
    for (int axis = 0; axis < axis_count; axis++) {
      if (!std::isfinite(rigid.shape_size[axis]) || rigid.shape_size[axis] <= 0.0f) {
        add_mapping_issue(report,
                          MMDPhysicsMappingIssueSeverity::Error,
                          prefix + "shape_size",
                          "required dimensions must be positive and finite");
        valid = false;
        break;
      }
    }
  }
  for (const char *field : {"shape_size", "position", "rotation"}) {
    const std::array<float, 3> *values = std::string(field) == "shape_size" ?
                                              &rigid.shape_size :
                                              std::string(field) == "position" ? &rigid.position :
                                                                                   &rigid.rotation;
    if (!finite_array(*values)) {
      add_mapping_issue(report,
                        MMDPhysicsMappingIssueSeverity::Error,
                        prefix + field,
                        "contains NaN or Inf");
      valid = false;
    }
  }
  if (!std::isfinite(rigid.mass) || rigid.mass < 0.0f) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "mass",
                      "must be finite and non-negative");
    valid = false;
  }
  if (!std::isfinite(rigid.linear_damping) || !std::isfinite(rigid.angular_damping)) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "damping",
                      "must be finite");
    valid = false;
  }
  else if (rigid.linear_damping < 0.0f || rigid.angular_damping < 0.0f) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Warning,
                      prefix + "damping",
                      "contains a negative value");
  }
  if (!std::isfinite(rigid.restitution) || !std::isfinite(rigid.friction) ||
      rigid.restitution < 0.0f || rigid.friction < 0.0f)
  {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "surface_parameters",
                      "must be finite and non-negative");
    valid = false;
  }
  else if (rigid.restitution > 1.0f || rigid.friction > 1.0f) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Warning,
                      prefix + "surface_parameters",
                      "contains a value above 1");
  }
  if (rigid.physics_type > 2) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "physics_type",
                      "unsupported physics type");
    valid = false;
  }
  if (rigid.pmx_bone_index < -1 || rigid.pmx_bone_index >= int(definition.bone_mapping.size())) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "pmx_bone_index",
                      "out of range");
    valid = false;
  }
  return valid;
}

bool validate_joint(const MMDJointDefinition &joint,
                    const int index,
                    const MMDPhysicsDefinition &definition,
                    MMDPhysicsMappingReport &report)
{
  bool valid = true;
  const std::string prefix = indexed_path("joints", index, "") + ".";
  if (joint.pmx_index != index) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "pmx_index",
                      "does not match array index");
    valid = false;
  }
  if (joint.type != 0) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "type",
                      "unsupported joint type");
    valid = false;
  }
  const bool rigid_a_valid = joint.rigid_a_index >= 0 &&
                             joint.rigid_a_index < int(definition.rigid_bodies.size());
  const bool rigid_b_valid = joint.rigid_b_index >= 0 &&
                             joint.rigid_b_index < int(definition.rigid_bodies.size());
  if (!rigid_a_valid) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "rigid_a_index",
                      "out of range");
    valid = false;
  }
  if (!rigid_b_valid) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      prefix + "rigid_b_index",
                      "out of range");
    valid = false;
  }
  if (rigid_a_valid && rigid_b_valid && joint.rigid_a_index == joint.rigid_b_index) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Warning,
                      prefix + "rigid_endpoints",
                      "both endpoints reference the same rigid body");
  }
  for (const char *field : {"position", "rotation", "translation_min", "translation_max", "rotation_min", "rotation_max", "spring_translation", "spring_rotation"}) {
    const std::array<float, 3> *values = nullptr;
    if (std::string(field) == "position") values = &joint.position;
    else if (std::string(field) == "rotation") values = &joint.rotation;
    else if (std::string(field) == "translation_min") values = &joint.translation_min;
    else if (std::string(field) == "translation_max") values = &joint.translation_max;
    else if (std::string(field) == "rotation_min") values = &joint.rotation_min;
    else if (std::string(field) == "rotation_max") values = &joint.rotation_max;
    else if (std::string(field) == "spring_translation") values = &joint.spring_translation;
    else values = &joint.spring_rotation;
    if (!finite_array(*values)) {
      add_mapping_issue(report,
                        MMDPhysicsMappingIssueSeverity::Error,
                        prefix + field,
                        "contains NaN or Inf");
      valid = false;
    }
  }
  for (int axis = 0; axis < 3; axis++) {
    if (joint.translation_min[axis] > joint.translation_max[axis]) {
      add_mapping_issue(report,
                        MMDPhysicsMappingIssueSeverity::Error,
                        prefix + "translation_limits",
                        "minimum exceeds maximum");
      valid = false;
    }
    if (joint.rotation_limit_mode[axis] != MMDJointAxisLimitMode::Limited &&
        joint.rotation_limit_mode[axis] != MMDJointAxisLimitMode::Free)
    {
      add_mapping_issue(report,
                        MMDPhysicsMappingIssueSeverity::Error,
                        prefix + "rotation_limit_mode",
                        "unsupported axis mode");
      valid = false;
    }
    else if (joint.rotation_limit_mode[axis] == MMDJointAxisLimitMode::Limited &&
             joint.rotation_min[axis] > joint.rotation_max[axis])
    {
      add_mapping_issue(report,
                        MMDPhysicsMappingIssueSeverity::Error,
                        prefix + "rotation_limits",
                        "limited axis minimum exceeds maximum");
      valid = false;
    }
    if (joint.spring_translation[axis] < 0.0f || joint.spring_rotation[axis] < 0.0f) {
      add_mapping_issue(report,
                        MMDPhysicsMappingIssueSeverity::Warning,
                        prefix + "spring",
                        "contains a negative value");
    }
  }
  return valid;
}

}  // namespace

MMDPhysicsMappingReport validate_physics_mapping(const MMDPhysicsDefinition &definition,
                                                 const bArmature *armature)
{
  MMDPhysicsMappingReport report;
  report.definition_valid = definition.validation.valid;
  if (!definition.validation.valid) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      "definition",
                      "persisted definition is not valid");
  }

  if (!armature) {
    add_mapping_issue(report,
                      MMDPhysicsMappingIssueSeverity::Error,
                      "armature",
                      "current Armature data is missing");
  }
  else {
    bArmature *mutable_armature = const_cast<bArmature *>(armature);
    BKE_armature_bone_hash_make(mutable_armature);
  }

  for (int i = 0; i < int(definition.bone_mapping.size()); i++) {
    const MMDBoneMapping &bone = definition.bone_mapping[i];
    bool valid = bone.pmx_index == i;
    if (!valid) {
      add_mapping_issue(report,
                        MMDPhysicsMappingIssueSeverity::Error,
                        indexed_path("bones", i, "pmx_index"),
                        "does not match array index");
    }
    if (bone.blender_bone_name.empty()) {
      add_mapping_issue(report,
                        MMDPhysicsMappingIssueSeverity::Error,
                        indexed_path("bones", i, "blender_bone_name"),
                        "is empty");
      valid = false;
    }
    const bool found = armature && !bone.blender_bone_name.empty() &&
                       BKE_armature_find_bone_name(const_cast<bArmature *>(armature),
                                                   bone.blender_bone_name.c_str()) != nullptr;
    if (found) {
      report.resolved_bones++;
    }
    else {
      add_mapping_issue(report,
                        MMDPhysicsMappingIssueSeverity::Error,
                        indexed_path("bones", i, "blender_bone_name"),
                        "not found in current Armature");
      valid = false;
    }
    if (valid) {
      /* The mapping is valid for this entry. */
    }
    else {
      report.unresolved_bones++;
    }
  }

  for (int i = 0; i < int(definition.rigid_bodies.size()); i++) {
    const MMDRigidBodyDefinition &rigid = definition.rigid_bodies[i];
    const bool valid = validate_rigid_body(rigid, i, definition, report);
    if (!valid) {
      report.invalid_rigid_bodies++;
    }
    if (rigid.pmx_bone_index == -1) {
      report.resolved_rigid_bones++;
    }
    else if (rigid.pmx_bone_index >= 0 && rigid.pmx_bone_index < int(definition.bone_mapping.size()) &&
             armature)
    {
      const MMDBoneMapping &mapping = definition.bone_mapping[rigid.pmx_bone_index];
      const bool snapshot_matches = rigid.blender_bone_name == mapping.blender_bone_name;
      const bool current_bone_exists =
          BKE_armature_find_bone_name(const_cast<bArmature *>(armature),
                                      mapping.blender_bone_name.c_str()) != nullptr;
      if (snapshot_matches && current_bone_exists) {
        report.resolved_rigid_bones++;
      }
      else {
        report.unresolved_rigid_bones++;
        add_mapping_issue(report,
                          MMDPhysicsMappingIssueSeverity::Error,
                          indexed_path("rigid_bodies", i, "blender_bone_name"),
                          snapshot_matches ? "does not resolve to current Armature" :
                                              "does not match referenced bone mapping");
      }
    }
    else {
      report.unresolved_rigid_bones++;
      add_mapping_issue(report,
                        MMDPhysicsMappingIssueSeverity::Error,
                        indexed_path("rigid_bodies", i, "blender_bone_name"),
                        "does not resolve to current Armature");
    }
  }

  for (int i = 0; i < int(definition.joints.size()); i++) {
    const MMDJointDefinition &joint = definition.joints[i];
    const bool valid = validate_joint(joint, i, definition, report);
    if (!valid) {
      report.invalid_joints++;
    }
    const bool endpoints_valid = joint.rigid_a_index >= 0 &&
                                  joint.rigid_a_index < int(definition.rigid_bodies.size()) &&
                                  joint.rigid_b_index >= 0 &&
                                  joint.rigid_b_index < int(definition.rigid_bodies.size());
    if (endpoints_valid) {
      report.resolved_joint_endpoints++;
    }
    else {
      report.invalid_joint_endpoints++;
    }
  }
  return report;
}

MMDPhysicsDebugReport build_physics_debug_report(
    const MMDPhysicsDefinition &definition, const MMDPhysicsMappingReport &mapping_report)
{
  MMDPhysicsDebugReport report;
  report.definition_valid = definition.validation.valid;
  report.mapping_valid = mapping_report.mapping_valid;
  report.bone_count = int(definition.bone_mapping.size());
  report.resolved_bone_count = mapping_report.resolved_bones;
  report.rigid_body_count = int(definition.rigid_bodies.size());
  report.joint_count = int(definition.joints.size());
  report.resolved_joint_count = mapping_report.resolved_joint_endpoints;

  for (const MMDPhysicsMappingIssue &issue : mapping_report.issues) {
    if (issue.severity == MMDPhysicsMappingIssueSeverity::Error) {
      report.error_count++;
    }
    else if (issue.severity == MMDPhysicsMappingIssueSeverity::Warning) {
      report.warning_count++;
    }
    if (issue.severity != MMDPhysicsMappingIssueSeverity::Info) {
      const char *severity = issue.severity == MMDPhysicsMappingIssueSeverity::Error ? "ERROR" :
                                                                                         "WARNING";
      report.diagnostics.push_back(std::string(severity) + " " + issue.path + ": " +
                                   issue.message);
    }
  }

  for (const MMDRigidBodyDefinition &rigid : definition.rigid_bodies) {
    if (rigid.collision_group < report.collision_group_counts.size()) {
      report.collision_group_counts[rigid.collision_group]++;
    }
    for (int group = 0; group < 16; group++) {
      if ((rigid.no_collision_group & (uint16_t(1) << group)) != 0) {
        report.collision_group_mask_counts[group]++;
      }
    }
    if (rigid.shape_type < report.rigid_shape_counts.size()) {
      report.rigid_shape_counts[rigid.shape_type]++;
    }
    if (rigid.physics_type < report.rigid_type_counts.size()) {
      report.rigid_type_counts[rigid.physics_type]++;
    }

    if (rigid.pmx_bone_index == -1) {
      report.unbound_rigid_body_count++;
    }
    else if (rigid.pmx_bone_index >= 0 &&
             rigid.pmx_bone_index < int(definition.bone_mapping.size()) && rigid.bone_resolved &&
             definition.bone_mapping[rigid.pmx_bone_index].resolved)
    {
      report.bound_rigid_body_count++;
    }
    else {
      report.invalid_rigid_body_binding_count++;
    }
  }

  for (const MMDJointDefinition &joint : definition.joints) {
    if (joint.rigid_a_index == joint.rigid_b_index && joint.rigid_a_index >= 0 &&
        joint.rigid_a_index < int(definition.rigid_bodies.size()))
    {
      report.joints_with_same_endpoint++;
    }
    for (int axis = 0; axis < 3; axis++) {
      if (joint.rotation_limit_mode[axis] == MMDJointAxisLimitMode::Limited) {
        report.joint_rotation_limited_axes[axis]++;
      }
      else if (joint.rotation_limit_mode[axis] == MMDJointAxisLimitMode::Free) {
        report.joint_rotation_free_axes[axis]++;
      }
      if (joint.spring_translation[axis] < 0.0f) {
        report.negative_spring_count++;
      }
      if (joint.spring_rotation[axis] < 0.0f) {
        report.negative_spring_count++;
      }
    }
  }

  report.static_ready = report.definition_valid && report.mapping_valid &&
                        report.error_count == 0 && mapping_report.invalid_rigid_bodies == 0 &&
                        mapping_report.invalid_joints == 0 &&
                        report.invalid_rigid_body_binding_count == 0 &&
                        report.resolved_joint_count == report.joint_count;
  return report;
}

namespace {

/** Map a PMX rigid-body shape type (0=SPHERE, 1=BOX, 2=CAPSULE) to the Blender
 * native #eRigidBody_Shape value (RB_SHAPE_SPHERE=1, RB_SHAPE_BOX=0,
 * RB_SHAPE_CAPSULE=2). */
eRigidBody_Shape mmd_shape_to_blender_shape(const uint8_t shape_type)
{
  switch (shape_type) {
    case 1:
      return RB_SHAPE_BOX;
    case 2:
      return RB_SHAPE_CAPSULE;
    case 0:
    default:
      return RB_SHAPE_SPHERE;
  }
}

/** Compute the full bounding-box extents (in Blender units) that a placeholder
 * display mesh should span so the native rigid-body collider (derived from the
 * object bounding box) matches the MMD shape size, mirroring mmd_tools which
 * doubles the sphere size (PMX stores the radius) while keeping box/capsule as
 * full dimensions. */
void mmd_shape_bounds(const MMDRigidBodyDefinition &rigid, float r_bounds[3])
{
  const float s0 = std::max(rigid.shape_size[0], 0.0001f);
  const float s1 = std::max(rigid.shape_size[1], 0.0001f);
  const float s2 = std::max(rigid.shape_size[2], 0.0001f);
  switch (rigid.shape_type) {
    case 0: {
      /* Sphere: PMX stores the radius; the native sphere collider derives its
       * radius from half the largest bounding dimension, so use the diameter. */
      const float diameter = s0 * 2.0f;
      r_bounds[0] = r_bounds[1] = r_bounds[2] = diameter;
      break;
    }
    case 1:
      r_bounds[0] = s0;
      r_bounds[1] = s1;
      r_bounds[2] = s2;
      break;
    default:
      r_bounds[0] = s0;
      r_bounds[1] = s1;
      r_bounds[2] = s2;
      break;
  }
}

/** Create an 8-vertex placeholder box mesh spanning `bounds` for display. The
 * native rigid-body collision shape (BOX/SPHERE/CAPSULE) is derived from the
 * object bounding box, so the mesh only needs to provide those extents. */
Mesh *create_display_box_mesh(Main *bmain, const char *name, const float bounds[3])
{
  const float hx = bounds[0] * 0.5f;
  const float hy = bounds[1] * 0.5f;
  const float hz = bounds[2] * 0.5f;

  Mesh *tmp = BKE_mesh_new_nomain(8, 0, 0, 0);
  MutableSpan<float3> positions = tmp->vert_positions_for_write();
  const float coords[8][3] = {
      {-hx, -hy, -hz}, {hx, -hy, -hz}, {-hx, hy, -hz}, {hx, hy, -hz},
      {-hx, -hy, hz},  {hx, -hy, hz},  {-hx, hy, hz},  {hx, hy, hz},
  };
  for (int i = 0; i < 8; i++) {
    positions[i] = float3(coords[i][0], coords[i][1], coords[i][2]);
  }
  tmp->tag_positions_changed();

  Mesh *mesh_in_main = BKE_mesh_add(bmain, name);
  BKE_mesh_nomain_to_mesh(tmp, mesh_in_main, nullptr);
  return mesh_in_main;
}

/** Build a bone's REST matrix in Armature space by walking the parent chain and
 * composing `bone_mat / bone->head` (+ parent length offset), mirroring
 * `BKE_armature_where_is_bone`. This does not rely on `Bone::arm_mat` being
 * pre-populated (which can be all-zero on the original Armature data). */
void compute_bone_rest_armature_matrix(const Bone *bone, float r_mat[4][4])
{
  const Bone *chain[64];
  int n = 0;
  for (const Bone *b = bone; b != nullptr && n < 64; b = b->parent) {
    chain[n++] = b;
  }
  if (n == 0) {
    unit_m4(r_mat);
    return;
  }
  /* Root bone: rotation from bone_mat, translation at head. */
  float m[4][4];
  copy_m4_m3(m, chain[n - 1]->bone_mat);
  copy_v3_v3(m[3], chain[n - 1]->head);
  /* Compose children down to the target bone. */
  for (int i = n - 2; i >= 0; i--) {
    float offs[4][4];
    BKE_bone_offset_matrix_get(chain[i], offs);
    float t[4][4];
    mul_m4_m4m4(t, m, offs);
    copy_m4_m4(m, t);
  }
  copy_m4_m4(r_mat, m);
}

/** Add a Copy Transforms (type 1) or Copy Rotation (type 2) constraint on a pose
 * bone that targets `target_obj`, so the simulated rigid-body motion drives the
 * bone (and therefore the mesh) — the mmd_tools `mmd_tools_rigid_track`
 * mechanism. */
void add_bone_copy_constraint(Object *armature,
                              bPoseChannel *pchan,
                              Object *target_obj,
                              const bool rotation_only)
{
  if (armature == nullptr || pchan == nullptr || target_obj == nullptr) {
    return;
  }
  bConstraint *con = BKE_constraint_add_for_pose(
      armature,
      pchan,
      "mmd_tools_rigid_track",
      rotation_only ? CONSTRAINT_TYPE_ROTLIKE : CONSTRAINT_TYPE_TRANSLIKE);
  if (con == nullptr) {
    return;
  }
  con->enforce = 1.0f;
  /* Active (not muted): the constraint drives the bone whenever the simulated
   * rigid-body object moves (mmd_tools unmutes `mmd_tools_rigid_track` during
   * physics; we keep it active so the ptcache bake drives the bone). */
  con->flag &= ~CONSTRAINT_OFF;
  con->ownspace = CONSTRAINT_SPACE_WORLD;
  con->tarspace = CONSTRAINT_SPACE_WORLD;
  if (rotation_only) {
    bRotateLikeConstraint *data = static_cast<bRotateLikeConstraint *>(con->data);
    data->tar = target_obj;
  }
  else {
    bTransLikeConstraint *data = static_cast<bTransLikeConstraint *>(con->data);
    data->tar = target_obj;
  }
}


void compute_object_world_matrix(Object *obj, float r_world[4][4])
{
  if (obj == nullptr) {
    unit_m4(r_world);
    return;
  }
  const Object *chain[64];
  int n = 0;
  for (const Object *cur = obj; cur != nullptr && n < 64; cur = cur->parent) {
    chain[n++] = cur;
  }
  if (n == 0) {
    unit_m4(r_world);
    return;
  }
  float world[4][4];
  BKE_object_matrix_local_get(const_cast<Object *>(chain[n - 1]), world);
  for (int i = n - 2; i >= 0; i--) {
    float local[4][4];
    BKE_object_matrix_local_get(const_cast<Object *>(chain[i]), local);
    float tmp[4][4];
    mul_m4_m4m4(tmp, world, local);
    copy_m4_m4(world, tmp);
  }
  copy_m4_m4(r_world, world);
}

/** World-space rest matrix of a rigid body: the definition (blender_import_space
 * local coords) carried into world by the Armature/model-root world matrix. */
void rigid_body_world_rest_matrix(const float arm_world[4][4],
                                  const MMDRigidBodyDefinition &rigid,
                                  float r_world[4][4])
{
  float def_local[4][4];
  eulO_to_mat4(def_local, rigid.rotation.data(), EULER_ORDER_YXZ);
  copy_v3_v3(def_local[3], rigid.position.data());
  mul_m4_m4m4(r_world, arm_world, def_local);
}

void update_native_rigid_object_transform(Object *ob,
                                          const float arm_world[4][4],
                                          const MMDRigidBodyDefinition &rigid)
{
  float world[4][4];
  rigid_body_world_rest_matrix(arm_world, rigid, world);
  copy_v3_v3(ob->loc, world[3]);
  /* mmd_tools sets rotation_mode="YXZ"; the object/depsgraph use this order. */
  ob->rotmode = ROT_MODE_YXZ;
  mat4_to_eulO(ob->rot, EULER_ORDER_YXZ, world);
}

std::string rigid_body_object_name(const MMDRigidBodyDefinition &rigid, const int index)
{
  if (!rigid.name_local.empty()) {
    return rigid.name_local;
  }
  return "MMD_Rigid_" + std::to_string(index);
}

std::string joint_object_name(const MMDJointDefinition &joint, const int index)
{
  if (!joint.name_local.empty()) {
    return joint.name_local;
  }
  return "MMD_Joint_" + std::to_string(index);
}

/** Bind a static (physics_type 0) rigid body object to its bone so the collider
 * follows the animated skeleton (mmd_tools: BONE PARENT + kinematic). The object
 * keeps its rest world matrix via `parentinv`; as the bone moves (VMD pose) the
 * collider tracks it, and because the body is passive/kinematic it does not
 * participate in the physics response.
 *
 * \param arm_world: The Armature's (model-root) world matrix, computed reliably
 * so the collider rest position matches the mesh even before depsgraph eval. */
void setup_static_bone_parent(Object *ob,
                              Object *armature,
                              const float arm_world[4][4],
                              const MMDRigidBodyDefinition &rigid)
{
  if (armature == nullptr || armature->type != OB_ARMATURE || armature->data == nullptr ||
      !rigid.bone_resolved || rigid.blender_bone_name.empty())
  {
    update_native_rigid_object_transform(ob, arm_world, rigid);
    return;
  }
  bArmature *arm = reinterpret_cast<bArmature *>(armature->data);
  Bone *bone = BKE_armature_find_bone_name(arm, rigid.blender_bone_name.c_str());
  if (bone == nullptr) {
    update_native_rigid_object_transform(ob, arm_world, rigid);
    return;
  }

  /* Rigid-body rest WORLD matrix (definition local coords carried into world). */
  float R_rest[4][4];
  rigid_body_world_rest_matrix(arm_world, rigid, R_rest);

  /* Bone parent matrix at rest (armature space); Blender parents at the bone tail. */
  float bone_mat[4][4];
  copy_m4_m4(bone_mat, bone->arm_mat);
  float tail[3];
  copy_v3_v3(tail, bone->arm_mat[1]);
  mul_v3_fl(tail, bone->length * ob->parent_bone_head_tail_factor);
  add_v3_v3(bone_mat[3], tail);

  /* World-space bone parent matrix. */
  float parent_world[4][4];
  mul_m4_m4m4(parent_world, arm_world, bone_mat);

  float pinv[4][4];
  invert_m4_m4(pinv, parent_world);
  mul_m4_m4m4(ob->parentinv, pinv, R_rest);

  ob->parent = armature;
  ob->partype = PARBONE;
  STRNCPY(ob->parsubstr, rigid.blender_bone_name.c_str());

  /* Local transform is identity; `parentinv` carries the rest offset. */
  zero_v3(ob->loc);
  ob->rotmode = ROT_MODE_QUAT;
  ob->quat[0] = 1.0f;
  ob->quat[1] = ob->quat[2] = ob->quat[3] = 0.0f;
  ob->scale[0] = ob->scale[1] = ob->scale[2] = 1.0f;
}

/** Find a direct child collection of `parent` by name. */
Collection *find_child_collection(Collection *parent, const char *name)
{
  if (parent == nullptr) {
    return nullptr;
  }
  for (CollectionChild *child = static_cast<CollectionChild *>(parent->children.first);
       child != nullptr;
       child = child->next)
  {
    if (child->collection != nullptr && STREQ(child->collection->id.name + 2, name)) {
      return child->collection;
    }
  }
  return nullptr;
}

/** Remove every native rigid body object / joint / NCC object this model already
 * has (they live in the "Rigid Bodies" collection under `model_collection`), so
 * #create_native_rigid_bodies is idempotent: calling `build_rig` after a PMX
 * import must not double the body count. */
void cleanup_existing_native_rigid_bodies(Main *bmain, Scene *scene, Collection *model_collection)
{
  if (bmain == nullptr || scene == nullptr || model_collection == nullptr) {
    return;
  }
  Collection *rigid_col = find_child_collection(model_collection, "Rigid Bodies");
  if (rigid_col == nullptr) {
    return;
  }
  std::vector<Object *> to_remove;
  FOREACH_COLLECTION_OBJECT_RECURSIVE_BEGIN (rigid_col, ob) {
    to_remove.push_back(ob);
  }
  FOREACH_COLLECTION_OBJECT_RECURSIVE_END;
  for (Object *ob : to_remove) {
    if (ob->rigidbody_object != nullptr) {
      /* Remove rigid-body settings and this object from the RBW group collection. */
      BKE_rigidbody_remove_object(bmain, scene, ob, false);
    }
    if (ob->rigidbody_constraint != nullptr) {
      BKE_rigidbody_remove_constraint(bmain, scene, ob, false);
    }
    /* Delete the object; BKE_id_delete unlinks it from every collection it is in
     * (the "Rigid Bodies" display collection, master collection, etc.) and frees it. */
    BKE_id_delete(bmain, ob);
  }
  if (RigidBodyWorld *rbw = BKE_rigidbody_get_world(scene)) {
    BKE_rigidbody_cache_reset(rbw);
  }
}

}  // namespace

bool create_native_rigid_bodies(Main *bmain,
                                Scene *scene,
                                Object *armature,
                                Collection *model_collection,
                                const MMDPhysicsDefinition &definition,
                                ReportList *reports)
{
  if (bmain == nullptr || scene == nullptr) {
    return false;
  }

  auto report = [reports](const std::string &message) {
    if (reports != nullptr) {
      BKE_report(reports, RPT_WARNING, message.c_str());
    }
  };

  /* 1. Ensure the Scene has a native RigidBodyWorld and its object/constraint
   * group collections (these carry the bodies/constraints into the sim). */
  RigidBodyWorld *rbw = BKE_rigidbody_get_world(scene);
  if (rbw == nullptr) {
    rbw = BKE_rigidbody_create_world(scene);
    if (rbw == nullptr) {
      report("MMD physics: failed to create the native RigidBodyWorld");
      return false;
    }
    BKE_rigidbody_validate_sim_world(scene, rbw, false);
    scene->rigidbody_world = rbw;
  }
  if (rbw->group == nullptr) {
    rbw->group = BKE_collection_add(bmain, nullptr, "RigidBodyWorld");
    id_us_plus(&rbw->group->id);
  }
  if (rbw->constraints == nullptr) {
    rbw->constraints = BKE_collection_add(bmain, nullptr, "RigidBodyConstraints");
    id_us_plus(&rbw->constraints->id);
  }

  /* 1a. Idempotency: drop any native rigid body / joint / NCC objects this model
   * already owns so a repeated build_rig (after an import that already built them)
   * does not duplicate bodies. */
  cleanup_existing_native_rigid_bodies(bmain, scene, model_collection);

  /* 2. Display collection under the model root (so the helper objects are
   * organised under the imported PMX model in the Outliner). Reuse the existing
   * "Rigid Bodies" collection when present to avoid duplicates. */
  Collection *rigid_collection = nullptr;
  if (model_collection != nullptr) {
    rigid_collection = find_child_collection(model_collection, "Rigid Bodies");
    if (rigid_collection == nullptr) {
      rigid_collection = BKE_collection_add(bmain, model_collection, "Rigid Bodies");
    }
  }

  /* The Armature/model-root world matrix. The definition positions are in the
   * model's import space, so every rigid body is placed in world by multiplying
   * with this matrix (reliable even before depsgraph evaluation). */
  float arm_world[4][4];
  compute_object_world_matrix(armature, arm_world);

  /* 3. Create one native rigid body object per MMD rigid body. */
  const int rigid_count = int(definition.rigid_bodies.size());
  std::vector<Object *> rigid_objects(rigid_count, nullptr);
  std::vector<std::array<float, 3>> rigid_positions(rigid_count, {0.0f, 0.0f, 0.0f});
  std::vector<float> rigid_ranges(rigid_count, 0.0f);
  int created_rigid_bodies = 0;

  for (int i = 0; i < rigid_count; i++) {
    const MMDRigidBodyDefinition &rigid = definition.rigid_bodies[i];
    if (rigid.pmx_index != i) {
      continue;
    }
    const std::string object_name = rigid_body_object_name(rigid, i);

    Object *ob = BKE_object_add_only_object(bmain, OB_MESH, object_name.c_str());
    if (ob == nullptr) {
      continue;
    }

    /* Placeholder box mesh whose bounding box matches the MMD collider size. */
    float shape_bounds[3];
    mmd_shape_bounds(rigid, shape_bounds);
    rigid_ranges[i] = std::max({shape_bounds[0], shape_bounds[1], shape_bounds[2]});
    /* Store the body's WORLD position (for the NCC distance test). */
    float world_pos[4][4];
    rigid_body_world_rest_matrix(arm_world, rigid, world_pos);
    rigid_positions[i] = {world_pos[3][0], world_pos[3][1], world_pos[3][2]};
    Mesh *mesh = create_display_box_mesh(bmain, object_name.c_str(), shape_bounds);
    ob->data = &mesh->id;

    /* Static bodies are bone-parented to the animated skeleton (mmd_tools
     * updateRigid for type 0), so the collider follows the bone. Dynamic bodies
     * are free objects whose simulated motion is transferred back to their bones
     * after the bake (see bake_rigidbody_physics_to_bones). */
    if (rigid.physics_type == 0) {
      setup_static_bone_parent(ob, armature, arm_world, rigid);
    }
    else {
      update_native_rigid_object_transform(ob, arm_world, rigid);
    }

    if (rigid_collection != nullptr) {
      BKE_collection_object_add(bmain, rigid_collection, ob);
    }
    BKE_collection_object_add(bmain, rbw->group, ob);

    /* Record the rigid-body index on the object so the post-bake transfer can map
     * a baked body back to its definition entry and bound bone. */
    IDProperty *obj_props = IDP_ID_system_properties_ensure(&ob->id);
    IDProperty *idx_prop = IDP_GetPropertyTypeFromGroup(obj_props, "mmd_physics_rigid_index", IDP_INT);
    if (idx_prop != nullptr) {
      IDP_int_set(idx_prop, i);
    }
    else {
      IDP_AddToGroup(obj_props, IDP_NewInt(i, "mmd_physics_rigid_index"));
    }
    IDProperty *type_props = IDP_GetPropertyTypeFromGroup(obj_props, "mmd_physics_rigid_type", IDP_INT);
    if (type_props != nullptr) {
      IDP_int_set(type_props, rigid.physics_type);
    }
    else {
      IDP_AddToGroup(obj_props, IDP_NewInt(rigid.physics_type, "mmd_physics_rigid_type"));
    }

    /* Register the object with the native rigid body settings. */
    const eRigidBodyOb_Type rbo_type = (rigid.physics_type == 0) ? RBO_TYPE_PASSIVE :
                                                                   RBO_TYPE_ACTIVE;
    RigidBodyOb *rbo = BKE_rigidbody_create_object(scene, ob, rbo_type);
    if (rbo == nullptr) {
      continue;
    }
    rbo->shape = mmd_shape_to_blender_shape(rigid.shape_type);
    rbo->mass = rigid.mass;
    rbo->friction = rigid.friction;
    rbo->restitution = rigid.restitution;
    rbo->lin_damping = rigid.linear_damping;
    rbo->ang_damping = rigid.angular_damping;
    /* Set every body's collision collections to the full range so bodies collide
     * with each other by default; per-pair non-collision is handled by the NCC
     * (non-collision constraint) objects below, exactly like mmd_tools. Blender's
     * native collision filter is `(col_groups_a & col_groups_b) != 0` (see the
     * RB filter callback in rb_bullet_api.cpp), so a shared bit means "may collide"
     * and a disable-collisions constraint removes individual pairs at narrowphase. */
    rbo->col_groups = 0xFFFF;
    if (rigid.physics_type == 0) {
      rbo->type = RBO_TYPE_PASSIVE;
      rbo->flag |= RBO_FLAG_KINEMATIC;
    }
    else {
      rbo->type = RBO_TYPE_ACTIVE;
    }
    /* Keep the physics body's initial transform in sync with the object so the
     * first evaluated state is correct even before the depsgraph fills in the
     * object matrix. Uses the WORLD rest matrix (definition carried into world). */
    float world_rest[4][4];
    rigid_body_world_rest_matrix(arm_world, rigid, world_rest);
    mat4_to_loc_quat(rbo->pos, rbo->orn, world_rest);

    /* Dynamic bodies: drive the bound bone with a Copy Transforms / Copy
     * Rotation constraint targeting this rigid-body object, so the simulated
     * rigid-body motion drives the bone (and the mesh). This is the mmd_tools
     * `mmd_tools_rigid_track` mechanism and replaces the keyframe transfer. */
    if (rigid.physics_type != 0 && rigid.bone_resolved && !rigid.blender_bone_name.empty()) {
      bPoseChannel *pchan = BKE_pose_channel_find_name(armature->pose,
                                                       rigid.blender_bone_name.c_str());
      if (pchan != nullptr) {
        add_bone_copy_constraint(armature, pchan, ob, rigid.physics_type == 2);
      }
    }

    rigid_objects[i] = ob;
    created_rigid_bodies++;
    DEG_id_tag_update_ex(bmain, &ob->id, ID_RECALC_TRANSFORM);
  }

  /* 4. Create one native joint (constraint) per MMD joint. */
  const int joint_count = int(definition.joints.size());
  int created_joints = 0;
  /* Map an ordered rigid-body pair (a<b) to the rigid-body constraint that
   * connects them, so the NCC pass can enable `disable_collisions` on joints
   * linking a "should-not-collide" pair (mmd_tools buildRigids behaviour). */
  using RigidPair = std::pair<int, int>;
  std::map<RigidPair, RigidBodyCon *> joint_constraint_for_pair;
  for (int i = 0; i < joint_count; i++) {
    const MMDJointDefinition &joint = definition.joints[i];
    if (joint.pmx_index != i) {
      continue;
    }
    if (joint.rigid_a_index < 0 || joint.rigid_a_index >= rigid_count ||
        joint.rigid_b_index < 0 || joint.rigid_b_index >= rigid_count)
    {
      report("MMD physics: joint[" + std::to_string(i) +
             "] has out-of-range rigid endpoints; skipped");
      continue;
    }
    Object *ob_a = rigid_objects[joint.rigid_a_index];
    Object *ob_b = rigid_objects[joint.rigid_b_index];
    if (ob_a == nullptr || ob_b == nullptr) {
      report("MMD physics: joint[" + std::to_string(i) +
             "] endpoints were not created; skipped");
      continue;
    }
    const std::string object_name = joint_object_name(joint, i);
    Object *joint_ob = BKE_object_add_only_object(bmain, OB_EMPTY, object_name.c_str());
    if (joint_ob == nullptr) {
      continue;
    }
    copy_v3_v3(joint_ob->loc, joint.position.data());
    joint_ob->rotmode = ROT_MODE_YXZ;
    copy_v3_v3(joint_ob->rot, joint.rotation.data());

    /* IMPORTANT: create the 6-DOF SPRING constraint FIRST and configure it, and
     * ONLY THEN add the object to `rbw->constraints`. Adding to that collection
     * triggers `BKE_rigidbody_main_collection_object_add`, which auto-creates a
     * plain `RBC_TYPE_FIXED` constraint on any object without one. If we added the
     * object to `rbw->constraints` first, that auto-FIXED constraint would make our
     * later `BKE_rigidbody_create_constraint(..., RBC_TYPE_6DOF_SPRING)` return
     * nullptr (the object already has a constraint), and every joint would end up
     * as a FIXED constraint with no spring/limits -> the cloth would have no
     * restoring force and free-fall. */
    RigidBodyCon *rbc = BKE_rigidbody_create_constraint(scene, joint_ob, RBC_TYPE_6DOF_SPRING);
    if (rbc == nullptr) {
      report("MMD physics: joint[" + std::to_string(i) +
             "] constraint could not be created; skipped");
      continue;
    }
    rbc->ob1 = ob_a;
    rbc->ob2 = ob_b;
    rbc->spring_type = RBC_SPRING_TYPE2; /* btGeneric6DofSpringConstraint2, the Blender
                                          * default that mmd_tools' constraint_add leaves in
                                          * place (SPRING1 caps damping at 1.0). */

    rbc->limit_lin_x_lower = joint.translation_min[0];
    rbc->limit_lin_x_upper = joint.translation_max[0];
    rbc->limit_lin_y_lower = joint.translation_min[1];
    rbc->limit_lin_y_upper = joint.translation_max[1];
    rbc->limit_lin_z_lower = joint.translation_min[2];
    rbc->limit_lin_z_upper = joint.translation_max[2];
    rbc->limit_ang_x_lower = joint.rotation_min[0];
    rbc->limit_ang_x_upper = joint.rotation_max[0];
    rbc->limit_ang_y_lower = joint.rotation_min[1];
    rbc->limit_ang_y_upper = joint.rotation_max[1];
    rbc->limit_ang_z_lower = joint.rotation_min[2];
    rbc->limit_ang_z_upper = joint.rotation_max[2];

    rbc->flag |= RBC_FLAG_USE_LIMIT_LIN_X;
    rbc->flag |= RBC_FLAG_USE_LIMIT_LIN_Y;
    rbc->flag |= RBC_FLAG_USE_LIMIT_LIN_Z;
    rbc->flag |= RBC_FLAG_USE_LIMIT_ANG_X;
    rbc->flag |= RBC_FLAG_USE_LIMIT_ANG_Y;
    rbc->flag |= RBC_FLAG_USE_LIMIT_ANG_Z;
    rbc->flag |= RBC_FLAG_USE_SPRING_X;
    rbc->flag |= RBC_FLAG_USE_SPRING_Y;
    rbc->flag |= RBC_FLAG_USE_SPRING_Z;
    rbc->flag |= RBC_FLAG_USE_SPRING_ANG_X;
    rbc->flag |= RBC_FLAG_USE_SPRING_ANG_Y;
    rbc->flag |= RBC_FLAG_USE_SPRING_ANG_Z;
    /* mmd_tools sets disable_collisions=False for joint constraints; the
     * BKE_rigidbody_create_constraint default enables it, so clear it. */
    rbc->flag &= ~RBC_FLAG_DISABLE_COLLISIONS;

    /* mmd_tools soft-constraint: cloth joints are anchored with a
     * Generic6DofSpringConstraint whose spring keeps the cloth from free-falling.
     * The PMX file's spring constants here are 0 (this model relies on the
     * mmd_tools "soft" default), so when a PMX spring value is 0 fall back to the
     * Blender default that mmd_tools' constraint_add leaves in place
     * (spring_stiffness = 10, spring_damping = 0.5). */
    const float kSoftSpringStiffness = 10.0f;
    const float kSoftSpringDamping = 0.5f;
    auto spring_stiffness = [](float v, float fallback) {
      return (v > 0.0f) ? v : fallback;
    };
    rbc->spring_stiffness_x = spring_stiffness(joint.spring_translation[0], kSoftSpringStiffness);
    rbc->spring_stiffness_y = spring_stiffness(joint.spring_translation[1], kSoftSpringStiffness);
    rbc->spring_stiffness_z = spring_stiffness(joint.spring_translation[2], kSoftSpringStiffness);
    rbc->spring_stiffness_ang_x = spring_stiffness(joint.spring_rotation[0], kSoftSpringStiffness);
    rbc->spring_stiffness_ang_y = spring_stiffness(joint.spring_rotation[1], kSoftSpringStiffness);
    rbc->spring_stiffness_ang_z = spring_stiffness(joint.spring_rotation[2], kSoftSpringStiffness);

    rbc->spring_damping_x = kSoftSpringDamping;
    rbc->spring_damping_y = kSoftSpringDamping;
    rbc->spring_damping_z = kSoftSpringDamping;
    rbc->spring_damping_ang_x = kSoftSpringDamping;
    rbc->spring_damping_ang_y = kSoftSpringDamping;
    rbc->spring_damping_ang_z = kSoftSpringDamping;

    /* Add the joint object to the sim constraint group collection only AFTER the
     * constraint exists and is configured. Adding it earlier would let
     * BKE_rigidbody_main_collection_object_add auto-create a plain FIXED constraint.
     * That auto-FIXED constraint would silently replace our configured spring/limits
     * (it has default values and no endpoints), so the joint would do nothing.
     * `BKE_rigidbody_create_constraint` only allocates and returns the RigidBodyCon;
     * it does NOT link it onto the object, so we must do that explicitly here.
     */
    joint_ob->rigidbody_constraint = rbc;
    if (rigid_collection != nullptr) {
      BKE_collection_object_add(bmain, rigid_collection, joint_ob);
    }
    BKE_collection_object_add(bmain, rbw->constraints, joint_ob);

    /* Remember the constraint that links this rigid-body pair so the NCC pass can
     * flip it to `disable_collisions` when the pair is a should-not-collide one. */
    const RigidPair key(std::min(joint.rigid_a_index, joint.rigid_b_index),
                        std::max(joint.rigid_a_index, joint.rigid_b_index));
    joint_constraint_for_pair[key] = rbc;

    created_joints++;
    DEG_id_tag_update_ex(bmain, &joint_ob->id, ID_RECALC_TRANSFORM);
  }

  /* 5. Build non-collision (NCC) constraints, replicating mmd_tools
   * Model.buildRigids + __createNonCollisionConstraint. For every rigid-body pair
   * that the PMX collision-group mask marks as "must not collide":
   *   - if the pair is connected by a joint, turn that joint's disable_collisions
   *     on (the joint is the collision gate), and
   *   - otherwise, if the bodies are close enough (the mmd_tools
   *     non_collision_distance_scale heuristic), create a GENERIC constraint that
   *     only disables collision (Bullet `addConstraint(con, disableCollisions)`,
   *     see RB_dworld_add_constraint in rb_bullet_api.cpp).
   */
  constexpr float kNonCollisionDistanceScale = 1.5f;
  int created_ncc = 0;
  int joint_ncc = 0;
  std::vector<RigidPair> non_collision_pairs;
  for (int a = 0; a < rigid_count; a++) {
    if (rigid_objects[a] == nullptr) {
      continue;
    }
    const uint8_t group_a = definition.rigid_bodies[a].collision_group;
    /* mmd_tools inverted convention: the plugin imports the raw 16-bit PMX
     * no_collision_group into a per-group bool vector with
     * `collision_group_mask[i] = (raw & (1<<i)) == 0`, i.e. a True entry means
     * "this body does NOT collide with group i", which happens exactly when the
     * raw bit i is ZERO. So "must not collide with group n" == `~no_collision_group`
     * bit n set. Using the raw no_collision_group (bit set = not collide) is the
     * OPPOSITE and makes the skirt avoid the body instead of itself. */
    const uint16_t mask_a = uint16_t(0xFFFFu & ~definition.rigid_bodies[a].no_collision_group);
    for (int b = a + 1; b < rigid_count; b++) {
      if (rigid_objects[b] == nullptr) {
        continue;
      }
      const uint8_t group_b = definition.rigid_bodies[b].collision_group;
      const uint16_t mask_b = uint16_t(0xFFFFu & ~definition.rigid_bodies[b].no_collision_group);
      /* mmd_tools semantics: a body does not collide with group i when bit i of
       * its "must-not-collide" mask (~no_collision_group) is set, so the pair is
       * "should-not-collide" if either body forbids the other's group. */
      const bool no_collide = ((mask_a & (uint16_t(1) << group_b)) != 0) ||
                              ((mask_b & (uint16_t(1) << group_a)) != 0);
      if (!no_collide) {
        continue;
      }
      const RigidPair key(a, b);
      const auto joint_it = joint_constraint_for_pair.find(key);
      if (joint_it != joint_constraint_for_pair.end()) {
        /* The pair is already rigidly tied by a joint: remove collision there. */
        joint_it->second->flag |= RBC_FLAG_DISABLE_COLLISIONS;
        joint_ncc++;
        continue;
      }
      /* mmd_tools creates an NCC object only for nearby bodies. */
      const float dx = rigid_positions[a][0] - rigid_positions[b][0];
      const float dy = rigid_positions[a][1] - rigid_positions[b][1];
      const float dz = rigid_positions[a][2] - rigid_positions[b][2];
      const float distance = std::sqrt(dx * dx + dy * dy + dz * dz);
      const float threshold = kNonCollisionDistanceScale * (rigid_ranges[a] + rigid_ranges[b]) *
                              0.5f;
      if (distance < threshold) {
        non_collision_pairs.emplace_back(a, b);
      }
    }
  }

  /* Create one GENERIC (6-DOF, no limits -> free) constraint per non-collision
   * pair. The constraint only carries `disable_collisions`; its limits are left
   * disabled so it does not bind the bodies' relative motion. */
  for (const RigidPair &pair : non_collision_pairs) {
    const int a = pair.first;
    const int b = pair.second;
    const std::string object_name = "NCC_" + std::to_string(a) + "_" + std::to_string(b);
    Object *ncc_ob = BKE_object_add_only_object(bmain, OB_EMPTY, object_name.c_str());
    if (ncc_ob == nullptr) {
      continue;
    }
    if (rigid_collection != nullptr) {
      BKE_collection_object_add(bmain, rigid_collection, ncc_ob);
    }
    BKE_collection_object_add(bmain, rbw->constraints, ncc_ob);

    RigidBodyCon *rbc = BKE_rigidbody_create_constraint(scene, ncc_ob, RBC_TYPE_6DOF);
    if (rbc == nullptr) {
      continue;
    }
    rbc->ob1 = rigid_objects[a];
    rbc->ob2 = rigid_objects[b];
    /* The whole point of an NCC object is to disable collision; setting the flag
     * makes Bullet's addConstraint(con, disableCollisions=true) skip the pair at
     * narrowphase while leaving the constraint itself free (no limits set). */
    rbc->flag |= RBC_FLAG_DISABLE_COLLISIONS;
    created_ncc++;
    DEG_id_tag_update_ex(bmain, &ncc_ob->id, ID_RECALC_TRANSFORM);
  }

  DEG_relations_tag_update(bmain);
  BKE_rigidbody_cache_reset(rbw);

  if (reports != nullptr) {
    BKE_reportf(reports,
                RPT_INFO,
                "MMD physics: built %d native rigid bodies, %d native joints, %d non-collision "
                "pairs (%d via joint disable, %d via NCC constraint)",
                created_rigid_bodies,
                created_joints,
                joint_ncc + created_ncc,
                joint_ncc,
                created_ncc);
  }

  return true;
}

bool sync_rigidbodies_to_bake_start(Scene *scene,
                                    Object *armature,
                                    const MMDPhysicsDefinition &definition,
                                    Depsgraph *depsgraph,
                                    RigidBodyWorld *rbw)
{
  if (scene == nullptr || armature == nullptr || rbw == nullptr ||
      armature->type != OB_ARMATURE || armature->pose == nullptr)
  {
    return false;
  }
  /* Mute the rigid-track Copy constraints so the bones evaluate purely from the
   * animation (VMD) at the bake start frame, letting us read each bone's true
   * animated pose (without the constraint forcing bone == rigid body). */
  for (bPoseChannel *pchan = static_cast<bPoseChannel *>(armature->pose->chanbase.first);
       pchan != nullptr;
       pchan = pchan->next)
  {
    for (bConstraint *con = static_cast<bConstraint *>(pchan->constraints.first);
         con != nullptr;
         con = con->next)
    {
      if (STREQ(con->name, "mmd_tools_rigid_track")) {
        con->flag |= CONSTRAINT_OFF;
      }
    }
  }
  /* Evaluate the depsgraph at the bake start frame so the bones carry the
   * VMD-driven pose at `sfra` (used below to place the dynamic bodies). */
  scene->r.cfra = scene->r.sfra;
  if (depsgraph != nullptr) {
    BKE_scene_graph_update_for_newframe(depsgraph);
  }

  /* The Armature (model-root) world matrix. The bones' `pose_mat` lives in the
   * armature's local space, so each bone's world matrix is arm_world @ pose_mat. */
  float arm_world[4][4];
  compute_object_world_matrix(armature, arm_world);

  /* Read the evaluated armature so the bone pose reflects the animated frame
   * (the original `armature->pose` is not updated by the depsgraph). */
  Object *arm_eval = DEG_get_evaluated(depsgraph, armature);
  bPose *pose_eval = (arm_eval != nullptr) ? arm_eval->pose : armature->pose;
  if (pose_eval == nullptr) {
    return false;
  }

  const int rigid_count = int(definition.rigid_bodies.size());
  FOREACH_COLLECTION_OBJECT_RECURSIVE_BEGIN (rbw->group, ob) {
    if (ob->type != OB_MESH || ob->rigidbody_object == nullptr) {
      continue;
    }
    IDProperty *props = ob->id.system_properties;
    if (props == nullptr) {
      continue;
    }
    IDProperty *idx_prop = IDP_GetPropertyTypeFromGroup(props, "mmd_physics_rigid_index", IDP_INT);
    if (idx_prop == nullptr) {
      continue;
    }
    const int ridx = IDP_int_get(idx_prop);
    if (ridx < 0 || ridx >= rigid_count) {
      continue;
    }
    const MMDRigidBodyDefinition &rigid = definition.rigid_bodies[ridx];
    if (rigid.physics_type == 0 || !rigid.bone_resolved || rigid.blender_bone_name.empty()) {
      continue;
    }
    bPoseChannel *pchan = BKE_pose_channel_find_name(pose_eval, rigid.blender_bone_name.c_str());
    if (pchan == nullptr) {
      continue;
    }
    /* World matrix of the bone at the bake start frame. */
    float bone_world[4][4];
    mul_m4_m4m4(bone_world, arm_world, pchan->pose_mat);

    float loc[3], quat[4];
    mat4_to_loc_quat(loc, quat, bone_world);

    /* Seed the COLLIDER OBJECT (not just rbo->pos/orn): Blender builds each Bullet
     * body from `ob->object_to_world()` (rigidbody.cc:`rigidbody_validate_sim_object`),
     * so writing only `rbo->pos/orn` is ignored for initial body placement. Setting
     * the object's loc/rot makes the depsgraph produce the correct world matrix when
     * the sim world is (re)built at the bake start. */
    copy_v3_v3(ob->loc, loc);
    ob->rotmode = ROT_MODE_YXZ;
    quat_to_eulO(ob->rot, EULER_ORDER_YXZ, quat);
    /* Keep the rigid-body settings copy in sync too. */
    copy_v3_v3(ob->rigidbody_object->pos, loc);
    copy_v4_v4(ob->rigidbody_object->orn, quat);
    /* Tag the object for transform re-evaluation so the depsgraph's evaluated
     * copy (whose object_to_world() builds the Bullet body) picks up the seed. */
    DEG_id_tag_update(&ob->id, ID_RECALC_TRANSFORM);
  }
  FOREACH_COLLECTION_OBJECT_RECURSIVE_END;

  /* Unmute and re-evaluate so the bone constraints pick the seeded bodies back up. */
  for (bPoseChannel *pchan = static_cast<bPoseChannel *>(armature->pose->chanbase.first);
       pchan != nullptr;
       pchan = pchan->next)
  {
    for (bConstraint *con = static_cast<bConstraint *>(pchan->constraints.first);
         con != nullptr;
         con = con->next)
    {
      if (STREQ(con->name, "mmd_tools_rigid_track")) {
        con->flag &= ~CONSTRAINT_OFF;
      }
    }
  }
  if (depsgraph != nullptr) {
    BKE_scene_graph_update_for_newframe(depsgraph);
  }
  /* Invalidate the baked cache and force the sim world to rebuild so every Bullet
   * body is (re)created from the freshly seeded object transforms. This is what
   * makes the bake start with the bodies aligned to the animated skeleton instead
   * of the rest pose (which is the cause of the cloth falling). */
  BKE_rigidbody_cache_reset(rbw);
  return true;
}

bool bake_rigidbody_physics_to_bones(Main *bmain,
                                     Scene *scene,
                                     Object *armature,
                                     const MMDPhysicsDefinition &definition,
                                     ReportList *reports,
                                     Depsgraph *depsgraph)
{
  if (bmain == nullptr || scene == nullptr || armature == nullptr ||
      armature->type != OB_ARMATURE || armature->data == nullptr)
  {
    return false;
  }
  auto report = [reports](const std::string &message) {
    if (reports != nullptr) {
      BKE_report(reports, RPT_WARNING, message.c_str());
    }
  };
  bPose *pose = armature->pose;
  if (pose == nullptr) {
    return false;
  }
  /* Ensure the bones' rest matrices (arm_mat) are computed on the original
   * Armature data so the transfer can derive each bone's rest world matrix.
   * `BKE_armature_where_is` populates `Bone::arm_mat`; without it the values
   * can be all-zero when reading the original (non-evaluated) Armature. */
  if (armature->data != nullptr) {
    BKE_armature_where_is(reinterpret_cast<bArmature *>(armature->data));
  }
  RigidBodyWorld *rbw = BKE_rigidbody_get_world(scene);
  if (rbw == nullptr || rbw->group == nullptr) {
    report("MMD physics: no native RigidBodyWorld to transfer from");
    return false;
  }

  /* Collect dynamic rigid-body objects that are bound to a bone. We need the
   * baked object (rbo) and the bone to write to. */
  struct BoneBinding {
    Object *ob = nullptr;
    bPoseChannel *pchan = nullptr;
    float R_rest[4][4] = {};
    float bone_offset[4][4] = {}; /* R_rest^-1 @ B_rest_world (constant) */
    float arm_world[4][4] = {};   /* reliable model-root world (parent chain) */
    float bone_arm_mat[4][4] = {}; /* bone rest matrix in armature space */
  };
  std::vector<BoneBinding> bindings;

  const int rigid_count = int(definition.rigid_bodies.size());
  /* Guard against duplicate rigid-body objects (e.g. a pre-idempotency build had
   * doubled them): only bind the first object for each rigid index. */
  std::set<int> seen_indices;

  FOREACH_COLLECTION_OBJECT_RECURSIVE_BEGIN (rbw->group, ob) {
    if (ob->type != OB_MESH || ob->rigidbody_object == nullptr) {
      continue;
    }
    IDProperty *props = ob->id.system_properties;
    if (props == nullptr) {
      continue;
    }
    IDProperty *idx_prop = IDP_GetPropertyTypeFromGroup(props, "mmd_physics_rigid_index", IDP_INT);
    IDProperty *type_prop = IDP_GetPropertyTypeFromGroup(props, "mmd_physics_rigid_type", IDP_INT);
    if (idx_prop == nullptr || type_prop == nullptr) {
      continue;
    }
    const int ridx = IDP_int_get(idx_prop);
    const int ptype = IDP_int_get(type_prop);
    if (ridx < 0 || ridx >= rigid_count) {
      continue;
    }
    if (ptype == 0) {
      /* Static bodies are bone-parented and follow the skeleton already. */
      continue;
    }
    if (seen_indices.find(ridx) != seen_indices.end()) {
      continue; /* Already bound from an earlier duplicate object. */
    }
    const MMDRigidBodyDefinition &rigid = definition.rigid_bodies[ridx];
    if (!rigid.bone_resolved || rigid.blender_bone_name.empty()) {
      continue;
    }
    bPoseChannel *pchan = BKE_pose_channel_find_name(pose, rigid.blender_bone_name.c_str());
    if (pchan == nullptr) {
      continue;
    }
    const Bone *bone = pchan->bone_get(*armature);
    if (bone == nullptr) {
      continue;
    }

    BoneBinding binding;
    binding.ob = ob;
    binding.pchan = pchan;
    /* Rigid body rest WORLD matrix (definition carried into world). */
    float arm_world[4][4];
    compute_object_world_matrix(armature, arm_world);
    copy_m4_m4(binding.arm_world, arm_world);
    compute_bone_rest_armature_matrix(bone, binding.bone_arm_mat);
    float r_rest[4][4];
    rigid_body_world_rest_matrix(arm_world, rigid, r_rest);
    copy_m4_m4(binding.R_rest, r_rest);
    /* Bone rest world matrix and the constant offset from the rigid-body rest. */
    float b_rest[4][4];
    mul_m4_m4m4(b_rest, arm_world, binding.bone_arm_mat);
    float r_inv[4][4];
    invert_m4_m4(r_inv, r_rest);
    mul_m4_m4m4(binding.bone_offset, r_inv, b_rest);
    seen_indices.insert(ridx);
    bindings.push_back(binding);
  }
  FOREACH_COLLECTION_OBJECT_RECURSIVE_END;

  if (bindings.empty()) {
    report("MMD physics: no dynamic rigid bodies bound to bones to transfer");
    return false;
  }

  /* Ensure the armature has an Action/slot/channelbag to write keyframes into. */
  AnimData *adt = BKE_animdata_from_id(&armature->id);
  if (adt == nullptr) {
    adt = BKE_animdata_ensure_id(&armature->id);
  }
  if (adt == nullptr) {
    report("MMD physics: no AnimData on the Armature");
    return false;
  }
  animrig::Action *action = nullptr;
  animrig::Slot *slot = nullptr;
  bool created_action = false;
  if (adt->action != nullptr) {
    action = &adt->action->wrap();
    slot = action->slot_for_handle(adt->slot_handle);
  }
  if (slot == nullptr) {
    if (action == nullptr) {
      action = &animrig::action_add(*bmain, "MMD_PhysicsBake");
      created_action = true;
      adt->action = action; /* animrig::Action derives from bAction. */
    }
    slot = &action->slot_add_for_id(armature->id);
    adt->slot_handle = slot->handle;
  }
  action->layer_keystrip_ensure();
  if (action->layers().is_empty() || action->layer(0)->strips().is_empty()) {
    if (created_action) {
      BKE_id_free(bmain, &action->id);
    }
    return false;
  }
  animrig::Strip &strip = *action->layer(0)->strip(0);
  animrig::StripKeyframeData &strip_data = strip.data<animrig::StripKeyframeData>(*action);
  animrig::Channelbag *channelbag = strip_data.channelbag_for_slot(*slot);
  if (channelbag == nullptr) {
    channelbag = &strip_data.channelbag_for_slot_add(*slot);
  }

  const animrig::KeyframeSettings settings = {BEZT_KEYTYPE_KEYFRAME, HD_AUTO_ANIM, BEZT_IPO_LIN};

  const int start = scene->r.sfra;
  const int end = scene->r.efra;
  int keyframed_bones = 0;

  for (const BoneBinding &binding : bindings) {
    const Bone *bone = binding.pchan->bone_get(*armature);
    if (bone == nullptr) {
      continue;
    }
    /* Force quaternion rotation so the baked rotation keys drive the bone. */
    binding.pchan->rotmode = ROT_MODE_QUAT;

    char esc[128] = {};
    BLI_str_escape(esc, bone->name, sizeof(esc));
    const std::string base = std::string("pose.bones[\"") + esc + "\"]";
    const std::string loc_path = base + ".location";
    const std::string rot_path = base + ".rotation_quaternion";
    const std::string scale_path = base + ".scale";

    std::vector<FCurve *> loc_curves;
    std::vector<FCurve *> rot_curves;
    std::vector<FCurve *> scale_curves;
    for (int c = 0; c < 3; c++) {
      animrig::FCurveDescriptor d;
      d.rna_path = loc_path;
      d.array_index = c;
      d.prop_type = PROP_FLOAT;
      d.prop_subtype = PROP_NONE;
      FCurve &f = channelbag->fcurve_ensure(nullptr, d);
      loc_curves.push_back(&f);
    }
    for (int c = 0; c < 4; c++) {
      animrig::FCurveDescriptor d;
      d.rna_path = rot_path;
      d.array_index = c;
      d.prop_type = PROP_FLOAT;
      d.prop_subtype = PROP_NONE;
      FCurve &f = channelbag->fcurve_ensure(nullptr, d);
      rot_curves.push_back(&f);
    }
    for (int c = 0; c < 3; c++) {
      animrig::FCurveDescriptor d;
      d.rna_path = scale_path;
      d.array_index = c;
      d.prop_type = PROP_FLOAT;
      d.prop_subtype = PROP_NONE;
      FCurve &f = channelbag->fcurve_ensure(nullptr, d);
      scale_curves.push_back(&f);
    }

    for (int frame = start; frame <= end; frame++) {
      scene->r.cfra = frame;
      if (depsgraph != nullptr) {
        BKE_scene_graph_update_for_newframe(depsgraph);
      }

      /* Baked rigid-body world matrix. `object_to_world()` on these helper
       * objects can be the default identity (they are not necessarily in a
       * depsgraph-evaluated view-layer collection), so read the authoritative
       * simulation transform from the Bullet body (`rbo->pos/orn`, world space). */
      RigidBodyOb *rbo = binding.ob->rigidbody_object;
      float r_now[4][4];
      copy_v3_v3(r_now[3], rbo->pos);
      quat_to_mat4(r_now, rbo->orn);
      /* B_now = R_now @ (R_rest^-1 @ B_rest_world) = R_now @ bone_offset. This
       * applies the rigid body's displacement-from-rest to the bone's rest. */
      float b_now[4][4];
      mul_m4_m4m4(b_now, r_now, binding.bone_offset);
      /* Convert world -> armature/pose space using the RELIABLE parent-chain
       * world matrix (NOT `BKE_armature_mat_world_to_pose`, which multiplies by
       * inverse(ob->object_to_world()) that can be a zero/un-evaluated matrix
       * during the transfer and corrupt every result). */
      float arm_world_inv[4][4];
      invert_m4_m4(arm_world_inv, binding.arm_world);
      float pose_arm[4][4];
      /* Match BKE_armature_mat_world_to_pose's convention: out = in @ inv(object_obmat). */
      mul_m4_m4m4(pose_arm, b_now, arm_world_inv);
      /* pose -> bone-local basis, then apply to the pose bone. */
      bke::PChanBoneConst chan_const(binding.pchan, binding.pchan->bone_get(*armature));
      float basis[4][4];
      BKE_armature_mat_pose_to_bone(chan_const, pose_arm, basis);
      BKE_pchan_apply_mat4(binding.pchan, basis, true);

      /* Insert keyframes on this bone's channels. */
      for (int c = 0; c < 3; c++) {
        animrig::insert_vert_fcurve(
            loc_curves[c], {float(frame), binding.pchan->loc[c]}, settings, INSERTKEY_FAST);
      }
      for (int c = 0; c < 4; c++) {
        animrig::insert_vert_fcurve(
            rot_curves[c], {float(frame), binding.pchan->quat[c]}, settings, INSERTKEY_FAST);
      }
      for (int c = 0; c < 3; c++) {
        animrig::insert_vert_fcurve(
            scale_curves[c], {float(frame), binding.pchan->scale[c]}, settings, INSERTKEY_FAST);
      }
    }
    for (FCurve *fcurve : loc_curves) {
      BKE_fcurve_handles_recalc(*fcurve);
    }
    for (FCurve *fcurve : rot_curves) {
      BKE_fcurve_handles_recalc(*fcurve);
    }
    for (FCurve *fcurve : scale_curves) {
      BKE_fcurve_handles_recalc(*fcurve);
    }
    keyframed_bones++;
  }

  DEG_id_tag_update(&armature->id, ID_RECALC_GEOMETRY | ID_RECALC_ANIMATION);
  if (reports != nullptr) {
    BKE_reportf(reports,
                RPT_INFO,
                "MMD physics: baked %d dynamic rigid bodies (frames %d-%d) onto Armature bones",
                keyframed_bones,
                start,
                end);
  }
  return keyframed_bones > 0;
}

}  // namespace blender::mmd_physics
