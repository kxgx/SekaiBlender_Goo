"""
MMD Tools — Goo native compatibility layer.

This is Goo's built-in, minimal `mmd_tools` Python module. It exists so that
plugins and scripts which depend on mmd_tools (e.g. `blander_ue5_link.mmd`)
can *recognize* mmd_tools in Goo and read its Python-visible data model.

Design (user-confirmed): the heavy lifting is done by Goo's native C++ MMD
kernel (io/pmx, io/vmd, io_mmd_physics/bake/render). This module only:

  * registers the mmd_tools RNA data model so `getattr(obj, "mmd_type")`,
    `obj.mmd_root`, `obj.mmd_bone`, `pose.bones[x].mmd_ik_toggle` and
    `Material.mmd_material` resolve as real attributes (not IDProperties),
    which is what `pyrna_struct_getattro` requires; and
  * provides the `bpy.ops.mmd_tools.*` namespace (which Goo also registers
    natively in C++ as `MMD_TOOLS_OT_*` operators) and its panel/menu surface.

None of the heavy MMD parsing/IK/physics is reimplemented here.
"""

import bpy

bl_info = {  # recognised by addon_utils / preferences.addons
    "name": "MMD Tools",
    "author": "Goo (native mmd_tools compatibility layer)",
    "version": (0, 1, 0),
    "blender": (5, 3, 0),
    "location": "Object Properties > MMD; Tool sidebar",
    "description": "Native MMD tooling compatibility layer (mmd_tools identity)",
    "category": "MMD",
}

MMD_TYPE_ITEMS = [
    ("NONE", "None", "", 1),
    ("ROOT", "Root", "", 2),
    ("RIGID_GRP_OBJ", "Rigid Body Grp Empty", "", 3),
    ("JOINT_GRP_OBJ", "Joint Grp Empty", "", 4),
    ("TEMPORARY_GRP_OBJ", "Temporary Grp Empty", "", 5),
    ("PLACEHOLDER", "Place Holder", "", 6),
    ("CAMERA", "Camera", "", 21),
    ("JOINT", "Joint", "", 22),
    ("RIGID_BODY", "Rigid body", "", 23),
    ("LIGHT", "Light", "", 24),
    ("TRACK_TARGET", "Track Target", "", 51),
    ("NON_COLLISION_CONSTRAINT", "Non Collision Constraint", "", 52),
    ("SPRING_CONSTRAINT", "Spring Constraint", "", 53),
    ("SPRING_GOAL", "Spring Goal", "", 54),
]

_REGISTERED = False


class MMDRoot(bpy.types.PropertyGroup):
    """Minimal Goo-native MMD root data model (mmd_tools compatible subset)."""

    name: bpy.props.StringProperty(name="Name", default="")
    name_e: bpy.props.StringProperty(name="Name (EN)", default="")
    comment_text: bpy.props.StringProperty(name="Comment", default="")
    comment_e_text: bpy.props.StringProperty(name="Comment (EN)", default="")

    show_meshes: bpy.props.BoolProperty(name="Show Meshes", default=True)
    show_armature: bpy.props.BoolProperty(name="Show Armature", default=True)
    show_rigid_bodies: bpy.props.BoolProperty(name="Show Rigid Bodies", default=True)
    show_joints: bpy.props.BoolProperty(name="Show Joints", default=True)
    show_temporary_objects: bpy.props.BoolProperty(name="Show Temp Objects", default=True)

    show_names_of_rigid_bodies: bpy.props.BoolProperty(
        name="Show Names of Rigid Bodies", default=False)
    show_names_of_joints: bpy.props.BoolProperty(name="Show Names of Joints", default=False)
    show_japanese_name: bpy.props.BoolProperty(name="Show Japanese Name", default=True)
    show_english_name: bpy.props.BoolProperty(name="Show English Name", default=True)

    use_toon_texture: bpy.props.BoolProperty(name="Use Toon Texture", default=True)
    use_sphere_texture: bpy.props.BoolProperty(name="Use Sphere Texture", default=True)
    use_sdef: bpy.props.BoolProperty(name="Use SDEF", default=True)
    use_property_driver: bpy.props.BoolProperty(name="Use Property Driver", default=False)
    is_built: bpy.props.BoolProperty(name="Is Built", default=False)

    ik_loop_factor: bpy.props.IntProperty(name="IK Loop Factor", default=1, min=1, max=100)

    active_rigidbody_index: bpy.props.IntProperty(name="Active Rigidbody", default=0)
    active_joint_index: bpy.props.IntProperty(name="Active Joint", default=0)
    active_mesh_index: bpy.props.IntProperty(name="Active Mesh", default=0)
    active_bone_index: bpy.props.IntProperty(name="Active Bone", default=0)


class MMDBone(bpy.types.PropertyGroup):
    """Minimal Goo-native MMD bone data model (mmd_tools compatible subset)."""

    name_j: bpy.props.StringProperty(name="Name (JP)", default="")
    name_e: bpy.props.StringProperty(name="Name (EN)", default="")
    bone_id: bpy.props.IntProperty(name="Bone ID", default=-1)

    transform_order: bpy.props.IntProperty(name="Transform Order", default=0)
    is_controllable: bpy.props.BoolProperty(name="Controllable", default=True)
    transform_after_dynamics: bpy.props.BoolProperty(name="Transform After Dynamics", default=False)

    is_tip: bpy.props.BoolProperty(name="Is Tip", default=False)

    enabled_fixed_axis: bpy.props.BoolProperty(name="Enable Fixed Axis", default=False)
    fixed_axis: bpy.props.FloatVectorProperty(name="Fixed Axis", size=3, default=(0, 0, 1))
    enabled_local_axes: bpy.props.BoolProperty(name="Enable Local Axes", default=False)
    local_axis_x: bpy.props.FloatVectorProperty(name="Local X", size=3, default=(1, 0, 0))
    local_axis_z: bpy.props.FloatVectorProperty(name="Local Z", size=3, default=(0, 0, 1))

    has_additional_rotation: bpy.props.BoolProperty(name="Additional Rotation", default=False)
    has_additional_location: bpy.props.BoolProperty(name="Additional Location", default=False)
    additional_transform_bone: bpy.props.StringProperty(name="Additional Transform", default="")
    additional_transform_bone_id: bpy.props.IntProperty(name="Additional Transform ID", default=-1)
    additional_transform_influence: bpy.props.FloatProperty(
        name="Additional Influence", default=1.0, min=0.0, max=1.0)

    display_connection_bone: bpy.props.StringProperty(name="Display Connection", default="")
    display_connection_bone_id: bpy.props.IntProperty(name="Display Connection ID", default=-1)


class MMDMaterial(bpy.types.PropertyGroup):
    """Minimal Goo-native MMD material data model."""

    name_j: bpy.props.StringProperty(name="Name (JP)", default="")
    name_e: bpy.props.StringProperty(name="Name (EN)", default="")
    comment: bpy.props.StringProperty(name="Comment", default="")


class MMDCamera(bpy.types.PropertyGroup):
    """Minimal Goo-native MMD camera data model."""

    angle: bpy.props.FloatProperty(name="Angle", subtype="ANGLE", default=0.0)
    is_perspective: bpy.props.BoolProperty(name="Perspective", default=True)


class MMDRigidBody(bpy.types.PropertyGroup):
    name_j: bpy.props.StringProperty(name="Name (JP)", default="")
    name_e: bpy.props.StringProperty(name="Name (EN)", default="")
    collision_group_number: bpy.props.IntProperty(name="Collision Group", default=0, min=0, max=15)


class MMDJoint(bpy.types.PropertyGroup):
    name_j: bpy.props.StringProperty(name="Name (JP)", default="")
    name_e: bpy.props.StringProperty(name="Name (EN)", default="")


def _register_groups():
    """Register the PropertyGroup subclasses so they can be used as
    PointerProperty(type=...) targets (Blender requires them registered first)."""
    for cls in (MMDRoot, MMDBone, MMDMaterial, MMDCamera, MMDRigidBody, MMDJoint):
        try:
            bpy.utils.register_class(cls)
        except (ValueError, RuntimeError):  # already registered
            pass


def _unregister_groups():
    for cls in (MMDRoot, MMDBone, MMDMaterial, MMDCamera, MMDRigidBody, MMDJoint):
        try:
            bpy.utils.unregister_class(cls)
        except (ValueError, RuntimeError):
            pass


def register():
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTERED = True

    # PropertyGroup targets must be registered before being used as PointerProperty targets.
    _register_groups()

    # Register the RNA data model (required so `getattr`/`obj.mmd_root` works).
    if not hasattr(bpy.types.Object, "mmd_type"):
        bpy.types.Object.mmd_type = bpy.props.EnumProperty(
            name="Type",
            description="Internal MMD type of this object (DO NOT CHANGE IT DIRECTLY)",
            default="NONE",
            items=MMD_TYPE_ITEMS,
        )
        bpy.types.Object.mmd_root = bpy.props.PointerProperty(type=MMDRoot)
        bpy.types.Object.mmd_rigid = bpy.props.PointerProperty(type=MMDRigidBody)
        bpy.types.Object.mmd_joint = bpy.props.PointerProperty(type=MMDJoint)
        bpy.types.Object.mmd_camera = bpy.props.PointerProperty(type=MMDCamera)

    if not hasattr(bpy.types.PoseBone, "mmd_ik_toggle"):
        bpy.types.PoseBone.mmd_ik_toggle = bpy.props.BoolProperty(
            name="IK Toggle",
            description="Temporarily enable/disable MMD IK on this bone",
            default=True,
        )
        bpy.types.PoseBone.mmd_bone = bpy.props.PointerProperty(type=MMDBone)

    if not hasattr(bpy.types.Material, "mmd_material"):
        bpy.types.Material.mmd_material = bpy.props.PointerProperty(type=MMDMaterial)


def unregister():
    global _REGISTERED
    if not _REGISTERED:
        return
    _REGISTERED = False

    for name in ("mmd_type", "mmd_root", "mmd_rigid", "mmd_joint", "mmd_camera"):
        if hasattr(bpy.types.Object, name):
            try:
                delattr(bpy.types.Object, name)
            except (AttributeError, TypeError):
                pass
    for name in ("mmd_ik_toggle", "mmd_bone"):
        if hasattr(bpy.types.PoseBone, name):
            try:
                delattr(bpy.types.PoseBone, name)
            except (AttributeError, TypeError):
                pass
    if hasattr(bpy.types.Material, "mmd_material"):
        try:
            delattr(bpy.types.Material, "mmd_material")
        except (AttributeError, TypeError):
            pass

    _unregister_groups()


if __name__ == "__main__":
    register()
