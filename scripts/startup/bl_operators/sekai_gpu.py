# SPDX-FileCopyrightText: 2026 MuAnChen
#
# SPDX-License-Identifier: GPL-2.0-or-later

"""SekaiBlender GPU 加速（R8-GPU）。

- NVIDIA 显卡自动启用 Cycles CUDA 设备（渲染走 CUDA），Vulkan 继续用于
  视口（EEVEE/GooEngine）与 GPU CCD 烘焙，两者同时可用。
- AMD 自动启用 HIP（若构建/驱动支持），否则视口与烘焙仍走 Vulkan。
- 渲染菜单提供 "GPU 渲染（图像/动画）" 按钮：优先 Cycles GPU，
  无可用 GPU 设备时回退当前引擎（GooEngine/EEVEE 本身即 GPU）。
- 加载 .blend 时自动执行一次设备设置（load_post）。
"""

import bpy
from bpy.types import Operator


def _vendor():
    try:
        import gpu
        return (gpu.platform.vendor_get() or "").lower()
    except Exception:
        return ""


def _cycles_prefs():
    addon = bpy.context.preferences.addons.get("cycles")
    return getattr(addon, "preferences", None) if addon else None


def _preferred_types(vendor):
    if "nvidia" in vendor:
        # OptiX 构建未开启时 get_devices_for_type 返回空，自然回退 CUDA。
        return ("OPTIX", "CUDA")
    if "advanced micro" in vendor or "amd" in vendor or "ati" in vendor:
        return ("HIP",)
    if "intel" in vendor:
        return ("ONEAPI",)
    return ("CUDA", "HIP")


def gpu_auto_setup():
    """启用厂商最优的 Cycles GPU 设备。返回 (device_type, enabled_count) 或 None。"""
    cprefs = _cycles_prefs()
    if cprefs is None:
        return None
    for dtype in _preferred_types(_vendor()):
        try:
            devices = cprefs.get_devices_for_type(dtype)
        except Exception:
            devices = []
        usable = [d for d in devices if d.type == dtype]
        if not usable:
            continue
        enabled = 0
        for d in usable:
            if not d.use:
                d.use = True
            enabled += 1
        cprefs.compute_device_type = dtype
        try:
            cprefs.update_device_entries(usable)
        except Exception:
            pass
        # 所有 Cycles 场景默认走 GPU 设备。
        for scene in bpy.data.scenes:
            if scene.render.engine == "CYCLES":
                scene.cycles.device = "GPU"
        return dtype, enabled
    return None


def _load_post(_dummy):
    # 每次打开 .blend 自动检测并启用最优 GPU 后端（幂等）。
    gpu_auto_setup()


class SEKAI_OT_gpu_auto_setup(Operator):
    bl_idname = "sekai.gpu_auto_setup"
    bl_label = "GPU 加速设置"
    bl_description = (
        "自动检测显卡并启用最优 GPU 后端：NVIDIA 使用 CUDA 渲染 + Vulkan "
        "视口/烘焙同时加速；AMD 使用 HIP/Vulkan"
    )

    def execute(self, context):
        res = gpu_auto_setup()
        if res:
            self.report(
                {"INFO"},
                "GPU 加速已启用：{} x {} 设备；Vulkan 用于视口与 GPU CCD 烘焙".format(
                    res[0], res[1]
                ),
            )
        else:
            self.report(
                {"WARNING"},
                "未找到可用的 Cycles GPU 设备；视口与 GPU 烘焙仍使用 Vulkan/OpenGL",
            )
        return {"FINISHED"}


class SEKAI_OT_gpu_render(Operator):
    bl_idname = "sekai.gpu_render"
    bl_label = "GPU 渲染"
    bl_description = (
        "使用 GPU 渲染：优先 Cycles CUDA/HIP；无可用 GPU 设备时回退当前引擎 "
        "（GooEngine/EEVEE 本身即 GPU 渲染）。渲染完成后恢复原引擎设置"
    )

    animation: bpy.props.BoolProperty(name="Animation", default=False)
    use_cycles: bpy.props.BoolProperty(
        name="Use Cycles GPU",
        default=True,
        description="优先切换到 Cycles 并使用 GPU 设备渲染",
    )

    def execute(self, context):
        scene = context.scene
        res = gpu_auto_setup()
        old_engine = scene.render.engine
        old_device = scene.cycles.device if scene.render.engine == "CYCLES" else ""
        # 有场景摄像机时按摄像机渲染（避免渲染视口当前视角，例如放大到腿部的
        # 视角会渲染出"满屏腿"）；没有摄像机才回退视口渲染。
        use_camera = scene.camera is not None
        try:
            if self.use_cycles and res is not None and scene.render.engine != "CYCLES":
                scene.render.engine = "CYCLES"
            if scene.render.engine == "CYCLES" and res is not None:
                scene.cycles.device = "GPU"
            bpy.ops.render.render(
                "INVOKE_DEFAULT",
                animation=self.animation,
                use_viewport=not use_camera,
                use_camera=use_camera,
                write_still=True,
            )
            if not use_camera:
                self.report(
                    {"INFO"},
                    "场景没有摄像机，已按视口渲染；导入 VMD 摄像机或添加摄像机后按场景相机渲染",
                )
        finally:
            if scene.render.engine != old_engine:
                scene.render.engine = old_engine
            elif old_device and scene.cycles.device != old_device:
                scene.cycles.device = old_device
        return {"FINISHED"}


classes = (
    SEKAI_OT_gpu_auto_setup,
    SEKAI_OT_gpu_render,
)


def register():
    bpy.app.handlers.load_post.append(_load_post)


def unregister():
    if _load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(_load_post)
