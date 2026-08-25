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
        # 所有 Cycles 场景默认走 GPU 设备 + GPU 降噪（OIDN CUDA/HIP 设备）。
        for scene in bpy.data.scenes:
            if scene.render.engine == "CYCLES":
                scene.cycles.device = "GPU"
                try:
                    scene.cycles.denoising_use_gpu = True
                    scene.cycles.preview_denoising_use_gpu = True
                except Exception:
                    pass
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
        "使用 GPU 渲染（默认 Goo Engine/EEVEE）。开启\"速度优先\"时临时套用"
        "EEVEE 加速预设（关闭光线追踪/运动模糊/过扫描、半分辨率阴影、降采样数），"
        "渲染完成后恢复原设置"
    )

    animation: bpy.props.BoolProperty(name="Animation", default=False)
    use_cycles: bpy.props.BoolProperty(
        name="Use Cycles GPU",
        default=False,
        description="优先切换到 Cycles 并使用 GPU 设备渲染（默认用当前引擎——"
        "Goo Engine/EEVEE 本身即 GPU 渲染）",
    )
    fast: bpy.props.BoolProperty(
        name="Fast",
        default=True,
        description="速度优先：临时套用 EEVEE 加速预设（关光追/运动模糊/过扫描、"
        "半分辨率阴影、16 渲染采样），完成后恢复",
    )

    # EEVEE（Goo Engine）速度预设：光追是最大的 GPU 时间消耗项。
    _EEVEE_FAST_KEYS = (
        ("use_raytracing", False),
        ("shadow_resolution_scale", 0.5),
        ("taa_render_samples", 16),
        ("use_overscan", False),
        ("use_volumetric_shadows", False),
    )

    def execute(self, context):
        scene = context.scene
        res = gpu_auto_setup()
        old_engine = scene.render.engine
        old_device = scene.cycles.device if scene.render.engine == "CYCLES" else ""
        # 有场景摄像机时按摄像机渲染（避免渲染视口当前视角，例如放大到腿部的
        # 视角会渲染出"满屏腿"）；没有摄像机才回退视口渲染。
        use_camera = scene.camera is not None
        # 速度预设：保存 → 套用 → 渲染后恢复。
        saved_eevee = {}
        old_motion_blur = scene.render.use_motion_blur
        applied_fast = False
        try:
            if self.use_cycles and res is not None and scene.render.engine != "CYCLES":
                scene.render.engine = "CYCLES"
            if scene.render.engine == "CYCLES" and res is not None:
                scene.cycles.device = "GPU"
            if self.fast and scene.render.engine != "CYCLES":
                eevee = scene.eevee
                for key, value in self._EEVEE_FAST_KEYS:
                    if hasattr(eevee, key):
                        saved_eevee[key] = getattr(eevee, key)
                        setattr(eevee, key, value)
                if old_motion_blur:
                    scene.render.use_motion_blur = False
                applied_fast = True
            bpy.ops.render.render(
                "INVOKE_DEFAULT",
                animation=self.animation,
                use_viewport=not use_camera,
                write_still=True,
            )
            if not use_camera:
                self.report(
                    {"INFO"},
                    "场景没有摄像机，已按视口渲染；导入 VMD 摄像机或添加摄像机后按场景相机渲染",
                )
            if applied_fast:
                self.report(
                    {"INFO"},
                    "已套用速度优先预设（关光追/半分辨率阴影/16 采样/关运动模糊），渲染完成已恢复",
                )
        finally:
            for key, value in saved_eevee.items():
                try:
                    setattr(scene.eevee, key, value)
                except Exception:
                    pass
            scene.render.use_motion_blur = old_motion_blur
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
