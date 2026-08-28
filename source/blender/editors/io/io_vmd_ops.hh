/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup editor/io
 */

#pragma once

struct wmOperatorType;

namespace blender {

struct Main;
struct Object;
struct ReportList;

void WM_OT_vmd_import(wmOperatorType *ot);
void WM_OT_vmd_camera_import(wmOperatorType *ot);
void WM_OT_vmd_export(wmOperatorType *ot);
void WM_OT_vmd_camera_export(wmOperatorType *ot);

/** 挂起全部 MMD 近似 IK 约束与原生 CCD（手动 GPU 烘焙完成后调用）。 */
int vmd_suspend_all_ik_after_bake(Main *bmain, Object *ob);

/** 导入后处理：与原生 VMD 导入一致地挂起 axis/append 近似约束、激活 Rigify
 * 播放模式,供 mmd_tools 命名空间导入代理复用,避免"腿找原点/姿态破坏"
 * (iTaSC 或原生 CCD 双重求解让 IK 覆盖 FK)。返回挂起的约束数。 */
int vmd_tools_prepare_imported_action(Main *bmain, Object *ob, ReportList *reports);

namespace ed::io {
void vmd_file_handler_add();
}

}  // namespace blender
