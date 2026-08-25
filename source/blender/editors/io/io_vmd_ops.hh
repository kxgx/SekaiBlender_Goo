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

void WM_OT_vmd_import(wmOperatorType *ot);
void WM_OT_vmd_camera_import(wmOperatorType *ot);
void WM_OT_vmd_export(wmOperatorType *ot);
void WM_OT_vmd_camera_export(wmOperatorType *ot);

/** 挂起全部 MMD 近似 IK 约束与原生 CCD（手动 GPU 烘焙完成后调用）。 */
int vmd_suspend_all_ik_after_bake(Main *bmain, Object *ob);

namespace ed::io {
void vmd_file_handler_add();
}

}  // namespace blender
