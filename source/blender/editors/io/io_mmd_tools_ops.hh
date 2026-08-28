/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup editor/io
 */

#pragma once

struct wmOperatorType;
struct ARegionType;

namespace blender {

void MMD_TOOLS_OT_import_pmx(wmOperatorType *ot);
void MMD_TOOLS_OT_export_pmx(wmOperatorType *ot);
void MMD_TOOLS_OT_import_vmd(wmOperatorType *ot);
void MMD_TOOLS_OT_export_vmd(wmOperatorType *ot);
void MMD_TOOLS_OT_attach_meshes(wmOperatorType *ot);
void MMD_TOOLS_OT_convert_to_mmd_model(wmOperatorType *ot);
void MMD_TOOLS_OT_edge_preview_setup(wmOperatorType *ot);
void MMD_TOOLS_OT_convert_materials(wmOperatorType *ot);

/** Register all mmd_tools-namespace operators (called from ED_operatortypes_io). */
void mmd_tools_ops_register_operators();

/** Register the "MMD Tools" panel in the 3D View sidebar (N-panel). */
void ED_mmd_tools_panel_register(ARegionType *art);

}  // namespace blender
