#include "gpu_shader_utildefines_lib.glsl"

/* Goo-engine 移植（Set Depth 节点，5.3 Eevee Next 实现）：
 * 深度写入发生在 PREPASS（surf_depth 无 early_fragment_tests，管线自带 WRITE_DEPTH）；
 * USE_SET_DEPTH define 对材质所有管线（含 prepass）生效。View Depth > 0 时写自定义
 * 深度（reverse-Z：近=1 远=0）；否则纯直通。 */
[[node]]
void node_set_depth(in Closure _in, in float z_in, out Closure _out)
{
  _out = _in;
#ifdef USE_SET_DEPTH
  if (z_in > 0.0) {
    [[resource_table]] const draw::View &views = resource_table_get(draw::View);
    const ViewMatrices view = views.get(0);
    /* z_in 为期望视距（正）；视空间中相机前方是 -Z。 */
    float3 vP = view.point_world_to_view(g_data.P);
    vP.z = -z_in;
    float ndc_z = view.point_view_to_ndc(vP).z;
    gl_FragDepth = (1.0 - ndc_z) * 0.5;
  }
#endif
}
