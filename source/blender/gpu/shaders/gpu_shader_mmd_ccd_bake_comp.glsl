/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

#include "GPU_shader_shared.hh"
#include "infos/gpu_shader_mmd_ccd_infos.hh"

COMPUTE_SHADER_CREATE_INFO(gpu_shader_mmd_ccd_bake)

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

/* -------------------------------------------------------------------- */
/* MMD Y-up row-major 数学（与 mmd_ccd_ik_v8.cc 逐行对应）             */
/* -------------------------------------------------------------------- */

#define kEps 1.0e-10f

/* quat (w, x, y, z) 归一化（与 v8.cc 一致：先取倒数再乘，避免除法舍入差异） */
float4 quat_normalize(float4 q)
{
  float len = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (len < kEps) {
    return float4(1.0, 0.0, 0.0, 0.0);
  }
  float inv = 1.0 / len;
  return q * inv;
}

/* 数学 a * b（Hamilton） */
float4 quat_mul(float4 a, float4 b)
{
  return quat_normalize(float4(a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
                             a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
                             a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
                             a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x));
}

/* 四元数 → row-major 3x3（m[c] = 第 c 行） */
void quat_to_row3(float4 q_in, out float3 m0, out float3 m1, out float3 m2)
{
  float4 q = quat_normalize(q_in);
  float w = q.x, x = q.y, y = q.z, z = q.w;
  /* 先 column-major 再转置 */
  float c00 = 1.0 - 2.0 * (y * y + z * z);
  float c01 = 2.0 * (x * y - z * w);
  float c02 = 2.0 * (x * z + y * w);
  float c10 = 2.0 * (x * y + z * w);
  float c11 = 1.0 - 2.0 * (x * x + z * z);
  float c12 = 2.0 * (y * z - x * w);
  float c20 = 2.0 * (x * z - y * w);
  float c21 = 2.0 * (y * z + x * w);
  float c22 = 1.0 - 2.0 * (x * x + y * y);
  m0 = float3(c00, c10, c20);
  m1 = float3(c01, c11, c21);
  m2 = float3(c02, c12, c22);
}

/* row-major 3x3 → 四元数 */
float4 quat_from_row3(float3 r0, float3 r1, float3 r2)
{
  /* 转置到 column-major */
  float m00 = r0.x, m01 = r1.x, m02 = r2.x;
  float m10 = r0.y, m11 = r1.y, m12 = r2.y;
  float m20 = r0.z, m21 = r1.z, m22 = r2.z;
  float trace = m00 + m11 + m22;
  float4 q;
  if (trace > 0.0) {
    float s = sqrt(trace + 1.0) * 2.0;
    q = float4(0.25 * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s);
  }
  else if (m00 > m11 && m00 > m22) {
    float s = sqrt(max(0.0, 1.0 + m00 - m11 - m22)) * 2.0;
    q = float4((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s);
  }
  else if (m11 > m22) {
    float s = sqrt(max(0.0, 1.0 + m11 - m00 - m22)) * 2.0;
    q = float4((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s);
  }
  else {
    float s = sqrt(max(0.0, 1.0 + m22 - m00 - m11)) * 2.0;
    q = float4((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s);
  }
  return quat_normalize(q);
}

/* 绕单轴旋转的 row-major 3x3 */
void axis_rotation_row(int axis, float angle, out float3 m0, out float3 m1, out float3 m2)
{
  float4 q = float4(1.0, 0.0, 0.0, 0.0);
  float half_angle = angle * 0.5;
  q.x = cos(half_angle);
  q[axis + 1] = sin(half_angle);
  quat_to_row3(q, m0, m1, m2);
}

void mul3_rows(float3 a0, float3 a1, float3 a2, float3 b0, float3 b1, float3 b2,
               out float3 r0, out float3 r1, out float3 r2)
{
  /* result[r][c] = sum(a[r][k] * b[k][c]) */
  r0 = float3(dot(a0, float3(b0.x, b1.x, b2.x)),
            dot(a0, float3(b0.y, b1.y, b2.y)),
            dot(a0, float3(b0.z, b1.z, b2.z)));
  r1 = float3(dot(a1, float3(b0.x, b1.x, b2.x)),
            dot(a1, float3(b0.y, b1.y, b2.y)),
            dot(a1, float3(b0.z, b1.z, b2.z)));
  r2 = float3(dot(a2, float3(b0.x, b1.x, b2.x)),
            dot(a2, float3(b0.y, b1.y, b2.y)),
            dot(a2, float3(b0.z, b1.z, b2.z)));
}

/* -------------------------------------------------------------------- */
/* 帧内骨数据读取 helper（m0 以 4 行 float4 存储于 frame_buf，就地更新）  */
/* -------------------------------------------------------------------- */

void frame_get_m0(int idx, out float3 m0, out float3 m1, out float3 m2, out float3 m3)
{
  MmdBakeFrameBone d = frame_buf[idx];
  m0 = d.m0_row0.xyz;
  m1 = d.m0_row1.xyz;
  m2 = d.m0_row2.xyz;
  /* 平移在 row3（与 C++ 上传的 row-major 布局一致） */
  m3 = d.m0_row3.xyz;
}

void frame_set_m0(int idx, float3 m0, float3 m1, float3 m2, float3 m3)
{
  MmdBakeFrameBone d = frame_buf[idx];
  d.m0_row0 = float4(m0, 0.0);
  d.m0_row1 = float4(m1, 0.0);
  d.m0_row2 = float4(m2, 0.0);
  d.m0_row3 = float4(m3, 1.0);
  frame_buf[idx] = d;
}

/* row-vector 点变换：result[c] = sum(point[k] * M[k][c]) + M[3][c]
 * （与 mmd_ccd_ik_v8.cc 一致：m0/m1/m2 为矩阵行，平移在 m3） */
float3 transform_point(float3 point, float3 m0, float3 m1, float3 m2, float3 m3)
{
  return float3(dot(point, float3(m0.x, m1.x, m2.x)) + m3.x,
                dot(point, float3(m0.y, m1.y, m2.y)) + m3.y,
                dot(point, float3(m0.z, m1.z, m2.z)) + m3.z);
}

/* 列向量左乘 row-major M：result[r] = sum(M[r][k] * v[k]) */
float3 mul_mat_vec(float3 m0, float3 m1, float3 m2, float3 v)
{
  return float3(dot(m0, v), dot(m1, v), dot(m2, v));
}

/* pivot_matrix (T(pos)*R*T(-pos) 的 row-major 形式) */
void pivot_matrix(float3 position, float3 r0, float3 r1, float3 r2,
                  out float3 m0, out float3 m1, out float3 m2, out float3 m3)
{
  m0 = r0;
  m1 = r1;
  m2 = r2;
  /* trans[c] = position[c] - sum(position[k] * rotation[k][c])（点乘列） */
  m3 = float3(position.x - dot(position, float3(r0.x, r1.x, r2.x)),
            position.y - dot(position, float3(r0.y, r1.y, r2.y)),
            position.z - dot(position, float3(r0.z, r1.z, r2.z)));
}

/* 四元数 → pivot 矩阵 + 与父矩阵层级组合：m0[bone] = m1[bone] * m0[parent] */
void compose_pivot_mul_parent(int idx,
                              int parent,
                              float3 base_pos,
                              float4 q,
                              out float3 m0, out float3 m1, out float3 m2, out float3 m3)
{
  float3 r0, r1, r2;
  quat_to_row3(q, r0, r1, r2);
  float3 p0, p1, p2, p3;
  pivot_matrix(base_pos, r0, r1, r2, p0, p1, p2, p3);
  if (parent >= 0) {
    float3 pm0, pm1, pm2, pm3;
    frame_get_m0(parent, pm0, pm1, pm2, pm3);
    /* mul4: result[r][c] = sum(a[r][k] * b[k][c])
     * 平移行 result[3][c] = dot(a3, b_col_c) + b[3][c]（a[3][3]=1）。 */
    mul3_rows(p0, p1, p2, pm0, pm1, pm2, m0, m1, m2);
    m3 = float3(dot(p3, float3(pm0.x, pm1.x, pm2.x)) + pm3.x,
                dot(p3, float3(pm0.y, pm1.y, pm2.y)) + pm3.y,
                dot(p3, float3(pm0.z, pm1.z, pm2.z)) + pm3.z);
  }
  else {
    m0 = p0;
    m1 = p1;
    m2 = p2;
    m3 = p3;
  }
}

/* -------------------------------------------------------------------- */
/* Euler limit 分支（与 v8.cc 逐行对应）                                */
/* -------------------------------------------------------------------- */

int branch_for_limits(float3 limit_min, float3 limit_max)
{
  if (limit_min.x <= -M_PI * 0.5 || limit_max.x >= M_PI * 0.5) {
    if (limit_min.y <= -M_PI * 0.5 || limit_max.y >= M_PI * 0.5) {
      return 0; /* 'A' */
    }
    return 1; /* 'B' */
  }
  return 2; /* 'C' */
}

void decompose_branch_euler(int branch, float3 r0, float3 r1, float3 r2, out float3 angles)
{
  if (branch == 0) {
    float z = asin(clamp(-r1.x, -1.0, 1.0));
    float cz = cos(z);
    if (abs(cz) < 1.0e-8) {
      angles = float3(0.0, atan(-r0.z, r2.z), z);
    }
    else {
      angles = float3(atan(r1.z, r1.y), atan(r2.x, r0.x), z);
    }
  }
  else if (branch == 1) {
    float y = asin(clamp(-r0.z, -1.0, 1.0));
    float cy = cos(y);
    if (abs(cy) < 1.0e-8) {
      angles = float3(atan(-r2.y, r1.y), y, 0.0);
    }
    else {
      angles = float3(atan(r1.z, r2.z), y, atan(r0.y, r0.x));
    }
  }
  else {
    float x = asin(clamp(-r2.y, -1.0, 1.0));
    float cx = cos(x);
    if (abs(cx) < 1.0e-8) {
      angles = float3(x, 0.0, atan(-r1.x, r0.x));
    }
    else {
      angles = float3(x, atan(r2.x, r2.z), atan(r0.y, r1.y));
    }
  }
}

void compose_branch_euler(int branch, float3 angles, out float3 m0, out float3 m1, out float3 m2)
{
  float3 rx0, rx1, rx2, ry0, ry1, ry2, rz0, rz1, rz2;
  axis_rotation_row(0, angles.x, rx0, rx1, rx2);
  axis_rotation_row(1, angles.y, ry0, ry1, ry2);
  axis_rotation_row(2, angles.z, rz0, rz1, rz2);
  if (branch == 0) {
    /* Ry * Rz * Rx */
    float3 t0, t1, t2;
    mul3_rows(ry0, ry1, ry2, rz0, rz1, rz2, t0, t1, t2);
    mul3_rows(t0, t1, t2, rx0, rx1, rx2, m0, m1, m2);
  }
  else if (branch == 1) {
    /* Rx * Ry * Rz */
    float3 t0, t1, t2;
    mul3_rows(rx0, rx1, rx2, ry0, ry1, ry2, t0, t1, t2);
    mul3_rows(t0, t1, t2, rz0, rz1, rz2, m0, m1, m2);
  }
  else {
    /* Rz * Rx * Ry */
    float3 t0, t1, t2;
    mul3_rows(rz0, rz1, rz2, rx0, rx1, rx2, t0, t1, t2);
    mul3_rows(t0, t1, t2, ry0, ry1, ry2, m0, m1, m2);
  }
}

float4 apply_mmd_link_limit(float3 limit_min,
                          float3 limit_max,
                          float4 q,
                          int iteration_index,
                          int loop_count)
{
  int branch = branch_for_limits(limit_min, limit_max);
  float3 r0, r1, r2;
  quat_to_row3(q, r0, r1, r2);
  float3 angles;
  decompose_branch_euler(branch, r0, r1, r2, angles);

  bool hard_clamp = (loop_count / 2) <= iteration_index;
  for (int axis = 0; axis < 3; axis++) {
    float value = angles[axis];
    float low = limit_min[axis];
    float high = limit_max[axis];
    if (value < low) {
      float reflected = low * 2.0 - value;
      if (hard_clamp || reflected < low || reflected > high) {
        value = low;
      }
      else {
        value = reflected;
      }
    }
    else if (value > high) {
      float reflected = high * 2.0 - value;
      if (hard_clamp || reflected < low || reflected > high) {
        value = high;
      }
      else {
        value = reflected;
      }
    }
    angles[axis] = value;
  }
  float3 c0, c1, c2;
  compose_branch_euler(branch, angles, c0, c1, c2);
  return quat_from_row3(c0, c1, c2);
}

/* 单链求解（与 solve_chain_v8 逐行对应）                              */
/* -------------------------------------------------------------------- */

void solve_chain_v8(MmdBakeChainConst chain, int frame_base, int out_frame_base)
{
  int target_idx = chain.target_bone_index;
  int effector_idx = chain.effector_bone_index;
  int iterations = chain.iterations;
  int half_iter = iterations / 2;
  float ik_angle = chain.runtime_angle;

  /* target 世界位置：m0[target] 用 initial_m0（不传播） */
  float3 tm0, tm1, tm2, tm3;
  frame_get_m0(frame_base + target_idx, tm0, tm1, tm2, tm3);
  float3 target = transform_point(bone_const_buf[target_idx].base_pos_mmd, tm0, tm1, tm2, tm3);

  for (int iteration_index = 0; iteration_index < iterations; iteration_index++) {
    for (int runtime_order = 0; runtime_order < chain.link_count; runtime_order++) {
      MmdBakeLinkConst link = link_const_buf[chain.link_offset + runtime_order];
      int link_idx = link.bone_index;

      float3 jm0, jm1, jm2, jm3;
      frame_get_m0(frame_base + link_idx, jm0, jm1, jm2, jm3);
      float3 joint = transform_point(bone_const_buf[link_idx].base_pos_mmd, jm0, jm1, jm2, jm3);

      float3 em0, em1, em2, em3;
      frame_get_m0(frame_base + effector_idx, em0, em1, em2, em3);
      float3 effector = transform_point(
          bone_const_buf[effector_idx].base_pos_mmd, em0, em1, em2, em3);

      float3 to_effector_raw = effector - joint;
      float to_eff_len = length(to_effector_raw);
      if (to_eff_len < kEps) {
        continue;
      }
      float3 to_effector = to_effector_raw / to_eff_len;

      float3 to_target_raw = target - joint;
      float to_tgt_len = length(to_target_raw);
      if (to_tgt_len < kEps) {
        continue;
      }
      float3 to_target = to_target_raw / to_tgt_len;

      /* 方向收敛检查 */
      float3 diff = to_target - to_effector;
      if (dot(diff, diff) < 1.0e-7) {
        continue;
      }

      float3 axis_world = cross(to_effector, to_target);

      int parent_idx = int(bone_const_buf[link_idx].parent_index);
      float3 pm0, pm1, pm2, pm3;
      frame_get_m0(frame_base + parent_idx, pm0, pm1, pm2, pm3);
      float3 cl = mul_mat_vec(pm0, pm1, pm2, axis_world);

      float3 axis_local;
      if (link.has_limit != 0 && iteration_index < half_iter) {
        float3 lmin = link.limit_min_mmd.xyz;
        float3 lmax = link.limit_max_mmd.xyz;
        if (lmin.y == 0.0 && lmax.y == 0.0 && lmin.z == 0.0 && lmax.z == 0.0) {
          axis_local = float3((cl.x >= 0.0) ? 1.0 : -1.0, 0.0, 0.0);
        }
        else if (lmin.x == 0.0 && lmax.x == 0.0 && lmin.z == 0.0 && lmax.z == 0.0) {
          axis_local = float3(0.0, (cl.y >= 0.0) ? 1.0 : -1.0, 0.0);
        }
        else {
          axis_local = float3(0.0, 0.0, (cl.z >= 0.0) ? 1.0 : -1.0);
        }
      }
      else {
        float cl_len = length(cl);
        if (cl_len < kEps) {
          continue;
        }
        axis_local = cl / cl_len;
      }

      float cosine = clamp(dot(to_effector, to_target), -1.0, 1.0);
      float half_angle = 0.5 * acos(cosine);

      float cap = float(runtime_order + 1) * ik_angle * 2.0;
      half_angle = clamp(half_angle, -cap, cap);

      float4 delta = float4(cos(half_angle),
                        axis_local.x * sin(half_angle),
                        axis_local.y * sin(half_angle),
                        axis_local.z * sin(half_angle));

      /* D3DX 反序：q_cur = delta * q_cur（紧凑输出槽位） */
      int out_idx = link.out_index;
      float4 q_cur = frame_out_buf[out_frame_base + out_idx].q_current_mmd;
      q_cur = quat_mul(delta, q_cur);

      /* 首轮：q_cur = q_cur * q_base */
      if (iteration_index == 0) {
        q_cur = quat_mul(q_cur, frame_buf[frame_base + link_idx].q_base_mmd);
      }

      if (link.has_limit != 0) {
        q_cur = apply_mmd_link_limit(
            link.limit_min_mmd.xyz, link.limit_max_mmd.xyz, q_cur, iteration_index, iterations);
      }

      frame_out_buf[out_frame_base + out_idx].q_current_mmd = q_cur;

      /* 刷新 links[0..current] 反向 + effector 折叠到 links[0] */
      for (int refresh_index = runtime_order; refresh_index >= 0; refresh_index--) {
        MmdBakeLinkConst rlink = link_const_buf[chain.link_offset + refresh_index];
        int bone_idx = rlink.bone_index;
        int parent = int(bone_const_buf[bone_idx].parent_index);
        float4 rq = frame_out_buf[out_frame_base + rlink.out_index].q_current_mmd;
        float3 rr0, rr1, rr2;
        quat_to_row3(rq, rr0, rr1, rr2);
        float3 m0, m1, m2, m3;
        compose_pivot_mul_parent(frame_base + bone_idx,
                                 (parent >= 0) ? (frame_base + parent) : -1,
                                 bone_const_buf[bone_idx].base_pos_mmd,
                                 rq,
                                 m0, m1, m2, m3);
        frame_set_m0(frame_base + bone_idx, m0, m1, m2, m3);
      }
      /* effector 折叠到 links[0] */
      {
        int first_link = link_const_buf[chain.link_offset].bone_index;
        float3 m0, m1, m2, m3;
        frame_get_m0(frame_base + first_link, m0, m1, m2, m3);
        frame_set_m0(frame_base + effector_idx, m0, m1, m2, m3);
      }
    }
  }
}

/* -------------------------------------------------------------------- */
/* 主入口：每个 invocation 求解一帧                                     */
/* -------------------------------------------------------------------- */

void main()
{
  uint frame = gl_GlobalInvocationID.x;
  if (int(frame) >= frame_count) {
    return;
  }
  int frame_base = int(frame) * bone_count;

  /* 1. q_current 初始 identity（仅链骨有紧凑输出槽位） */
  int out_frame_base = int(frame) * link_count;
  for (int i = 0; i < bone_count; i++) {
    int out_idx = bone_const_buf[i].out_index;
    if (out_idx >= 0) {
      frame_out_buf[out_frame_base + out_idx].q_current_mmd = float4(1.0, 0.0, 0.0, 0.0);
    }
  }

  /* 2. 骨架 m0 初始化（与 v8.cc 662-693 行对应） */
  for (int i = 0; i < bone_count; i++) {
    MmdBakeBoneConst bc = bone_const_buf[i];
    MmdBakeFrameBone fd = frame_buf[frame_base + i];
    int parent = bc.parent_index;
    bool is_target = (bc.flags & 1) != 0;
    bool is_anchor = (bc.flags & 4) != 0;
    if (is_target) {
      /* 保持 direct pose matrix（initial_m0） */
      frame_buf[frame_base + i].m0_row0 = fd.m0_row0;
      frame_buf[frame_base + i].m0_row1 = fd.m0_row1;
      frame_buf[frame_base + i].m0_row2 = fd.m0_row2;
      frame_buf[frame_base + i].m0_row3 = fd.m0_row3;
      continue;
    }
    float3 m0, m1, m2, m3;
    compose_pivot_mul_parent(frame_base + i,
                             (parent >= 0) ? (frame_base + parent) : -1,
                             bc.base_pos_mmd,
                             fd.q_base_mmd,
                             m0, m1, m2, m3);
    if (is_anchor) {
      /* 独立链根父：旋转用 q_base 传播，平移用 direct pose head
       * head_mmd = transform_point(base_pos, initial_m0)（与 v8 逐行对应） */
      float3 head_mmd = transform_point(bc.base_pos_mmd,
                                        fd.m0_row0.xyz,
                                        fd.m0_row1.xyz,
                                        fd.m0_row2.xyz,
                                        fd.m0_row3.xyz);
      float3 base_times_r = float3(dot(bc.base_pos_mmd, float3(m0.x, m1.x, m2.x)),
                               dot(bc.base_pos_mmd, float3(m0.y, m1.y, m2.y)),
                               dot(bc.base_pos_mmd, float3(m0.z, m1.z, m2.z)));
      m3 = head_mmd - base_times_r;
    }
    frame_set_m0(frame_base + i, m0, m1, m2, m3);
  }

  /* 3. 按顺序求解每条链 */
  for (int c = 0; c < chain_count; c++) {
    solve_chain_v8(chain_const_buf[c], frame_base, out_frame_base);
  }
}
