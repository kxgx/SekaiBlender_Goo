/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup mmd_ccd_ik
 *
 * MMD CCD 批量烘焙的 CUDA 后端（R9-CUDA）。与
 * gpu_shader_mmd_ccd_bake_comp.glsl 逐行对应：每线程求解一帧，帧内链按
 * CPU v8 相同顺序求解。数据布局与 mmd_bake_gpu_types.hh 逐字节一致。
 */

#include "mmd_bake_gpu_types.hh"

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace blender::mmd::bake {

namespace {

constexpr float kEps = 1.0e-10f;

/* 与 GLSL float3/float4 对应的轻量向量（避免依赖 CUDA 向量库语义）。 */
struct F3 {
  float x, y, z;
};
struct F4 {
  float x, y, z, w;
};

__device__ __forceinline__ F3 make_f3(float x, float y, float z)
{
  F3 v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
}

__device__ __forceinline__ F4 make_f4(float x, float y, float z, float w)
{
  F4 v;
  v.x = x;
  v.y = y;
  v.z = z;
  v.w = w;
  return v;
}

__device__ __forceinline__ F3 f3_add(F3 a, F3 b)
{
  return make_f3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ F3 f3_sub(F3 a, F3 b)
{
  return make_f3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ F3 f3_scale(F3 a, float s)
{
  return make_f3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ F4 f4_scale(F4 a, float s)
{
  return make_f4(a.x * s, a.y * s, a.z * s, a.w * s);
}
__device__ __forceinline__ float dot3(F3 a, F3 b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ F3 cross3(F3 a, F3 b)
{
  return make_f3(a.y * b.z - a.z * b.y,
                 a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x);
}
__device__ __forceinline__ float len3(F3 v)
{
  return sqrtf(dot3(v, v));
}

/* quat (w, x, y, z) 归一化（与 v8.cc 一致：先取倒数再乘） */
__device__ F4 quat_normalize(F4 q)
{
  float len = sqrtf(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (len < kEps) {
    return make_f4(1.0f, 0.0f, 0.0f, 0.0f);
  }
  float inv = 1.0f / len;
  return f4_scale(q, inv);
}

/* 数学 a * b（Hamilton） */
__device__ F4 quat_mul(F4 a, F4 b)
{
  return quat_normalize(make_f4(a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
                                a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
                                a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
                                a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x));
}

/* 四元数 → row-major 3x3（m[c] = 第 c 行） */
__device__ void quat_to_row3(F4 q_in, F3 &m0, F3 &m1, F3 &m2)
{
  F4 q = quat_normalize(q_in);
  float w = q.x, x = q.y, y = q.z, z = q.w;
  float c00 = 1.0f - 2.0f * (y * y + z * z);
  float c01 = 2.0f * (x * y - z * w);
  float c02 = 2.0f * (x * z + y * w);
  float c10 = 2.0f * (x * y + z * w);
  float c11 = 1.0f - 2.0f * (x * x + z * z);
  float c12 = 2.0f * (y * z - x * w);
  float c20 = 2.0f * (x * z - y * w);
  float c21 = 2.0f * (y * z + x * w);
  float c22 = 1.0f - 2.0f * (x * x + y * y);
  m0 = make_f3(c00, c10, c20);
  m1 = make_f3(c01, c11, c21);
  m2 = make_f3(c02, c12, c22);
}

/* row-major 3x3 → 四元数 */
__device__ F4 quat_from_row3(F3 r0, F3 r1, F3 r2)
{
  float m00 = r0.x, m01 = r1.x, m02 = r2.x;
  float m10 = r0.y, m11 = r1.y, m12 = r2.y;
  float m20 = r0.z, m21 = r1.z, m22 = r2.z;
  float trace = m00 + m11 + m22;
  F4 q;
  if (trace > 0.0f) {
    float s = sqrtf(trace + 1.0f) * 2.0f;
    q = make_f4(0.25f * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s);
  }
  else if (m00 > m11 && m00 > m22) {
    float s = sqrtf(fmaxf(0.0f, 1.0f + m00 - m11 - m22)) * 2.0f;
    q = make_f4((m21 - m12) / s, 0.25f * s, (m01 + m10) / s, (m02 + m20) / s);
  }
  else if (m11 > m22) {
    float s = sqrtf(fmaxf(0.0f, 1.0f + m11 - m00 - m22)) * 2.0f;
    q = make_f4((m02 - m20) / s, (m01 + m10) / s, 0.25f * s, (m12 + m21) / s);
  }
  else {
    float s = sqrtf(fmaxf(0.0f, 1.0f + m22 - m00 - m11)) * 2.0f;
    q = make_f4((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25f * s);
  }
  return quat_normalize(q);
}

/* 绕单轴旋转的 row-major 3x3 */
__device__ void axis_rotation_row(int axis, float angle, F3 &m0, F3 &m1, F3 &m2)
{
  F4 q = make_f4(1.0f, 0.0f, 0.0f, 0.0f);
  float half_angle = angle * 0.5f;
  q.x = cosf(half_angle);
  if (axis == 0) {
    q.y = sinf(half_angle);
  }
  else if (axis == 1) {
    q.z = sinf(half_angle);
  }
  else {
    q.w = sinf(half_angle);
  }
  quat_to_row3(q, m0, m1, m2);
}

__device__ void mul3_rows(F3 a0, F3 a1, F3 a2, F3 b0, F3 b1, F3 b2, F3 &r0, F3 &r1, F3 &r2)
{
  /* result[r][c] = sum(a[r][k] * b[k][c]) */
  r0 = make_f3(dot3(a0, make_f3(b0.x, b1.x, b2.x)),
               dot3(a0, make_f3(b0.y, b1.y, b2.y)),
               dot3(a0, make_f3(b0.z, b1.z, b2.z)));
  r1 = make_f3(dot3(a1, make_f3(b0.x, b1.x, b2.x)),
               dot3(a1, make_f3(b0.y, b1.y, b2.y)),
               dot3(a1, make_f3(b0.z, b1.z, b2.z)));
  r2 = make_f3(dot3(a2, make_f3(b0.x, b1.x, b2.x)),
               dot3(a2, make_f3(b0.y, b1.y, b2.y)),
               dot3(a2, make_f3(b0.z, b1.z, b2.z)));
}

/* 帧内骨数据读取 helper（m0 就地存储于 frame_buf，平移在 row3） */
__device__ void frame_get_m0(const FrameBone *frame_buf,
                             int idx,
                             F3 &m0,
                             F3 &m1,
                             F3 &m2,
                             F3 &m3)
{
  const FrameBone &d = frame_buf[idx];
  m0 = make_f3(d.m0_row0[0], d.m0_row0[1], d.m0_row0[2]);
  m1 = make_f3(d.m0_row1[0], d.m0_row1[1], d.m0_row1[2]);
  m2 = make_f3(d.m0_row2[0], d.m0_row2[1], d.m0_row2[2]);
  m3 = make_f3(d.m0_row3[0], d.m0_row3[1], d.m0_row3[2]);
}

__device__ void frame_set_m0(FrameBone *frame_buf,
                             int idx,
                             F3 m0,
                             F3 m1,
                             F3 m2,
                             F3 m3)
{
  FrameBone &d = frame_buf[idx];
  d.m0_row0[0] = m0.x;
  d.m0_row0[1] = m0.y;
  d.m0_row0[2] = m0.z;
  d.m0_row0[3] = 0.0f;
  d.m0_row1[0] = m1.x;
  d.m0_row1[1] = m1.y;
  d.m0_row1[2] = m1.z;
  d.m0_row1[3] = 0.0f;
  d.m0_row2[0] = m2.x;
  d.m0_row2[1] = m2.y;
  d.m0_row2[2] = m2.z;
  d.m0_row2[3] = 0.0f;
  d.m0_row3[0] = m3.x;
  d.m0_row3[1] = m3.y;
  d.m0_row3[2] = m3.z;
  d.m0_row3[3] = 1.0f;
}

/* row-vector 点变换：result[c] = sum(point[k] * M[k][c]) + M[3][c] */
__device__ F3 transform_point(F3 point, F3 m0, F3 m1, F3 m2, F3 m3)
{
  return make_f3(dot3(point, make_f3(m0.x, m1.x, m2.x)) + m3.x,
                 dot3(point, make_f3(m0.y, m1.y, m2.y)) + m3.y,
                 dot3(point, make_f3(m0.z, m1.z, m2.z)) + m3.z);
}

/* 列向量左乘 row-major M：result[r] = sum(M[r][k] * v[k]) */
__device__ F3 mul_mat_vec(F3 m0, F3 m1, F3 m2, F3 v)
{
  return make_f3(dot3(m0, v), dot3(m1, v), dot3(m2, v));
}

/* pivot_matrix (T(pos)*R*T(-pos) 的 row-major 形式) */
__device__ void pivot_matrix(F3 position, F3 r0, F3 r1, F3 r2, F3 &m0, F3 &m1, F3 &m2, F3 &m3)
{
  m0 = r0;
  m1 = r1;
  m2 = r2;
  /* trans[c] = position[c] - sum(position[k] * rotation[k][c])（点乘列） */
  m3 = make_f3(position.x - dot3(position, make_f3(r0.x, r1.x, r2.x)),
               position.y - dot3(position, make_f3(r0.y, r1.y, r2.y)),
               position.z - dot3(position, make_f3(r0.z, r1.z, r2.z)));
}

/* 四元数 → pivot 矩阵 + 与父矩阵层级组合：m0[bone] = m1[bone] * m0[parent] */
__device__ void compose_pivot_mul_parent(FrameBone *frame_buf,
                                         int parent,
                                         F3 base_pos,
                                         F4 q,
                                         F3 &m0,
                                         F3 &m1,
                                         F3 &m2,
                                         F3 &m3)
{
  F3 r0, r1, r2;
  quat_to_row3(q, r0, r1, r2);
  F3 p0, p1, p2, p3;
  pivot_matrix(base_pos, r0, r1, r2, p0, p1, p2, p3);
  if (parent >= 0) {
    F3 pm0, pm1, pm2, pm3;
    frame_get_m0(frame_buf, parent, pm0, pm1, pm2, pm3);
    /* mul4: result[r][c] = sum(a[r][k] * b[k][c])
     * 平移行 result[3][c] = dot(a3, b_col_c) + b[3][c]（a[3][3]=1）。 */
    mul3_rows(p0, p1, p2, pm0, pm1, pm2, m0, m1, m2);
    m3 = make_f3(dot3(p3, make_f3(pm0.x, pm1.x, pm2.x)) + pm3.x,
                 dot3(p3, make_f3(pm0.y, pm1.y, pm2.y)) + pm3.y,
                 dot3(p3, make_f3(pm0.z, pm1.z, pm2.z)) + pm3.z);
  }
  else {
    m0 = p0;
    m1 = p1;
    m2 = p2;
    m3 = p3;
  }
}

/* Euler limit 分支（与 v8.cc 逐行对应） */
__device__ int branch_for_limits(F3 limit_min, F3 limit_max)
{
  if (limit_min.x <= -M_PI * 0.5f || limit_max.x >= M_PI * 0.5f) {
    if (limit_min.y <= -M_PI * 0.5f || limit_max.y >= M_PI * 0.5f) {
      return 0; /* 'A' */
    }
    return 1; /* 'B' */
  }
  return 2; /* 'C' */
}

__device__ void decompose_branch_euler(int branch, F3 r0, F3 r1, F3 r2, F3 &angles)
{
  if (branch == 0) {
    float z = asinf(fminf(1.0f, fmaxf(-1.0f, -r1.x)));
    float cz = cosf(z);
    if (fabsf(cz) < 1.0e-8f) {
      angles = make_f3(0.0f, atan2f(-r0.z, r2.z), z);
    }
    else {
      angles = make_f3(atan2f(r1.z, r1.y), atan2f(r2.x, r0.x), z);
    }
  }
  else if (branch == 1) {
    float y = asinf(fminf(1.0f, fmaxf(-1.0f, -r0.z)));
    float cy = cosf(y);
    if (fabsf(cy) < 1.0e-8f) {
      angles = make_f3(atan2f(-r2.y, r1.y), y, 0.0f);
    }
    else {
      angles = make_f3(atan2f(r1.z, r2.z), y, atan2f(r0.y, r0.x));
    }
  }
  else {
    float x = asinf(fminf(1.0f, fmaxf(-1.0f, -r2.y)));
    float cx = cosf(x);
    if (fabsf(cx) < 1.0e-8f) {
      angles = make_f3(x, 0.0f, atan2f(-r1.x, r0.x));
    }
    else {
      angles = make_f3(x, atan2f(r2.x, r2.z), atan2f(r0.y, r1.y));
    }
  }
}

__device__ void compose_branch_euler(int branch, F3 angles, F3 &m0, F3 &m1, F3 &m2)
{
  F3 rx0, rx1, rx2, ry0, ry1, ry2, rz0, rz1, rz2;
  axis_rotation_row(0, angles.x, rx0, rx1, rx2);
  axis_rotation_row(1, angles.y, ry0, ry1, ry2);
  axis_rotation_row(2, angles.z, rz0, rz1, rz2);
  if (branch == 0) {
    /* Ry * Rz * Rx */
    F3 t0, t1, t2;
    mul3_rows(ry0, ry1, ry2, rz0, rz1, rz2, t0, t1, t2);
    mul3_rows(t0, t1, t2, rx0, rx1, rx2, m0, m1, m2);
  }
  else if (branch == 1) {
    /* Rx * Ry * Rz */
    F3 t0, t1, t2;
    mul3_rows(rx0, rx1, rx2, ry0, ry1, ry2, t0, t1, t2);
    mul3_rows(t0, t1, t2, rz0, rz1, rz2, m0, m1, m2);
  }
  else {
    /* Rz * Rx * Ry */
    F3 t0, t1, t2;
    mul3_rows(rz0, rz1, rz2, rx0, rx1, rx2, t0, t1, t2);
    mul3_rows(t0, t1, t2, ry0, ry1, ry2, m0, m1, m2);
  }
}

__device__ F4 apply_mmd_link_limit(
    F3 limit_min, F3 limit_max, F4 q, int iteration_index, int loop_count)
{
  int branch = branch_for_limits(limit_min, limit_max);
  F3 r0, r1, r2;
  quat_to_row3(q, r0, r1, r2);
  F3 angles;
  decompose_branch_euler(branch, r0, r1, r2, angles);

  bool hard_clamp = (loop_count / 2) <= iteration_index;
  float a[3] = {angles.x, angles.y, angles.z};
  float lo[3] = {limit_min.x, limit_min.y, limit_min.z};
  float hi[3] = {limit_max.x, limit_max.y, limit_max.z};
  for (int axis = 0; axis < 3; axis++) {
    float value = a[axis];
    float low = lo[axis];
    float high = hi[axis];
    if (value < low) {
      float reflected = low * 2.0f - value;
      if (hard_clamp || reflected < low || reflected > high) {
        value = low;
      }
      else {
        value = reflected;
      }
    }
    else if (value > high) {
      float reflected = high * 2.0f - value;
      if (hard_clamp || reflected < low || reflected > high) {
        value = high;
      }
      else {
        value = reflected;
      }
    }
    a[axis] = value;
  }
  F3 c0, c1, c2;
  compose_branch_euler(branch, make_f3(a[0], a[1], a[2]), c0, c1, c2);
  return quat_from_row3(c0, c1, c2);
}

/* 单链求解（与 solve_chain_v8 逐行对应） */
__device__ void solve_chain_v8(const BoneConst *bone_const_buf,
                               const LinkConst *link_const_buf,
                               FrameBone *frame_buf,
                               FrameOut *frame_out_buf,
                               ChainConst chain,
                               int frame_base,
                               int out_frame_base)
{
  int target_idx = chain.target;
  int effector_idx = chain.effector;
  int iterations = chain.iterations;
  int half_iter = iterations / 2;
  float ik_angle = chain.runtime_angle;

  /* target 世界位置：m0[target] 用 initial_m0（不传播） */
  F3 tm0, tm1, tm2, tm3;
  frame_get_m0(frame_buf, frame_base + target_idx, tm0, tm1, tm2, tm3);
  F3 target = transform_point(
      make_f3(bone_const_buf[target_idx].base_pos[0],
              bone_const_buf[target_idx].base_pos[1],
              bone_const_buf[target_idx].base_pos[2]),
      tm0, tm1, tm2, tm3);

  for (int iteration_index = 0; iteration_index < iterations; iteration_index++) {
    for (int runtime_order = 0; runtime_order < chain.link_count; runtime_order++) {
      LinkConst link = link_const_buf[chain.link_offset + runtime_order];
      int link_idx = link.bone;

      F3 jm0, jm1, jm2, jm3;
      frame_get_m0(frame_buf, frame_base + link_idx, jm0, jm1, jm2, jm3);
      F3 joint = transform_point(make_f3(bone_const_buf[link_idx].base_pos[0],
                                         bone_const_buf[link_idx].base_pos[1],
                                         bone_const_buf[link_idx].base_pos[2]),
                                 jm0, jm1, jm2, jm3);

      F3 em0, em1, em2, em3;
      frame_get_m0(frame_buf, frame_base + effector_idx, em0, em1, em2, em3);
      F3 effector = transform_point(make_f3(bone_const_buf[effector_idx].base_pos[0],
                                            bone_const_buf[effector_idx].base_pos[1],
                                            bone_const_buf[effector_idx].base_pos[2]),
                                    em0, em1, em2, em3);

      F3 to_effector_raw = f3_sub(effector, joint);
      float to_eff_len = len3(to_effector_raw);
      if (to_eff_len < kEps) {
        continue;
      }
      F3 to_effector = f3_scale(to_effector_raw, 1.0f / to_eff_len);

      F3 to_target_raw = f3_sub(target, joint);
      float to_tgt_len = len3(to_target_raw);
      if (to_tgt_len < kEps) {
        continue;
      }
      F3 to_target = f3_scale(to_target_raw, 1.0f / to_tgt_len);

      /* 方向收敛检查 */
      F3 diff = f3_sub(to_target, to_effector);
      if (dot3(diff, diff) < 1.0e-7f) {
        continue;
      }

      F3 axis_world = cross3(to_effector, to_target);

      int parent_idx = bone_const_buf[link_idx].parent;
      F3 pm0, pm1, pm2, pm3;
      frame_get_m0(frame_buf, frame_base + parent_idx, pm0, pm1, pm2, pm3);
      F3 cl = mul_mat_vec(pm0, pm1, pm2, axis_world);

      F3 axis_local;
      if (link.has_limit != 0 && iteration_index < half_iter) {
        float lminx = link.limit_min[0], lminy = link.limit_min[1], lminz = link.limit_min[2];
        float lmaxx = link.limit_max[0], lmaxy = link.limit_max[1], lmaxz = link.limit_max[2];
        if (lminy == 0.0f && lmaxy == 0.0f && lminz == 0.0f && lmaxz == 0.0f) {
          axis_local = make_f3((cl.x >= 0.0f) ? 1.0f : -1.0f, 0.0f, 0.0f);
        }
        else if (lminx == 0.0f && lmaxx == 0.0f && lminz == 0.0f && lmaxz == 0.0f) {
          axis_local = make_f3(0.0f, (cl.y >= 0.0f) ? 1.0f : -1.0f, 0.0f);
        }
        else {
          axis_local = make_f3(0.0f, 0.0f, (cl.z >= 0.0f) ? 1.0f : -1.0f);
        }
      }
      else {
        float cl_len = len3(cl);
        if (cl_len < kEps) {
          continue;
        }
        axis_local = f3_scale(cl, 1.0f / cl_len);
      }

      float cosine = fminf(1.0f, fmaxf(-1.0f, dot3(to_effector, to_target)));
      float half_angle = 0.5f * acosf(cosine);

      float cap = float(runtime_order + 1) * ik_angle * 2.0f;
      half_angle = fminf(cap, fmaxf(-cap, half_angle));

      F4 delta = make_f4(cosf(half_angle),
                         axis_local.x * sinf(half_angle),
                         axis_local.y * sinf(half_angle),
                         axis_local.z * sinf(half_angle));

      /* D3DX 反序：q_cur = delta * q_cur（紧凑输出槽位） */
      int out_idx = link.out_index;
      F4 q_cur = make_f4(frame_out_buf[out_frame_base + out_idx].q_current[0],
                         frame_out_buf[out_frame_base + out_idx].q_current[1],
                         frame_out_buf[out_frame_base + out_idx].q_current[2],
                         frame_out_buf[out_frame_base + out_idx].q_current[3]);
      q_cur = quat_mul(delta, q_cur);

      /* 首轮：q_cur = q_cur * q_base */
      if (iteration_index == 0) {
        const FrameBone &fb = frame_buf[frame_base + link_idx];
        q_cur = quat_mul(
            q_cur, make_f4(fb.q_base[0], fb.q_base[1], fb.q_base[2], fb.q_base[3]));
      }

      if (link.has_limit != 0) {
        q_cur = apply_mmd_link_limit(make_f3(link.limit_min[0], link.limit_min[1], link.limit_min[2]),
                                     make_f3(link.limit_max[0], link.limit_max[1], link.limit_max[2]),
                                     q_cur,
                                     iteration_index,
                                     iterations);
      }

      frame_out_buf[out_frame_base + out_idx].q_current[0] = q_cur.x;
      frame_out_buf[out_frame_base + out_idx].q_current[1] = q_cur.y;
      frame_out_buf[out_frame_base + out_idx].q_current[2] = q_cur.z;
      frame_out_buf[out_frame_base + out_idx].q_current[3] = q_cur.w;

      /* 刷新 links[0..current] 反向 + effector 折叠到 links[0] */
      for (int refresh_index = runtime_order; refresh_index >= 0; refresh_index--) {
        LinkConst rlink = link_const_buf[chain.link_offset + refresh_index];
        int bone_idx = rlink.bone;
        int parent = bone_const_buf[bone_idx].parent;
        const FrameOut &ro = frame_out_buf[out_frame_base + rlink.out_index];
        F4 rq = make_f4(ro.q_current[0], ro.q_current[1], ro.q_current[2], ro.q_current[3]);
        F3 rr0, rr1, rr2;
        quat_to_row3(rq, rr0, rr1, rr2);
        F3 m0, m1, m2, m3;
        compose_pivot_mul_parent(frame_buf,
                                 (parent >= 0) ? (frame_base + parent) : -1,
                                 make_f3(bone_const_buf[bone_idx].base_pos[0],
                                         bone_const_buf[bone_idx].base_pos[1],
                                         bone_const_buf[bone_idx].base_pos[2]),
                                 rq,
                                 m0, m1, m2, m3);
        frame_set_m0(frame_buf, frame_base + bone_idx, m0, m1, m2, m3);
      }
      /* effector 折叠到 links[0] */
      {
        int first_link = link_const_buf[chain.link_offset].bone;
        F3 m0, m1, m2, m3;
        frame_get_m0(frame_buf, frame_base + first_link, m0, m1, m2, m3);
        frame_set_m0(frame_buf, frame_base + effector_idx, m0, m1, m2, m3);
      }
    }
  }
}

}  // namespace

/* 主入口：每线程求解一帧 */
extern "C" __global__ void mmd_ccd_bake_kernel(const BoneConst *__restrict__ bone_const_buf,
                                               const ChainConst *__restrict__ chain_const_buf,
                                               const LinkConst *__restrict__ link_const_buf,
                                               FrameBone *__restrict__ frame_buf,
                                               FrameOut *__restrict__ frame_out_buf,
                                               int bone_count,
                                               int chain_count,
                                               int frame_count,
                                               int link_count)
{
  int frame = blockIdx.x * blockDim.x + threadIdx.x;
  if (frame >= frame_count) {
    return;
  }
  int frame_base = frame * bone_count;

  /* 1. q_current 初始 identity（仅链骨有紧凑输出槽位） */
  int out_frame_base = frame * link_count;
  for (int i = 0; i < bone_count; i++) {
    int out_idx = bone_const_buf[i].out_index;
    if (out_idx >= 0) {
      frame_out_buf[out_frame_base + out_idx].q_current[0] = 1.0f;
      frame_out_buf[out_frame_base + out_idx].q_current[1] = 0.0f;
      frame_out_buf[out_frame_base + out_idx].q_current[2] = 0.0f;
      frame_out_buf[out_frame_base + out_idx].q_current[3] = 0.0f;
    }
  }

  /* 2. 骨架 m0 初始化（与 v8.cc 662-693 行对应） */
  for (int i = 0; i < bone_count; i++) {
    BoneConst bc = bone_const_buf[i];
    FrameBone fd = frame_buf[frame_base + i];
    int parent = bc.parent;
    bool is_target = (bc.flags & 1) != 0;
    bool is_anchor = (bc.flags & 4) != 0;
    if (is_target) {
      /* 保持 direct pose matrix（initial_m0） */
      frame_buf[frame_base + i].m0_row0[0] = fd.m0_row0[0];
      frame_buf[frame_base + i].m0_row0[1] = fd.m0_row0[1];
      frame_buf[frame_base + i].m0_row0[2] = fd.m0_row0[2];
      frame_buf[frame_base + i].m0_row0[3] = fd.m0_row0[3];
      frame_buf[frame_base + i].m0_row1[0] = fd.m0_row1[0];
      frame_buf[frame_base + i].m0_row1[1] = fd.m0_row1[1];
      frame_buf[frame_base + i].m0_row1[2] = fd.m0_row1[2];
      frame_buf[frame_base + i].m0_row1[3] = fd.m0_row1[3];
      frame_buf[frame_base + i].m0_row2[0] = fd.m0_row2[0];
      frame_buf[frame_base + i].m0_row2[1] = fd.m0_row2[1];
      frame_buf[frame_base + i].m0_row2[2] = fd.m0_row2[2];
      frame_buf[frame_base + i].m0_row2[3] = fd.m0_row2[3];
      frame_buf[frame_base + i].m0_row3[0] = fd.m0_row3[0];
      frame_buf[frame_base + i].m0_row3[1] = fd.m0_row3[1];
      frame_buf[frame_base + i].m0_row3[2] = fd.m0_row3[2];
      frame_buf[frame_base + i].m0_row3[3] = fd.m0_row3[3];
      continue;
    }
    F3 m0, m1, m2, m3;
    compose_pivot_mul_parent(frame_buf,
                             (parent >= 0) ? (frame_base + parent) : -1,
                             make_f3(bc.base_pos[0], bc.base_pos[1], bc.base_pos[2]),
                             make_f4(fd.q_base[0], fd.q_base[1], fd.q_base[2], fd.q_base[3]),
                             m0, m1, m2, m3);
    if (is_anchor) {
      /* 独立链根父：旋转用 q_base 传播，平移用 direct pose head */
      F3 head_mmd = transform_point(make_f3(bc.base_pos[0], bc.base_pos[1], bc.base_pos[2]),
                                    make_f3(fd.m0_row0[0], fd.m0_row0[1], fd.m0_row0[2]),
                                    make_f3(fd.m0_row1[0], fd.m0_row1[1], fd.m0_row1[2]),
                                    make_f3(fd.m0_row2[0], fd.m0_row2[1], fd.m0_row2[2]),
                                    make_f3(fd.m0_row3[0], fd.m0_row3[1], fd.m0_row3[2]));
      F3 base_times_r = make_f3(
          dot3(make_f3(bc.base_pos[0], bc.base_pos[1], bc.base_pos[2]),
               make_f3(m0.x, m1.x, m2.x)),
          dot3(make_f3(bc.base_pos[0], bc.base_pos[1], bc.base_pos[2]),
               make_f3(m0.y, m1.y, m2.y)),
          dot3(make_f3(bc.base_pos[0], bc.base_pos[1], bc.base_pos[2]),
               make_f3(m0.z, m1.z, m2.z)));
      m3 = f3_sub(head_mmd, base_times_r);
    }
    frame_set_m0(frame_buf, frame_base + i, m0, m1, m2, m3);
  }

  /* 3. 按顺序求解每条链 */
  for (int c = 0; c < chain_count; c++) {
    solve_chain_v8(bone_const_buf, link_const_buf, frame_buf, frame_out_buf, chain_const_buf[c],
                   frame_base, out_frame_base);
  }
}

/* 主机启动器：由 mmd_ccd_ik_bake.cc 调用（nvcc 负责 fatbin 注册与启动）。 */
extern "C" cudaError_t mmd_ccd_bake_launch(const BoneConst *bones,
                                           const ChainConst *chains,
                                           const LinkConst *links,
                                           FrameBone *frames,
                                           FrameOut *out,
                                           int bone_count,
                                           int chain_count,
                                           int frame_count,
                                           int link_count)
{
  const int block = 64;
  int grid = (frame_count + block - 1) / block;
  if (grid < 1) {
    grid = 1;
  }
  mmd_ccd_bake_kernel<<<grid, block>>>(
      bones, chains, links, frames, out, bone_count, chain_count, frame_count, link_count);
  return cudaGetLastError();
}

}  // namespace blender::mmd::bake
