/* SPDX-FileCopyrightText: 2026 MuAnChen
 *
 * SPDX-License-Identifier: GPL-2.0-or-later */

/** \file
 * \ingroup mmd_ccd_ik
 *
 * MMD CCD 批量烘焙的 CUDA 后端（R9-CUDA）。与
 * gpu_shader_mmd_ccd_bake_comp.glsl 逐行对应：每线程求解一帧，帧内链按
 * CPU v8 相同顺序求解。数据布局与 mmd_bake_gpu_types.hh 逐字节一致。
 *
 * 内核按标量类型 T 模板化并实例化两套：
 * - T=float：逐位与 GLSL/CPU float 参照一致（默认路径）。
 * - T=double：FP64 双精度求解（MMD_BAKE_FP64=1）。四元数与 m0 矩阵全程
 *   double，迭代过程不再被 float 存储反复舍入——q_current/m0 使用独立
 *   设备端 double 缓存，仅在输入（q_base/initial m0）与最终输出处做
 *   float<->double 转换。
 */

#include "mmd_bake_gpu_types.hh"

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace blender::mmd::bake {

namespace {

template<typename T> constexpr T kEps = T(1.0e-10);

/* 与 GLSL float3/float4 对应的轻量向量（避免依赖 CUDA 向量库语义）。 */
template<typename T> struct V3 {
  T x, y, z;
};
template<typename T> struct V4 {
  T x, y, z, w;
};

/* FP64 模式的设备端私有缓存（不改变对外打包布局）。 */
struct M0Row {
  double r0[4];
  double r1[4];
  double r2[4];
  double r3[4];
};
struct QRow {
  double q[4];
};

template<typename T> __device__ __forceinline__ V3<T> make_v3(T x, T y, T z)
{
  V3<T> v;
  v.x = x;
  v.y = y;
  v.z = z;
  return v;
}

template<typename T> __device__ __forceinline__ V4<T> make_v4(T x, T y, T z, T w)
{
  V4<T> v;
  v.x = x;
  v.y = y;
  v.z = z;
  v.w = w;
  return v;
}

template<typename T> __device__ __forceinline__ V3<T> v3_add(V3<T> a, V3<T> b)
{
  return make_v3(a.x + b.x, a.y + b.y, a.z + b.z);
}
template<typename T> __device__ __forceinline__ V3<T> v3_sub(V3<T> a, V3<T> b)
{
  return make_v3(a.x - b.x, a.y - b.y, a.z - b.z);
}
template<typename T> __device__ __forceinline__ V3<T> v3_scale(V3<T> a, T s)
{
  return make_v3(a.x * s, a.y * s, a.z * s);
}
template<typename T> __device__ __forceinline__ V4<T> v4_scale(V4<T> a, T s)
{
  return make_v4(a.x * s, a.y * s, a.z * s, a.w * s);
}
template<typename T> __device__ __forceinline__ T dot3(V3<T> a, V3<T> b)
{
  return a.x * b.x + a.y * b.y + a.z * b.z;
}
template<typename T> __device__ __forceinline__ V3<T> cross3(V3<T> a, V3<T> b)
{
  return make_v3(a.y * b.z - a.z * b.y,
                 a.z * b.x - a.x * b.z,
                 a.x * b.y - a.y * b.x);
}
template<typename T> __device__ __forceinline__ T len3(V3<T> v)
{
  return sqrt(dot3(v, v));
}

/* quat (w, x, y, z) 归一化（与 v8.cc 一致：先取倒数再乘） */
template<typename T> __device__ V4<T> quat_normalize(V4<T> q)
{
  T len = sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
  if (len < kEps<T>) {
    return make_v4(T(1), T(0), T(0), T(0));
  }
  T inv = T(1) / len;
  return v4_scale(q, inv);
}

/* 数学 a * b（Hamilton） */
template<typename T> __device__ V4<T> quat_mul(V4<T> a, V4<T> b)
{
  return quat_normalize(make_v4(a.x * b.x - a.y * b.y - a.z * b.z - a.w * b.w,
                                a.x * b.y + a.y * b.x + a.z * b.w - a.w * b.z,
                                a.x * b.z - a.y * b.w + a.z * b.x + a.w * b.y,
                                a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x));
}

/* 四元数 → row-major 3x3（m[c] = 第 c 行） */
template<typename T>
__device__ void quat_to_row3(V4<T> q_in, V3<T> &m0, V3<T> &m1, V3<T> &m2)
{
  V4<T> q = quat_normalize(q_in);
  T w = q.x, x = q.y, y = q.z, z = q.w;
  T c00 = T(1) - T(2) * (y * y + z * z);
  T c01 = T(2) * (x * y - z * w);
  T c02 = T(2) * (x * z + y * w);
  T c10 = T(2) * (x * y + z * w);
  T c11 = T(1) - T(2) * (x * x + z * z);
  T c12 = T(2) * (y * z - x * w);
  T c20 = T(2) * (x * z - y * w);
  T c21 = T(2) * (y * z + x * w);
  T c22 = T(1) - T(2) * (x * x + y * y);
  m0 = make_v3(c00, c10, c20);
  m1 = make_v3(c01, c11, c21);
  m2 = make_v3(c02, c12, c22);
}

/* row-major 3x3 → 四元数 */
template<typename T> __device__ V4<T> quat_from_row3(V3<T> r0, V3<T> r1, V3<T> r2)
{
  T m00 = r0.x, m01 = r1.x, m02 = r2.x;
  T m10 = r0.y, m11 = r1.y, m12 = r2.y;
  T m20 = r0.z, m21 = r1.z, m22 = r2.z;
  T trace = m00 + m11 + m22;
  V4<T> q;
  if (trace > T(0)) {
    T s = sqrt(trace + T(1)) * T(2);
    q = make_v4(T(0.25) * s, (m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s);
  }
  else if (m00 > m11 && m00 > m22) {
    T s = sqrt(fmax(T(0), T(1) + m00 - m11 - m22)) * T(2);
    q = make_v4((m21 - m12) / s, T(0.25) * s, (m01 + m10) / s, (m02 + m20) / s);
  }
  else if (m11 > m22) {
    T s = sqrt(fmax(T(0), T(1) + m11 - m00 - m22)) * T(2);
    q = make_v4((m02 - m20) / s, (m01 + m10) / s, T(0.25) * s, (m12 + m21) / s);
  }
  else {
    T s = sqrt(fmax(T(0), T(1) + m22 - m00 - m11)) * T(2);
    q = make_v4((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, T(0.25) * s);
  }
  return quat_normalize(q);
}

/* 绕单轴旋转的 row-major 3x3 */
template<typename T>
__device__ void axis_rotation_row(int axis, T angle, V3<T> &m0, V3<T> &m1, V3<T> &m2)
{
  V4<T> q = make_v4(T(1), T(0), T(0), T(0));
  T half_angle = angle * T(0.5);
  q.x = cos(half_angle);
  if (axis == 0) {
    q.y = sin(half_angle);
  }
  else if (axis == 1) {
    q.z = sin(half_angle);
  }
  else {
    q.w = sin(half_angle);
  }
  quat_to_row3(q, m0, m1, m2);
}

template<typename T>
__device__ void mul3_rows(
    V3<T> a0, V3<T> a1, V3<T> a2, V3<T> b0, V3<T> b1, V3<T> b2, V3<T> &r0, V3<T> &r1, V3<T> &r2)
{
  /* result[r][c] = sum(a[r][k] * b[k][c]) */
  r0 = make_v3(dot3(a0, make_v3(b0.x, b1.x, b2.x)),
               dot3(a0, make_v3(b0.y, b1.y, b2.y)),
               dot3(a0, make_v3(b0.z, b1.z, b2.z)));
  r1 = make_v3(dot3(a1, make_v3(b0.x, b1.x, b2.x)),
               dot3(a1, make_v3(b0.y, b1.y, b2.y)),
               dot3(a1, make_v3(b0.z, b1.z, b2.z)));
  r2 = make_v3(dot3(a2, make_v3(b0.x, b1.x, b2.x)),
               dot3(a2, make_v3(b0.y, b1.y, b2.y)),
               dot3(a2, make_v3(b0.z, b1.z, b2.z)));
}

/* -------------------------------------------------------------------- */
/* m0 存储抽象：float 路径就地读写 FrameBone；FP64 路径读写 double 缓存 */
/* -------------------------------------------------------------------- */

__device__ __forceinline__ void m0_get(
    const FrameBone *frame_buf, int idx, V3<float> &m0, V3<float> &m1, V3<float> &m2, V3<float> &m3)
{
  const FrameBone &d = frame_buf[idx];
  m0 = make_v3(d.m0_row0[0], d.m0_row0[1], d.m0_row0[2]);
  m1 = make_v3(d.m0_row1[0], d.m0_row1[1], d.m0_row1[2]);
  m2 = make_v3(d.m0_row2[0], d.m0_row2[1], d.m0_row2[2]);
  m3 = make_v3(d.m0_row3[0], d.m0_row3[1], d.m0_row3[2]);
}

/* float 输入 → double 转换读取（FP64 模式的初始 m0/直姿态读取）。 */
__device__ __forceinline__ void m0_get(
    const FrameBone *frame_buf, int idx, V3<double> &m0, V3<double> &m1, V3<double> &m2, V3<double> &m3)
{
  const FrameBone &d = frame_buf[idx];
  m0 = make_v3(double(d.m0_row0[0]), double(d.m0_row0[1]), double(d.m0_row0[2]));
  m1 = make_v3(double(d.m0_row1[0]), double(d.m0_row1[1]), double(d.m0_row1[2]));
  m2 = make_v3(double(d.m0_row2[0]), double(d.m0_row2[1]), double(d.m0_row2[2]));
  m3 = make_v3(double(d.m0_row3[0]), double(d.m0_row3[1]), double(d.m0_row3[2]));
}

__device__ __forceinline__ void m0_get(
    const M0Row *cache, int idx, V3<double> &m0, V3<double> &m1, V3<double> &m2, V3<double> &m3)
{
  const M0Row &d = cache[idx];
  m0 = make_v3(d.r0[0], d.r0[1], d.r0[2]);
  m1 = make_v3(d.r1[0], d.r1[1], d.r1[2]);
  m2 = make_v3(d.r2[0], d.r2[1], d.r2[2]);
  m3 = make_v3(d.r3[0], d.r3[1], d.r3[2]);
}

__device__ __forceinline__ void m0_set(
    FrameBone *frame_buf, int idx, V3<float> m0, V3<float> m1, V3<float> m2, V3<float> m3)
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

__device__ __forceinline__ void m0_set(
    M0Row *cache, int idx, V3<double> m0, V3<double> m1, V3<double> m2, V3<double> m3)
{
  M0Row &d = cache[idx];
  d.r0[0] = m0.x;
  d.r0[1] = m0.y;
  d.r0[2] = m0.z;
  d.r0[3] = 0.0;
  d.r1[0] = m1.x;
  d.r1[1] = m1.y;
  d.r1[2] = m1.z;
  d.r1[3] = 0.0;
  d.r2[0] = m2.x;
  d.r2[1] = m2.y;
  d.r2[2] = m2.z;
  d.r2[3] = 0.0;
  d.r3[0] = m3.x;
  d.r3[1] = m3.y;
  d.r3[2] = m3.z;
  d.r3[3] = 1.0;
}

/* -------------------------------------------------------------------- */
/* q_current 存储抽象（同 m0：float 就地 / FP64 独立缓存）               */
/* -------------------------------------------------------------------- */

__device__ __forceinline__ void q_get(const FrameOut *out, int idx, V4<float> &q)
{
  q = make_v4(out[idx].q_current[0], out[idx].q_current[1], out[idx].q_current[2],
              out[idx].q_current[3]);
}

__device__ __forceinline__ void q_get(const QRow *cache, int idx, V4<double> &q)
{
  q = make_v4(cache[idx].q[0], cache[idx].q[1], cache[idx].q[2], cache[idx].q[3]);
}

__device__ __forceinline__ void q_set(FrameOut *out, int idx, V4<float> q)
{
  out[idx].q_current[0] = q.x;
  out[idx].q_current[1] = q.y;
  out[idx].q_current[2] = q.z;
  out[idx].q_current[3] = q.w;
}

__device__ __forceinline__ void q_set(QRow *cache, int idx, V4<double> q)
{
  cache[idx].q[0] = q.x;
  cache[idx].q[1] = q.y;
  cache[idx].q[2] = q.z;
  cache[idx].q[3] = q.w;
}

/* row-vector 点变换：result[c] = sum(point[k] * M[k][c]) + M[3][c] */
template<typename T>
__device__ V3<T> transform_point(V3<T> point, V3<T> m0, V3<T> m1, V3<T> m2, V3<T> m3)
{
  return make_v3(dot3(point, make_v3(m0.x, m1.x, m2.x)) + m3.x,
                 dot3(point, make_v3(m0.y, m1.y, m2.y)) + m3.y,
                 dot3(point, make_v3(m0.z, m1.z, m2.z)) + m3.z);
}

/* 列向量左乘 row-major M：result[r] = sum(M[r][k] * v[k]) */
template<typename T> __device__ V3<T> mul_mat_vec(V3<T> m0, V3<T> m1, V3<T> m2, V3<T> v)
{
  return make_v3(dot3(m0, v), dot3(m1, v), dot3(m2, v));
}

/* pivot_matrix (T(pos)*R*T(-pos) 的 row-major 形式) */
template<typename T>
__device__ void pivot_matrix(V3<T> position, V3<T> r0, V3<T> r1, V3<T> r2, V3<T> &m0, V3<T> &m1, V3<T> &m2, V3<T> &m3)
{
  m0 = r0;
  m1 = r1;
  m2 = r2;
  /* trans[c] = position[c] - sum(position[k] * rotation[k][c])（点乘列） */
  m3 = make_v3(position.x - dot3(position, make_v3(r0.x, r1.x, r2.x)),
               position.y - dot3(position, make_v3(r0.y, r1.y, r2.y)),
               position.z - dot3(position, make_v3(r0.z, r1.z, r2.z)));
}

/* 四元数 → pivot 矩阵 + 与父矩阵层级组合：m0[bone] = m1[bone] * m0[parent] */
template<typename T, typename Store>
__device__ void compose_pivot_mul_parent(Store *m0_store,
                                         int parent,
                                         V3<T> base_pos,
                                         V4<T> q,
                                         V3<T> &m0,
                                         V3<T> &m1,
                                         V3<T> &m2,
                                         V3<T> &m3)
{
  V3<T> r0, r1, r2;
  quat_to_row3(q, r0, r1, r2);
  V3<T> p0, p1, p2, p3;
  pivot_matrix(base_pos, r0, r1, r2, p0, p1, p2, p3);
  if (parent >= 0) {
    V3<T> pm0, pm1, pm2, pm3;
    m0_get(m0_store, parent, pm0, pm1, pm2, pm3);
    /* mul4: result[r][c] = sum(a[r][k] * b[k][c])
     * 平移行 result[3][c] = dot(a3, b_col_c) + b[3][c]（a[3][3]=1）。 */
    mul3_rows(p0, p1, p2, pm0, pm1, pm2, m0, m1, m2);
    m3 = make_v3(dot3(p3, make_v3(pm0.x, pm1.x, pm2.x)) + pm3.x,
                 dot3(p3, make_v3(pm0.y, pm1.y, pm2.y)) + pm3.y,
                 dot3(p3, make_v3(pm0.z, pm1.z, pm2.z)) + pm3.z);
  }
  else {
    m0 = p0;
    m1 = p1;
    m2 = p2;
    m3 = p3;
  }
}

/* Euler limit 分支（与 v8.cc 逐行对应） */
template<typename T> __device__ int branch_for_limits(V3<T> limit_min, V3<T> limit_max)
{
  if (limit_min.x <= T(-M_PI * 0.5) || limit_max.x >= T(M_PI * 0.5)) {
    if (limit_min.y <= T(-M_PI * 0.5) || limit_max.y >= T(M_PI * 0.5)) {
      return 0; /* 'A' */
    }
    return 1; /* 'B' */
  }
  return 2; /* 'C' */
}

template<typename T>
__device__ void decompose_branch_euler(int branch, V3<T> r0, V3<T> r1, V3<T> r2, V3<T> &angles)
{
  if (branch == 0) {
    T z = asin(fmin(T(1), fmax(T(-1), -r1.x)));
    T cz = cos(z);
    if (fabs(cz) < T(1.0e-8)) {
      angles = make_v3(T(0), atan2(-r0.z, r2.z), z);
    }
    else {
      angles = make_v3(atan2(r1.z, r1.y), atan2(r2.x, r0.x), z);
    }
  }
  else if (branch == 1) {
    T y = asin(fmin(T(1), fmax(T(-1), -r0.z)));
    T cy = cos(y);
    if (fabs(cy) < T(1.0e-8)) {
      angles = make_v3(atan2(-r2.y, r1.y), y, T(0));
    }
    else {
      angles = make_v3(atan2(r1.z, r2.z), y, atan2(r0.y, r0.x));
    }
  }
  else {
    T x = asin(fmin(T(1), fmax(T(-1), -r2.y)));
    T cx = cos(x);
    if (fabs(cx) < T(1.0e-8)) {
      angles = make_v3(x, T(0), atan2(-r1.x, r0.x));
    }
    else {
      angles = make_v3(x, atan2(r2.x, r2.z), atan2(r0.y, r1.y));
    }
  }
}

template<typename T>
__device__ void compose_branch_euler(int branch, V3<T> angles, V3<T> &m0, V3<T> &m1, V3<T> &m2)
{
  V3<T> rx0, rx1, rx2, ry0, ry1, ry2, rz0, rz1, rz2;
  axis_rotation_row(0, angles.x, rx0, rx1, rx2);
  axis_rotation_row(1, angles.y, ry0, ry1, ry2);
  axis_rotation_row(2, angles.z, rz0, rz1, rz2);
  if (branch == 0) {
    /* Ry * Rz * Rx */
    V3<T> t0, t1, t2;
    mul3_rows(ry0, ry1, ry2, rz0, rz1, rz2, t0, t1, t2);
    mul3_rows(t0, t1, t2, rx0, rx1, rx2, m0, m1, m2);
  }
  else if (branch == 1) {
    /* Rx * Ry * Rz */
    V3<T> t0, t1, t2;
    mul3_rows(rx0, rx1, rx2, ry0, ry1, ry2, t0, t1, t2);
    mul3_rows(t0, t1, t2, rz0, rz1, rz2, m0, m1, m2);
  }
  else {
    /* Rz * Rx * Ry */
    V3<T> t0, t1, t2;
    mul3_rows(rz0, rz1, rz2, rx0, rx1, rx2, t0, t1, t2);
    mul3_rows(t0, t1, t2, ry0, ry1, ry2, m0, m1, m2);
  }
}

template<typename T>
__device__ V4<T> apply_mmd_link_limit(
    V3<T> limit_min, V3<T> limit_max, V4<T> q, int iteration_index, int loop_count)
{
  int branch = branch_for_limits(limit_min, limit_max);
  V3<T> r0, r1, r2;
  quat_to_row3(q, r0, r1, r2);
  V3<T> angles;
  decompose_branch_euler(branch, r0, r1, r2, angles);

  bool hard_clamp = (loop_count / 2) <= iteration_index;
  T a[3] = {angles.x, angles.y, angles.z};
  T lo[3] = {limit_min.x, limit_min.y, limit_min.z};
  T hi[3] = {limit_max.x, limit_max.y, limit_max.z};
  for (int axis = 0; axis < 3; axis++) {
    T value = a[axis];
    T low = lo[axis];
    T high = hi[axis];
    if (value < low) {
      T reflected = low * T(2) - value;
      if (hard_clamp || reflected < low || reflected > high) {
        value = low;
      }
      else {
        value = reflected;
      }
    }
    else if (value > high) {
      T reflected = high * T(2) - value;
      if (hard_clamp || reflected < low || reflected > high) {
        value = high;
      }
      else {
        value = reflected;
      }
    }
    a[axis] = value;
  }
  V3<T> c0, c1, c2;
  compose_branch_euler(branch, make_v3(a[0], a[1], a[2]), c0, c1, c2);
  return quat_from_row3(c0, c1, c2);
}

/* 单链求解（与 solve_chain_v8 逐行对应；m0/q_current 经存储抽象读写，
 * FP64 模式下全程 double，迭代过程无 float 舍入）。 */
template<typename T, typename Store, typename QStore>
__device__ void solve_chain_v8(const BoneConst *bone_const_buf,
                               const LinkConst *link_const_buf,
                               Store *m0_store,
                               QStore *q_store,
                               ChainConst chain,
                               int frame_base,
                               int out_frame_base)
{
  int target_idx = chain.target;
  int effector_idx = chain.effector;
  int iterations = chain.iterations;
  int half_iter = iterations / 2;
  T ik_angle = T(chain.runtime_angle);

  /* target 世界位置：m0[target] 用 initial_m0（不传播） */
  V3<T> tm0, tm1, tm2, tm3;
  m0_get(m0_store, frame_base + target_idx, tm0, tm1, tm2, tm3);
  V3<T> target = transform_point(
      make_v3(T(bone_const_buf[target_idx].base_pos[0]),
              T(bone_const_buf[target_idx].base_pos[1]),
              T(bone_const_buf[target_idx].base_pos[2])),
      tm0, tm1, tm2, tm3);

  for (int iteration_index = 0; iteration_index < iterations; iteration_index++) {
    for (int runtime_order = 0; runtime_order < chain.link_count; runtime_order++) {
      LinkConst link = link_const_buf[chain.link_offset + runtime_order];
      int link_idx = link.bone;

      V3<T> jm0, jm1, jm2, jm3;
      m0_get(m0_store, frame_base + link_idx, jm0, jm1, jm2, jm3);
      V3<T> joint = transform_point(make_v3(T(bone_const_buf[link_idx].base_pos[0]),
                                            T(bone_const_buf[link_idx].base_pos[1]),
                                            T(bone_const_buf[link_idx].base_pos[2])),
                                    jm0, jm1, jm2, jm3);

      V3<T> em0, em1, em2, em3;
      m0_get(m0_store, frame_base + effector_idx, em0, em1, em2, em3);
      V3<T> effector = transform_point(make_v3(T(bone_const_buf[effector_idx].base_pos[0]),
                                               T(bone_const_buf[effector_idx].base_pos[1]),
                                               T(bone_const_buf[effector_idx].base_pos[2])),
                                       em0, em1, em2, em3);

      V3<T> to_effector_raw = v3_sub(effector, joint);
      T to_eff_len = len3(to_effector_raw);
      if (to_eff_len < kEps<T>) {
        continue;
      }
      V3<T> to_effector = v3_scale(to_effector_raw, T(1) / to_eff_len);

      V3<T> to_target_raw = v3_sub(target, joint);
      T to_tgt_len = len3(to_target_raw);
      if (to_tgt_len < kEps<T>) {
        continue;
      }
      V3<T> to_target = v3_scale(to_target_raw, T(1) / to_tgt_len);

      /* 方向收敛检查 */
      V3<T> diff = v3_sub(to_target, to_effector);
      if (dot3(diff, diff) < T(1.0e-7)) {
        continue;
      }

      V3<T> axis_world = cross3(to_effector, to_target);

      int parent_idx = bone_const_buf[link_idx].parent;
      V3<T> pm0, pm1, pm2, pm3;
      m0_get(m0_store, frame_base + parent_idx, pm0, pm1, pm2, pm3);
      V3<T> cl = mul_mat_vec(pm0, pm1, pm2, axis_world);

      V3<T> axis_local;
      if (link.has_limit != 0 && iteration_index < half_iter) {
        T lminx = T(link.limit_min[0]), lminy = T(link.limit_min[1]), lminz = T(link.limit_min[2]);
        T lmaxx = T(link.limit_max[0]), lmaxy = T(link.limit_max[1]), lmaxz = T(link.limit_max[2]);
        if (lminy == T(0) && lmaxy == T(0) && lminz == T(0) && lmaxz == T(0)) {
          axis_local = make_v3((cl.x >= T(0)) ? T(1) : T(-1), T(0), T(0));
        }
        else if (lminx == T(0) && lmaxx == T(0) && lminz == T(0) && lmaxz == T(0)) {
          axis_local = make_v3(T(0), (cl.y >= T(0)) ? T(1) : T(-1), T(0));
        }
        else {
          axis_local = make_v3(T(0), T(0), (cl.z >= T(0)) ? T(1) : T(-1));
        }
      }
      else {
        T cl_len = len3(cl);
        if (cl_len < kEps<T>) {
          continue;
        }
        axis_local = v3_scale(cl, T(1) / cl_len);
      }

      T cosine = fmin(T(1), fmax(T(-1), dot3(to_effector, to_target)));
      T half_angle = T(0.5) * acos(cosine);

      T cap = T(runtime_order + 1) * ik_angle * T(2);
      half_angle = fmin(cap, fmax(-cap, half_angle));

      V4<T> delta = make_v4(cos(half_angle),
                            axis_local.x * sin(half_angle),
                            axis_local.y * sin(half_angle),
                            axis_local.z * sin(half_angle));

      /* D3DX 反序：q_cur = delta * q_cur（紧凑输出槽位）。
       * q_current 已初始化为 q_base（完整 FK 旋转），左乘 delta 即完整
       * "FK + CCD 修正"旋转；旧语义从 identity 起步、首轮再吸收 q_base，
       * 导致首轮即收敛的 link 输出恒为 identity。 */
      int out_idx = link.out_index;
      V4<T> q_cur;
      q_get(q_store, out_frame_base + out_idx, q_cur);
      q_cur = quat_mul(delta, q_cur);

      if (link.has_limit != 0) {
        q_cur = apply_mmd_link_limit(
            make_v3(T(link.limit_min[0]), T(link.limit_min[1]), T(link.limit_min[2])),
            make_v3(T(link.limit_max[0]), T(link.limit_max[1]), T(link.limit_max[2])),
            q_cur,
            iteration_index,
            iterations);
      }

      q_set(q_store, out_frame_base + out_idx, q_cur);

      /* 刷新 links[0..current] 反向 + effector 折叠到 links[0] */
      for (int refresh_index = runtime_order; refresh_index >= 0; refresh_index--) {
        LinkConst rlink = link_const_buf[chain.link_offset + refresh_index];
        int bone_idx = rlink.bone;
        int parent = bone_const_buf[bone_idx].parent;
        V4<T> rq;
        q_get(q_store, out_frame_base + rlink.out_index, rq);
        V3<T> rr0, rr1, rr2;
        quat_to_row3(rq, rr0, rr1, rr2);
        V3<T> m0, m1, m2, m3;
        compose_pivot_mul_parent(m0_store,
                                 (parent >= 0) ? (frame_base + parent) : -1,
                                 make_v3(T(bone_const_buf[bone_idx].base_pos[0]),
                                         T(bone_const_buf[bone_idx].base_pos[1]),
                                         T(bone_const_buf[bone_idx].base_pos[2])),
                                 rq,
                                 m0, m1, m2, m3);
        m0_set(m0_store, frame_base + bone_idx, m0, m1, m2, m3);
      }
      /* effector 折叠到 links[0] */
      {
        int first_link = link_const_buf[chain.link_offset].bone;
        V3<T> m0, m1, m2, m3;
        m0_get(m0_store, frame_base + first_link, m0, m1, m2, m3);
        m0_set(m0_store, frame_base + effector_idx, m0, m1, m2, m3);
      }
    }
  }
}

}  // namespace

/* 主入口：每线程求解一帧（T=float/double 双实例）。 */
template<typename T, typename Store, typename QStore>
__global__ void mmd_ccd_bake_kernel_t(const BoneConst *__restrict__ bone_const_buf,
                                      const ChainConst *__restrict__ chain_const_buf,
                                      const LinkConst *__restrict__ link_const_buf,
                                      const FrameBone *__restrict__ frame_buf,
                                      Store *__restrict__ m0_store,
                                      QStore *__restrict__ q_store,
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

  /* 1. q_current 初始为 q_base（仅链骨有紧凑输出槽位） */
  int out_frame_base = frame * link_count;
  for (int i = 0; i < bone_count; i++) {
    int out_idx = bone_const_buf[i].out_index;
    if (out_idx >= 0) {
      const FrameBone &fb = frame_buf[frame_base + i];
      q_set(q_store,
            out_frame_base + out_idx,
            make_v4(T(fb.q_base[0]), T(fb.q_base[1]), T(fb.q_base[2]), T(fb.q_base[3])));
    }
  }

  /* 2. 骨架 m0 初始化（与 v8.cc 662-693 行对应） */
  for (int i = 0; i < bone_count; i++) {
    BoneConst bc = bone_const_buf[i];
    int parent = bc.parent;
    bool is_target = (bc.flags & 1) != 0;
    bool is_anchor = (bc.flags & 4) != 0;
    if (is_target) {
      /* 保持 direct pose matrix（initial_m0） */
      V3<T> t0, t1, t2, t3;
      m0_get(frame_buf, frame_base + i, t0, t1, t2, t3);
      m0_set(m0_store, frame_base + i, t0, t1, t2, t3);
      continue;
    }
    const FrameBone &fd = frame_buf[frame_base + i];
    V3<T> m0, m1, m2, m3;
    compose_pivot_mul_parent(m0_store,
                             (parent >= 0) ? (frame_base + parent) : -1,
                             make_v3(T(bc.base_pos[0]), T(bc.base_pos[1]), T(bc.base_pos[2])),
                             make_v4(T(fd.q_base[0]), T(fd.q_base[1]), T(fd.q_base[2]), T(fd.q_base[3])),
                             m0, m1, m2, m3);
    if (is_anchor) {
      /* 独立链根父：旋转用 q_base 传播，平移用 direct pose head */
      V3<T> dm0, dm1, dm2, dm3;
      m0_get(frame_buf, frame_base + i, dm0, dm1, dm2, dm3);
      V3<T> head_mmd = transform_point(
          make_v3(T(bc.base_pos[0]), T(bc.base_pos[1]), T(bc.base_pos[2])), dm0, dm1, dm2, dm3);
      V3<T> base_times_r = make_v3(
          dot3(make_v3(T(bc.base_pos[0]), T(bc.base_pos[1]), T(bc.base_pos[2])),
               make_v3(m0.x, m1.x, m2.x)),
          dot3(make_v3(T(bc.base_pos[0]), T(bc.base_pos[1]), T(bc.base_pos[2])),
               make_v3(m0.y, m1.y, m2.y)),
          dot3(make_v3(T(bc.base_pos[0]), T(bc.base_pos[1]), T(bc.base_pos[2])),
               make_v3(m0.z, m1.z, m2.z)));
      m3 = v3_sub(head_mmd, base_times_r);
    }
    m0_set(m0_store, frame_base + i, m0, m1, m2, m3);
  }

  /* 3. 按顺序求解每条链 */
  for (int c = 0; c < chain_count; c++) {
    solve_chain_v8<T, Store, QStore>(bone_const_buf,
                                     link_const_buf,
                                     m0_store,
                                     q_store,
                                     chain_const_buf[c],
                                     frame_base,
                                     out_frame_base);
  }

  /* 4. FP64 模式：把 double q_current 一次性写回 float 紧凑输出；float
   * 模式就地自拷贝（由 if constexpr 跳过）。 */
  if constexpr (sizeof(T) > 4) {
    for (int li = 0; li < link_count; li++) {
      V4<T> q;
      q_get(q_store, out_frame_base + li, q);
      frame_out_buf[out_frame_base + li].q_current[0] = float(q.x);
      frame_out_buf[out_frame_base + li].q_current[1] = float(q.y);
      frame_out_buf[out_frame_base + li].q_current[2] = float(q.z);
      frame_out_buf[out_frame_base + li].q_current[3] = float(q.w);
    }
  }
}

/* 主机启动器：由 mmd_ccd_ik_bake.cc 调用（nvcc 负责 fatbin 注册与启动）。
 * use_fp64 时 m0_cache/q_cache 必须是 frame×bone / frame×link 的 double
 * 缓存（主机分配）；float 路径忽略并就地使用 frames/out。 */
extern "C" cudaError_t mmd_ccd_bake_launch(const BoneConst *bones,
                                           const ChainConst *chains,
                                           const LinkConst *links,
                                           FrameBone *frames,
                                           void *m0_cache,
                                           void *q_cache,
                                           FrameOut *out,
                                           int bone_count,
                                           int chain_count,
                                           int frame_count,
                                           int link_count,
                                           cudaStream_t stream,
                                           int use_fp64)
{
  const int block = 64;
  int grid = (frame_count + block - 1) / block;
  if (grid < 1) {
    grid = 1;
  }
  if (use_fp64) {
    mmd_ccd_bake_kernel_t<double, M0Row, QRow><<<grid, block, 0, stream>>>(
        bones,
        chains,
        links,
        frames,
        static_cast<M0Row *>(m0_cache),
        static_cast<QRow *>(q_cache),
        out,
        bone_count,
        chain_count,
        frame_count,
        link_count);
  }
  else {
    mmd_ccd_bake_kernel_t<float, FrameBone, FrameOut><<<grid, block, 0, stream>>>(
        bones, chains, links, frames, frames, out, out, bone_count, chain_count, frame_count,
        link_count);
  }
  return cudaGetLastError();
}

}  // namespace blender::mmd::bake
