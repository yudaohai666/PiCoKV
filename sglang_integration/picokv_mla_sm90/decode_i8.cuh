// PiCoKV b1280 code-space decode attention for MLA latent pools (GLM-4.7-Flash, 20 heads) on SM90, path B:
//   qproj:   q' = q_nope @ dec^T, quantised per head to int8 hi / lo terms (the q record of a row)
//   attn:    per unit of 128 compressed entries: int8 score (u8 codes x s8 q', + bf16 rope), P = fp16 2^(s - R),
//            Y^T [bits, heads] += codes^T @ P^T (fp16, fp32 accumulators across the units of a segment)
//   combine: the (CTA, row) segments of a row -> A'' = (2 y - L) / L (fp16) and lse_c (log2 units)
//   merge:   o_c = A'' @ dec, merged by lse with the dense part (sink + tail: the attn kernel's dense role, one CTA per
//            row ahead of the code CTAs; or fa3 with lse)
// Entry e of a row (position SINK + e) lives in slot r2t[rpi, SINK + e] of the layer arena (288 B per slot: 40 code
// words, bf16 rope at byte 160). All counts are on the device (plan), graph-safe.
#pragma once

#include <dlpack/dlpack.h>
#include <tvm/ffi/container/tensor.h>

#include <cuda.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace sglang {
namespace pki8 {
constexpr int H = 20, HN = 24, BITS = 1280, WORDS = 40, SLOT_B = 288, SINK = 64, TILE = 64;
constexpr int UE = 128;                  // entries per unit (2 tiles)
constexpr int NQ = 2 * HN;               // int8 wgmma N: (head, term)
constexpr int QCH = NQ * 128;            // SW128 chunk of the q image: [48][128 positions] s8
constexpr int NQCH = BITS / 128;
constexpr int QR_B = HN * 128;           // q_rope [24][64] bf16 SW128
constexpr int QREC = NQCH * QCH + QR_B + 2 * HN * 4;  // per row: q image | q_rope | sum X (int32) | 1 / K (float)
constexpr int PU_B = NQ * UE;            // P of a unit: [48][128 entries] u8 SW128 (p = 256 hi + lo = 65535 P)
constexpr uint32_t M01 = 0x01010101u;

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return static_cast<uint32_t>(__cvta_generic_to_shared(p)); }
__device__ __forceinline__ uint32_t lds32(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared.b32 %0, [%1];\n" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void sts8(uint32_t a, uint32_t v) {
  asm volatile("st.shared.b8 [%0], %1;\n" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void sts128z(uint32_t a) {
  asm volatile("st.shared.v4.b32 [%0], {%1, %1, %1, %1};\n" ::"r"(a), "r"(0) : "memory");
}
__device__ __forceinline__ void named_bar(int id, int n) { asm volatile("bar.sync %0, %1;\n" ::"r"(id), "r"(n) : "memory"); }
__device__ __forceinline__ void griddep_wait() { asm volatile("griddepcontrol.wait;\n" ::: "memory"); }
__device__ __forceinline__ void griddep_launch() { asm volatile("griddepcontrol.launch_dependents;\n" ::: "memory"); }
__device__ __forceinline__ void fence_async_smem() { asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory"); }

__device__ __forceinline__ void mbar_init(uint32_t bar, int count) {
  asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(bar), "r"(count) : "memory");
}
__device__ __forceinline__ void mbar_expect_arrive(uint32_t bar, uint32_t tx) {
  asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(bar), "r"(tx) : "memory");
}
__device__ __forceinline__ void mbar_arrive(uint32_t bar) {
  asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(bar) : "memory");
}
__device__ __forceinline__ void mbar_wait(uint32_t bar, uint32_t parity) {
  asm volatile(
      "{\n.reg .pred P1;\nLAB_WAIT:\n"
      "mbarrier.try_wait.parity.shared::cta.b64 P1, [%0], %1;\n"
      "@P1 bra DONE;\nbra LAB_WAIT;\nDONE:\n}\n" ::"r"(bar),
      "r"(parity)
      : "memory");
}
__device__ __forceinline__ void bulk_g2s(uint32_t dst, const void* src, uint32_t bytes, uint32_t bar) {
  asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];\n" ::"r"(dst),
               "l"(src), "r"(bytes), "r"(bar)
               : "memory");
}
__device__ __forceinline__ void bulk_s2g(void* dst, uint32_t src, uint32_t bytes) {
  asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;\n" ::"l"(dst), "r"(src), "r"(bytes)
               : "memory");
  asm volatile("cp.async.bulk.commit_group;\n" ::: "memory");
}
template <int N>
__device__ __forceinline__ void bulk_wait_read() { asm volatile("cp.async.bulk.wait_group.read %0;\n" ::"n"(N) : "memory"); }
template <int N>
__device__ __forceinline__ void bulk_wait() { asm volatile("cp.async.bulk.wait_group %0;\n" ::"n"(N) : "memory"); }

__device__ __forceinline__ uint64_t desc_sw128(uint32_t addr) {
  return static_cast<uint64_t>((addr & 0x3FFFF) >> 4) | (1ull << 16) | (64ull << 32) | (1ull << 62);
}
// byte offset of (row, byte col) in a K-major SW128 tile of 128-byte rows
__device__ __host__ __forceinline__ int sw128_b(int row, int col) {
  return row * 128 + ((((col >> 4) ^ (row & 7))) << 4) + (col & 15);
}
template <int N>
__device__ __forceinline__ void fence_acc(float (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+f"(d[i])::"memory");
}
template <int N>
__device__ __forceinline__ void fence_acc(int (&d)[N]) {
#pragma unroll
  for (int i = 0; i < N; ++i) asm volatile("" : "+r"(d[i])::"memory");
}
__device__ __forceinline__ void wg_arrive() { asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory"); }
__device__ __forceinline__ void wg_commit() { asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void wg_wait() { asm volatile("wgmma.wait_group.sync.aligned %0;\n" ::"n"(N) : "memory"); }

// d [64 x 48] (s32) += A (registers, u8 64 x 32) @ B (shared memory, s8 K-major [48][32 of 128])
__device__ __forceinline__ void mma_i8(const uint32_t (&a)[4], uint64_t desc, int (&d)[24]) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k32.s32.u8.s8 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p;\n}\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]), "+r"(d[4]), "+r"(d[5]), "+r"(d[6]), "+r"(d[7]), "+r"(d[8]),
        "+r"(d[9]), "+r"(d[10]), "+r"(d[11]), "+r"(d[12]), "+r"(d[13]), "+r"(d[14]), "+r"(d[15]), "+r"(d[16]),
        "+r"(d[17]), "+r"(d[18]), "+r"(d[19]), "+r"(d[20]), "+r"(d[21]), "+r"(d[22]), "+r"(d[23])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
}
// as mma_i8 with the accumulator overwritten (scale-d 0)
__device__ __forceinline__ void mma_i8z(const uint32_t (&a)[4], uint64_t desc, int (&d)[24]) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %29, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n48k32.s32.u8.s8 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, "
      "%14, %15, %16, %17, %18, %19, %20, %21, %22, %23}, {%24, %25, %26, %27}, %28, p;\n}\n"
      : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3]), "=r"(d[4]), "=r"(d[5]), "=r"(d[6]), "=r"(d[7]), "=r"(d[8]),
        "=r"(d[9]), "=r"(d[10]), "=r"(d[11]), "=r"(d[12]), "=r"(d[13]), "=r"(d[14]), "=r"(d[15]), "=r"(d[16]),
        "=r"(d[17]), "=r"(d[18]), "=r"(d[19]), "=r"(d[20]), "=r"(d[21]), "=r"(d[22]), "=r"(d[23])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(0));
}
// d [64 x 24] (f32) += A (registers, bf16 64 x 16) @ B (shared memory, bf16 K-major [24][16 of 64])
__device__ __forceinline__ void mma_bf16(const uint32_t (&a)[4], uint64_t desc, float (&d)[12]) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.bf16.bf16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
}

__device__ __forceinline__ float warp_max8(float v) {  // over the 8 lanes of a column (lane bits 2..4)
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 4));
  v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 8));
  return fmaxf(v, __shfl_xor_sync(0xffffffffu, v, 16));
}
__device__ __forceinline__ int warp_sum8i(int v) {
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  return v + __shfl_xor_sync(0xffffffffu, v, 16);
}

__device__ __forceinline__ void cp16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N) : "memory"); }
__device__ __forceinline__ void ldsm4(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void ldsm4t(uint32_t a, uint32_t (&r)[4]) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(a));
}
__device__ __forceinline__ void mma_bf16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_f16_16816(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
               "{%0, %1, %2, %3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t dsmem_map(uint32_t a, int rank) {
  uint32_t r;
  asm volatile("mapa.shared::cluster.u32 %0, %1, %2;\n" : "=r"(r) : "r"(a), "r"(rank));
  return r;
}
__device__ __forceinline__ uint32_t ld_dsmem(uint32_t a) {
  uint32_t v;
  asm volatile("ld.shared::cluster.b32 %0, [%1];\n" : "=r"(v) : "r"(a) : "memory");
  return v;
}
__device__ __forceinline__ void st_dsmem(uint32_t a, uint32_t v) {
  asm volatile("st.shared::cluster.b32 [%0], %1;\n" ::"r"(a), "r"(v) : "memory");
}
__device__ __forceinline__ void cluster_sync() {
  asm volatile("barrier.cluster.arrive.release.aligned;\nbarrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}
__device__ __forceinline__ int cluster_rank() {
  uint32_t r;
  asm volatile("mov.u32 %0, %%cluster_ctarank;\n" : "=r"(r));
  return (int)r;
}
template <int N>
__device__ __forceinline__ void cp_wait_le(int n) {  // cp.async.wait_group n for a run-time n <= N
  if constexpr (N > 0) {
    if (n >= N) {
      cp_wait<N>();
      return;
    }
    cp_wait_le<N - 1>(n);
  } else {
    cp_wait<0>();
  }
}

__device__ __forceinline__ void cp16z(uint32_t dst, const void* src, bool v) {  // zero-fills the 16 B when !v
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(v ? 16 : 0) : "memory");
}

// plan [4][B + 1] int32: cumulative tiles | cumulative units | nc | req pool index
struct Plan {
  const int *cu_t, *cu_u, *nc, *rpi;
  __device__ Plan(const int* plan, int B)
      : cu_t(plan), cu_u(plan + (B + 1)), nc(plan + 2 * (B + 1)), rpi(plan + 3 * (B + 1)) {}
};
// code CTAs of the attn grid (pki8_plan: G - B when the code part fits one wave beside the B dense CTAs, G - 3 B when
// it fits one unit a CTA beside 3 dense CTAs a row, else G) | dense mode: 0 = one CTA a row (code CTAs G), 1 = one CTA
// a row (G - B), 3 = one CTA a row and mma head group (G - 3 B)
__device__ __forceinline__ int plan_ncta(const int* plan, int B) { return __ldg(plan + 4 * (B + 1)); }
__device__ __forceinline__ int plan_dmode(const int* plan, int B) { return __ldg(plan + 4 * (B + 1) + 1); }
__device__ __forceinline__ int unit_row(const int* cu_u, int B, int u) {  // the largest b with cu_u[b] <= u
  int lo = 0, hi = B;
  while (lo < hi) {
    const int mid = (lo + hi + 1) >> 1;
    if (__ldg(cu_u + mid) <= u)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

// q record of a row (QREC bytes): 10 SW128 chunks [40][128] s8 (rows 8 k + 2 q + t: head 4 k + q, term t = hi / lo;
// a wgmma reads 48 rows, the last 8 are the next chunk and land in unused columns) | q_rope [24][64] bf16 SW128 (row
// 8 (k / 2) + 2 q + k % 2: head 4 k + q) | sum X [24] int32 | 1 / K [24] float.
// Code k of the bits sits at chunk k / 128 position 32 r + 16 t + 4 q + b for k = 32 (4 j + q) + 8 b + 2 r + t.
constexpr int QI_B = 40 * 128;
constexpr int OFF_QR_REC = NQCH * QI_B;
constexpr int OFF_PRM_REC = OFF_QR_REC + QR_B;
constexpr int QREC_B = OFF_PRM_REC + 2 * HN * 4;
constexpr int PUN_B = 40 * UE;  // P of a unit as stored: [40][128] u8 SW128
static_assert(QREC_B % 16 == 0, "q record");
static_assert(UE == 128 && SINK % 4 == 0, "one page per producer lane");
struct AParams {
  const uint8_t* qrec;
  const uint8_t* arena;
  const int* r2t;
  int64_t r2t_ld;
  const int* plan;
  int B;
  float* yp;   // [grid + B][H][BITS]: segment (CTA i, row b) at i + b: sum_t p_t bit_t, p_t = 2^(s_t - R)
  float* mlp;  // [grid + B][2][HN]: R (log2, an integer) | l = sum_t p_t
  float scale_log2;
  // dense part (sink + tail) of rows b < ND, CTA b each (the code CTAs follow): o_d fp32 [B][H][512], lse_d log2 [B][H]
  const __nv_bfloat16 *qn, *qr;  // [B][H][512] | [B][H][64]
  int64_t qn_sb, qn_sh, qr_sb, qr_sh;
  const int* dpt;   // [B][dw] dense rows of the row
  const int* dlen;  // [B]
  int dw;
  const __nv_bfloat16* dense;  // [rows][576]: latent | rope
  float* od;
  float* lsed;
  int ND;
};

constexpr int PH_B = HN * 128;  // P chunk: [24 heads][64 entries] fp16 SW128
constexpr int RINV = -(1 << 30);

struct ACfg {
  static constexpr int NS = 4;
  static constexpr int ST_B = UE * SLOT_B;
  static constexpr int OFF_Q = 0;
  static constexpr int OFF_ST = (QREC_B + 1023) / 1024 * 1024;
  static constexpr int OFF_P = OFF_ST + NS * ST_B;       // [2 parity][2 chunks][PH_B]
  static constexpr int OFF_RED = OFF_P + 4 * PH_B;       // max [HN] (ordered int) | l [8][HN] f32
  static constexpr int OFF_META = OFF_RED + 9 * HN * 4;  // max [HN] | l [8 warps][HN]
  static constexpr int OFF_BAR = OFF_META + NS * 16;  // full [NS] | empty [NS] | full_q
  static constexpr int OFF_FL = (OFF_BAR + (2 * NS + 1) * 8 + 127) / 128 * 128;  // flush stage [8 warps][2][640 B]
  static constexpr int SMEM = OFF_FL + 8 * 1280 + 1024;
  static_assert(SMEM <= 227 * 1024, "attn smem");
  static_assert(OFF_P % 1024 == 0 && PH_B % 1024 == 0, "P image alignment");
};

// d [64 x 24] (f32) += A (registers, f16 64 x 16) @ B (shared memory, f16 K-major [24][16 of 64])
__device__ __forceinline__ void mma_f16(const uint32_t (&a)[4], uint64_t desc, float (&d)[12]) {
  asm volatile(
      "{\n.reg .pred p;\nsetp.ne.b32 p, %17, 0;\n"
      "wgmma.mma_async.sync.aligned.m64n24k16.f32.f16.f16 "
      "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11}, {%12, %13, %14, %15}, %16, p, 1, 1, 0;\n}\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]),
        "+f"(d[9]), "+f"(d[10]), "+f"(d[11])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(1));
}
__device__ __forceinline__ int f2o(float f) {
  const int i = __float_as_int(f);
  return i ^ ((i >> 31) & 0x7fffffff);
}
__device__ __forceinline__ float o2f(int i) { return __int_as_float(i ^ ((i >> 31) & 0x7fffffff)); }
__device__ __forceinline__ float pow2i(int d) { return d < -126 ? 0.f : __int_as_float((d + 127) << 23); }  // d <= 0
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;\n" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ void sts16(uint32_t a, uint32_t v) {
  asm volatile("st.shared.b16 [%0], %1;\n" ::"r"(a), "h"((unsigned short)v) : "memory");
}
__device__ __forceinline__ bool bar_or(int id, int n, bool v) {  // named barrier returning the OR of v over n threads
  uint32_t r;
  asm volatile(
      "{\n.reg .pred p, q;\nsetp.ne.u32 q, %1, 0;\nbar.red.or.pred p, %2, %3, q;\nselp.u32 %0, 1, 0, p;\n}\n"
      : "=r"(r)
      : "r"((uint32_t)v), "r"(id), "r"(n)
      : "memory");
  return r != 0;
}
__device__ __forceinline__ float warp_sum8f(float v) {
  v += __shfl_xor_sync(0xffffffffu, v, 4);
  v += __shfl_xor_sync(0xffffffffu, v, 8);
  return v + __shfl_xor_sync(0xffffffffu, v, 16);
}
template <int R_>
__device__ __forceinline__ void reg_alloc() { asm volatile("setmaxnreg.inc.sync.aligned.u32 %0;\n" ::"n"(R_)); }
template <int R_>
__device__ __forceinline__ void reg_dealloc() { asm volatile("setmaxnreg.dec.sync.aligned.u32 %0;\n" ::"n"(R_)); }
__host__ __device__ constexpr int pv_pos(int t, int r8) { return (t < 5 ? 2 * t : 2 * t - 8) + r8; }

__device__ __forceinline__ void sts128(uint32_t a, float4 v) {
  asm volatile("st.shared.v4.f32 [%0], {%1, %2, %3, %4};\n" ::"r"(a), "f"(v.x), "f"(v.y), "f"(v.z), "f"(v.w) : "memory");
}
__device__ __forceinline__ float4 lds128(uint32_t a) {
  float4 v;
  asm volatile("ld.shared.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
  return v;
}
// partial of segment seg (all 256 consumer threads; the pv wgmmas of the caller's warpgroup have completed)
__device__ __forceinline__ void a_flush(const AParams& prm, int seg, const float (&Y)[10][12], const int (&R)[5],
                                        const float (&lsg)[5], float* red_l, int bit0, int g, int q, bool wml,
                                        uint32_t fl) {
  const int ws = 4 * (bit0 / 640) + (bit0 % 640) / 160;  // consumer warp: its slot of l (summed in order)
  float* yo = prm.yp + (int64_t)seg * H * BITS + 160 * ws;  // the warp's 160 bits, written through its stage
  const int lane = threadIdx.x % 32;
  const uint32_t fs = fl + ws * 1280;
#pragma unroll
  for (int kk = 0; kk < 5; ++kk) {
    const int c = kk >> 1, j = kk & 1;
#pragma unroll
    for (int hp = 0; hp < 2; ++hp) {
#pragma unroll
      for (int i = 0; i < 5; ++i) {
        float4 v;
        v.x = Y[2 * i][4 * c + j] * (float)(1 << (24 - pv_pos(2 * i, 0)));
        v.y = Y[2 * i][4 * c + 2 + j] * (float)(1 << (24 - pv_pos(2 * i, 1)));
        v.z = Y[2 * i + 1][4 * c + j] * (float)(1 << (24 - pv_pos(2 * i + 1, 0)));
        v.w = Y[2 * i + 1][4 * c + 2 + j] * (float)(1 << (24 - pv_pos(2 * i + 1, 1)));
        if ((q >> 1) == hp) sts128(fs + (q & 1) * 640 + 80 * g + 16 * i, v);
      }
      __syncwarp();
#pragma unroll
      for (int k = 0; k < 3; ++k) {
        const int cc = lane + 32 * k, r = cc / 40;
        if (cc < 80)
          *reinterpret_cast<float4*>(yo + (4 * kk + 2 * hp + r) * BITS + 4 * (cc % 40)) = lds128(fs + 16 * cc);
      }
      __syncwarp();
    }
    const float ls = warp_sum8f(lsg[kk]);
    if (g == 0) red_l[ws * HN + 4 * kk + q] = ls;
  }
  named_bar(1, 256);
  if (wml) {
    float* ml = prm.mlp + (int64_t)seg * 2 * HN;
#pragma unroll
    for (int kk = 0; kk < 5; ++kk) {
      ml[4 * kk + q] = (float)R[kk];
      float l = 0.f;
#pragma unroll
      for (int w = 0; w < 8; ++w) l += red_l[w * HN + 4 * kk + q];
      ml[HN + 4 * kk + q] = l;
    }
  }
}

#ifdef AT_TRACE
__device__ unsigned long long at_tr[64];
__device__ __forceinline__ unsigned long long gtime() {
  unsigned long long t;
  asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
  return t;
}
#define TR(i) if (blockIdx.x == AT_TRACE && tid == 0) at_tr[i] = gtime();
#else
#define TR(i)
#endif
#ifdef AT_PROF  // per-phase clock64 sums of thread 0 of CTA AT_PROF over all units: wait | score | epilogue | pv | redo
__device__ long long at_pf[8];
#define PF(i)                                     \
  {                                               \
    const long long c_ = clock64();               \
    if (i > 0) pf_a[i - 1] += c_ - pf_t;          \
    pf_t = c_;                                    \
  }
#else
#define PF(i)
#endif
// ---- dense role: exact attention over the dense rows of row b (sink + tail, at most MAXDW), one CTA of 384 threads.
// Tiles of 64 entries, double-buffered (cp.async, rows >= dlen zero-filled); S [64 entries][24 heads] = K q^T (bf16
// mma.sync, warp w: rows 16 (w & 3) .., dims 192 (w >> 2) .., partials summed in a fixed order), online softmax in
// log2 units (warp w < 10: heads 2 w, 2 w + 1; l over the fp32 P), O^T [512][24] += V^T (P_hi + P_lo)^T with
// P_hi = bf16(P), P_lo = bf16(P - P_hi) (warp w < 8: dims 64 w ..; a single bf16 P adds ~7% rel err on all-dense
// rows, ~3% above fa3). Output o_d = O / l (fp32: the merge rounds once) and lse_d = m + log2 l.
struct DCfg {
  static constexpr int KR = 576, LD = KR + 8, RB = 2 * LD;  // smem rows of 1168 B: conflict-free ldmatrix
  static constexpr int LDP = TILE + 8, LDR = TILE + 4, LDO = 512 + 4, MAXDW = 384;
  static constexpr int OFF_Q = 0;                         // q [24][LD] bf16 (heads 20..23 zero)
  static constexpr int OFF_K = OFF_Q + HN * RB;           // 2 x [64][LD] bf16; o staging [24][LDO] fp32 at the end
  static constexpr int OFF_P = OFF_K + 2 * TILE * RB;     // P_hi | P_lo [24][LDP] bf16
  static constexpr int OFF_R = OFF_P + 2 * HN * LDP * 2;  // score partials [3][24][LDR] f32
  static constexpr int OFF_A = OFF_R + 3 * HN * LDR * 4;  // alpha | m | l [24] f32
  static constexpr int OFF_I = OFF_A + 3 * HN * 4;        // dense rows [MAXDW] int (-1 past dlen)
  static constexpr int SMEM = OFF_I + MAXDW * 4;
  static_assert(SMEM + 1024 <= ACfg::SMEM, "dense role smem");
  static_assert(HN * LDO * 4 <= TILE * RB && OFF_K % 128 == 0 && OFF_P % 16 == 0, "dense role layout");
};
__device__ __forceinline__ float warp_max32(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
__device__ __forceinline__ float warp_sum32(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
struct DArgs {  // the dense role's fields of AParams
  const __nv_bfloat16 *qn, *qr;
  int64_t qn_sb, qn_sh, qr_sb, qr_sh;
  const int* dpt;
  const int* dlen;
  int dw;
  const __nv_bfloat16* dense;
  float* od;
  float* lsed;
  float scale_log2;
  const int* plan;
  int nd;
};
// No dependency wait: q, the dense page table and the pool are written before qproj, which triggers this grid only
// after its own wait; merge reads o_d / lse_d after combine, which waits for this grid. Out of line (fields by value,
// smem from the extern array) so that its register allocation does not follow the code-CTA path's.
// NG = 3: all heads; NG = 1: mma head group hg (local heads 0..7 = heads 8 hg .., zero past H) with the full K tiles.
// Every S / O element keeps its mma chain (columns of other heads never enter it): bitwise the NG = 3 result.
// NG = 1 is compiled only with PKI8_DSPLIT (SGLANG_PICOKV_MLA_DSPLIT=1): any NG = 1 code in the call graph changes the
// code-CTA path's register allocation (+1.2 .. 2.6 % at 8 x 128k, 32 x 32k), so the default build has NG = 3 only.
#ifndef PKI8_DSPLIT
#define PKI8_DSPLIT 0
#endif
template <int NG>
__device__ __forceinline__ void dense_body(const DArgs prm, int b, int hg) {
  using D = DCfg;
  constexpr int NSW = NG == 3 ? H / 2 : 8, KH = NG == 3 ? 2 : 1;  // softmax warps, heads a softmax warp
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const uint32_t s0 = smem_u32(smem), sq = s0 + D::OFF_Q, sk = s0 + D::OFF_K, sp = s0 + D::OFF_P;
  float* red = reinterpret_cast<float*>(smem + D::OFF_R);
  float* alf = reinterpret_cast<float*>(smem + D::OFF_A);
  int* sidx = reinterpret_cast<int*>(smem + D::OFF_I);
  __nv_bfloat16* pp = reinterpret_cast<__nv_bfloat16*>(smem + D::OFF_P);
  griddep_launch();
  const int dl = __ldg(prm.dlen + b), nt = (dl + TILE - 1) / TILE;
  // pipelined steps unless code CTAs wait for the SMs of the dense CTAs (code part over one wave)
  const bool pipe = NG == 3 && plan_dmode(prm.plan, prm.nd) == 1;
  for (int j = tid; j < nt * TILE; j += 384) sidx[j] = j < dl ? __ldg(prm.dpt + (int64_t)b * prm.dw + j) : -1;
  if constexpr (NG == 3) {
    for (int i = tid; i < H * 72; i += 384) {
      const int h = i / 72, c = i % 72;
      cp16(sq + h * D::RB + 16 * c, c < 64 ? prm.qn + b * prm.qn_sb + h * prm.qn_sh + 8 * c
                                           : prm.qr + b * prm.qr_sb + h * prm.qr_sh + 8 * (c - 64));
    }
    for (int i = tid; i < (HN - H) * 72; i += 384) sts128z(sq + (H + i / 72) * D::RB + 16 * (i % 72));
  } else {
    for (int i = tid; i < 8 * 72; i += 384) {
      const int hl = i / 72, h = 8 * hg + hl, c = i % 72;
      if (h < H)
        cp16(sq + hl * D::RB + 16 * c, c < 64 ? prm.qn + b * prm.qn_sb + h * prm.qn_sh + 8 * c
                                              : prm.qr + b * prm.qr_sb + h * prm.qr_sh + 8 * (c - 64));
      else
        sts128z(sq + hl * D::RB + 16 * c);
    }
  }
  for (int i = tid; i < 2 * HN * D::LDP / 8; i += 384) sts128z(sp + 16 * i);
  if (tid < 3 * HN) alf[tid] = 0.f;
  __syncthreads();
  auto ld_tile = [&](int tt) {
    static_assert(TILE == 64 && 2 * D::KR == 9 * 128, "dense tile = 64 rows x 9 lines of 128 B");
    const int ln = tid / 8, rr = ln / 3, lc = 128 * (ln % 3) + 16 * (tid % 8);  // row in block | byte in row
    const uint32_t dst = sk + (tt & 1) * TILE * D::RB + rr * D::RB + lc;
#pragma unroll
    for (int rb = 0; rb < 4; ++rb) {
      const int row = sidx[TILE * tt + 16 * rb + rr];
      const char* src = reinterpret_cast<const char*>(prm.dense) + (int64_t)max(row, 0) * (2 * D::KR) + lc;
#pragma unroll
      for (int lb = 0; lb < 3; ++lb) cp16z(dst + 16 * rb * D::RB + 384 * lb, src + 384 * lb, row >= 0);
    }
    cp_commit();
  };
  ld_tile(0);  // with q
  // tiles >= 1 from warps 10, 11 (64 threads, 8 per 128 B line) while warps < 10 run the softmax
  auto ld_tile2 = [&](int tt) {
    const int u = tid - 320, l8 = u % 8;
    const uint32_t dst0 = sk + (tt & 1) * TILE * D::RB + 16 * l8;
#pragma unroll 2
    for (int r = u / 8; r < TILE; r += 8) {
      const int row = sidx[TILE * tt + r];
      const char* src = reinterpret_cast<const char*>(prm.dense) + (int64_t)max(row, 0) * (2 * D::KR) + 16 * l8;
#pragma unroll
      for (int lb = 0; lb < 9; ++lb) cp16z(dst0 + r * D::RB + 128 * lb, src + 128 * lb, row >= 0);
    }
    cp_commit();
  };
  float m_run[2] = {-INFINITY, -INFINITY}, l_run[2] = {0.f, 0.f};
  float o[4][NG][4];
#pragma unroll
  for (int mi = 0; mi < 4; ++mi)
#pragma unroll
    for (int nn = 0; nn < NG; ++nn)
#pragma unroll
      for (int i = 0; i < 4; ++i) o[mi][nn][i] = 0.f;
  if (pipe) {
    cp_wait<0>();
    __syncthreads();  // q and tile 0 visible
    // step tt: warps 8..11 S(tt) (warp 8 + mt: rows 16 mt .., all 3 dim groups, the same mma chains as 12 warps)
    // while warps < 8 run PV(tt - 1); sync; warps < 10 softmax(tt), warps 10, 11 load tile tt + 1 (buffer of tile
    // tt - 1); sync
    for (int tt = 0; tt <= nt; ++tt) {
      if (warp >= 8) {
        if (tt < nt) {
          if (tt > 0) {
            cp_wait<0>();
            named_bar(2, 128);  // tile tt visible to warps 8..11
          }
          const uint32_t kb = sk + (tt & 1) * TILE * D::RB;
          const int mt = warp - 8;
          float s[3][NG][4];
#pragma unroll
          for (int kg = 0; kg < 3; ++kg)
#pragma unroll
            for (int n3 = 0; n3 < NG; ++n3)
#pragma unroll
              for (int i = 0; i < 4; ++i) s[kg][n3][i] = 0.f;
          const uint32_t a_ad = kb + (16 * mt + (lane & 7) + ((lane >> 3) & 1) * 8) * D::RB + (lane >> 4) * 16;
          const uint32_t b_ad = sq + (lane & 7) * D::RB + (lane >> 3) * 16;
#pragma unroll
          for (int kp = 0; kp < 6; ++kp)
#pragma unroll
            for (int kg = 0; kg < 3; ++kg) {
              const int k0 = 192 * kg + 32 * kp;
              uint32_t a0[4], a1[4];
              ldsm4(a_ad + 2 * k0, a0);
              ldsm4(a_ad + 2 * (k0 + 16), a1);
#pragma unroll
              for (int n3 = 0; n3 < NG; ++n3) {
                uint32_t bq[4];
                ldsm4(b_ad + 8 * n3 * D::RB + 2 * k0, bq);
                mma_bf16_16816(s[kg][n3], a0, bq[0], bq[1]);
                mma_bf16_16816(s[kg][n3], a1, bq[2], bq[3]);
              }
            }
#pragma unroll
          for (int kg = 0; kg < 3; ++kg) {
            float* rp = red + kg * HN * D::LDR + 16 * mt + g;
#pragma unroll
            for (int n3 = 0; n3 < NG; ++n3)
#pragma unroll
              for (int i = 0; i < 4; ++i) rp[(8 * n3 + 2 * t + (i & 1)) * D::LDR + 8 * (i >> 1)] = s[kg][n3][i];
          }
        }
      } else if (tt > 0) {
        const uint32_t kb = sk + ((tt - 1) & 1) * TILE * D::RB;
        float al[NG][2];
#pragma unroll
        for (int nn = 0; nn < NG; ++nn) {
          al[nn][0] = alf[8 * nn + 2 * t];
          al[nn][1] = alf[8 * nn + 2 * t + 1];
        }
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
          for (int nn = 0; nn < NG; ++nn)
#pragma unroll
            for (int i = 0; i < 4; ++i) o[mi][nn][i] *= al[nn][i & 1];
#pragma unroll
        for (int kp = 0; kp < 2; ++kp) {
          uint32_t bp[NG][4], bl[NG][4];
#pragma unroll
          for (int nn = 0; nn < NG; ++nn) {
            ldsm4(sp + ((8 * nn + (lane & 7)) * D::LDP + 32 * kp + (lane >> 3) * 8) * 2, bp[nn]);
            ldsm4(sp + ((HN + 8 * nn + (lane & 7)) * D::LDP + 32 * kp + (lane >> 3) * 8) * 2, bl[nn]);
          }
#pragma unroll
          for (int kh = 0; kh < 2; ++kh) {
            const int k0 = 32 * kp + 16 * kh;
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
              uint32_t a[4];
              ldsm4t(kb + (k0 + (lane & 7) + ((lane >> 4) & 1) * 8) * D::RB +
                         (64 * warp + 16 * mi + ((lane >> 3) & 1) * 8) * 2, a);
#pragma unroll
              for (int nn = 0; nn < NG; ++nn) {
                mma_bf16_16816(o[mi][nn], a, bp[nn][2 * kh], bp[nn][2 * kh + 1]);
                mma_bf16_16816(o[mi][nn], a, bl[nn][2 * kh], bl[nn][2 * kh + 1]);
              }
            }
          }
        }
      }
      if (tt == nt) break;
      __syncthreads();  // S(tt) in red; PV(tt - 1) done: P, alpha and the buffer of tile tt + 1 free
      if (warp < NSW) {
        const int e = TILE * tt + lane;
#pragma unroll
        for (int k = 0; k < KH; ++k) {
          const int h = NG == 3 ? 2 * warp + k : warp;
          if (NG == 1 && 8 * hg + h >= H) break;
          const float* r0 = red + h * D::LDR + lane;
          constexpr int KS = HN * D::LDR;
          float sa = (r0[0] + r0[KS]) + r0[2 * KS], sb = (r0[32] + r0[KS + 32]) + r0[2 * KS + 32];
          sa = e < dl ? sa * prm.scale_log2 : -INFINITY;
          sb = e + 32 < dl ? sb * prm.scale_log2 : -INFINITY;
          const float mn = fmaxf(m_run[k], warp_max32(fmaxf(sa, sb)));
          const float fa = exp2f(sa - mn), fb = exp2f(sb - mn);
          const __nv_bfloat16 pa = __float2bfloat16_rn(fa), pb = __float2bfloat16_rn(fb);
          pp[h * D::LDP + lane] = pa;
          pp[h * D::LDP + lane + 32] = pb;
          pp[(HN + h) * D::LDP + lane] = __float2bfloat16_rn(fa - __bfloat162float(pa));
          pp[(HN + h) * D::LDP + lane + 32] = __float2bfloat16_rn(fb - __bfloat162float(pb));
          const float ls = warp_sum32(fa + fb), al = exp2f(m_run[k] - mn);
          l_run[k] = l_run[k] * al + ls;
          m_run[k] = mn;
          if (lane == 0) alf[h] = al;
        }
      } else if ((NG == 3 || warp >= 10) && tt + 1 < nt) {
        ld_tile2(tt + 1);
      }
      __syncthreads();  // P(tt), alpha(tt) visible
    }
  } else {  // tile tt + 1 issued at the top of step tt
    const int mt = warp & 3, kg = warp >> 2;
    for (int tt = 0; tt < nt; ++tt) {
      cp_wait<0>();
      __syncthreads();  // tile tt visible; tile tt - 1, P and alpha no longer read
      if (tt + 1 < nt) ld_tile(tt + 1);
      const uint32_t kb = sk + (tt & 1) * TILE * D::RB;
      {
        float s[NG][4];
#pragma unroll
        for (int n3 = 0; n3 < NG; ++n3)
#pragma unroll
          for (int i = 0; i < 4; ++i) s[n3][i] = 0.f;
        const uint32_t a_ad = kb + (16 * mt + (lane & 7) + ((lane >> 3) & 1) * 8) * D::RB + (lane >> 4) * 16;
        const uint32_t b_ad = sq + (lane & 7) * D::RB + (lane >> 3) * 16;
#pragma unroll
        for (int kp = 0; kp < 6; ++kp) {
          const int k0 = 192 * kg + 32 * kp;
          uint32_t a0[4], a1[4];
          ldsm4(a_ad + 2 * k0, a0);
          ldsm4(a_ad + 2 * (k0 + 16), a1);
#pragma unroll
          for (int n3 = 0; n3 < NG; ++n3) {
            uint32_t bq[4];
            ldsm4(b_ad + 8 * n3 * D::RB + 2 * k0, bq);
            mma_bf16_16816(s[n3], a0, bq[0], bq[1]);
            mma_bf16_16816(s[n3], a1, bq[2], bq[3]);
          }
        }
        float* rp = red + kg * HN * D::LDR + 16 * mt + g;
#pragma unroll
        for (int n3 = 0; n3 < NG; ++n3)
#pragma unroll
          for (int i = 0; i < 4; ++i) rp[(8 * n3 + 2 * t + (i & 1)) * D::LDR + 8 * (i >> 1)] = s[n3][i];
      }
      __syncthreads();
      if (warp < NSW) {
        const int e = TILE * tt + lane;
#pragma unroll
        for (int k = 0; k < KH; ++k) {
          const int h = NG == 3 ? 2 * warp + k : warp;
          if (NG == 1 && 8 * hg + h >= H) break;
          const float* r0 = red + h * D::LDR + lane;
          constexpr int KS = HN * D::LDR;
          float sa = (r0[0] + r0[KS]) + r0[2 * KS], sb = (r0[32] + r0[KS + 32]) + r0[2 * KS + 32];
          sa = e < dl ? sa * prm.scale_log2 : -INFINITY;
          sb = e + 32 < dl ? sb * prm.scale_log2 : -INFINITY;
          const float mn = fmaxf(m_run[k], warp_max32(fmaxf(sa, sb)));
          const float fa = exp2f(sa - mn), fb = exp2f(sb - mn);
          const __nv_bfloat16 pa = __float2bfloat16_rn(fa), pb = __float2bfloat16_rn(fb);
          pp[h * D::LDP + lane] = pa;
          pp[h * D::LDP + lane + 32] = pb;
          pp[(HN + h) * D::LDP + lane] = __float2bfloat16_rn(fa - __bfloat162float(pa));
          pp[(HN + h) * D::LDP + lane + 32] = __float2bfloat16_rn(fb - __bfloat162float(pb));
          const float ls = warp_sum32(fa + fb), al = exp2f(m_run[k] - mn);
          l_run[k] = l_run[k] * al + ls;
          m_run[k] = mn;
          if (lane == 0) alf[h] = al;
        }
      }
      __syncthreads();
      if (warp < 8) {
        float al[NG][2];
#pragma unroll
        for (int nn = 0; nn < NG; ++nn) {
          al[nn][0] = alf[8 * nn + 2 * t];
          al[nn][1] = alf[8 * nn + 2 * t + 1];
        }
#pragma unroll
        for (int mi = 0; mi < 4; ++mi)
#pragma unroll
          for (int nn = 0; nn < NG; ++nn)
#pragma unroll
            for (int i = 0; i < 4; ++i) o[mi][nn][i] *= al[nn][i & 1];
#pragma unroll
        for (int kp = 0; kp < 2; ++kp) {
          uint32_t bp[NG][4], bl[NG][4];
#pragma unroll
          for (int nn = 0; nn < NG; ++nn) {
            ldsm4(sp + ((8 * nn + (lane & 7)) * D::LDP + 32 * kp + (lane >> 3) * 8) * 2, bp[nn]);
            ldsm4(sp + ((HN + 8 * nn + (lane & 7)) * D::LDP + 32 * kp + (lane >> 3) * 8) * 2, bl[nn]);
          }
#pragma unroll
          for (int kh = 0; kh < 2; ++kh) {
            const int k0 = 32 * kp + 16 * kh;
#pragma unroll
            for (int mi = 0; mi < 4; ++mi) {
              uint32_t a[4];
              ldsm4t(kb + (k0 + (lane & 7) + ((lane >> 4) & 1) * 8) * D::RB +
                         (64 * warp + 16 * mi + ((lane >> 3) & 1) * 8) * 2, a);
#pragma unroll
              for (int nn = 0; nn < NG; ++nn) {
                mma_bf16_16816(o[mi][nn], a, bp[nn][2 * kh], bp[nn][2 * kh + 1]);
                mma_bf16_16816(o[mi][nn], a, bl[nn][2 * kh], bl[nn][2 * kh + 1]);
              }
            }
          }
        }
      }
    }
  }
  // o_d = O / l straight from the accumulators (heads < H), lse_d = m + log2 l
  if (warp < NSW && lane == 0)
#pragma unroll
    for (int k = 0; k < KH; ++k) {
      const int h = NG == 3 ? 2 * warp + k : warp;
      alf[HN + h] = m_run[k];
      alf[2 * HN + h] = l_run[k];
    }
  __syncthreads();
  if (warp < 8) {
#pragma unroll
    for (int nn = 0; nn < NG; ++nn) {
      const float il0 = 1.f / alf[2 * HN + 8 * nn + 2 * t], il1 = 1.f / alf[2 * HN + 8 * nn + 2 * t + 1];
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const int h = 8 * nn + 2 * t + (i & 1), gh = NG == 3 ? h : 8 * hg + h;
        if (gh < H) {
          float* od = prm.od + ((int64_t)b * H + gh) * 512 + 64 * warp + g + 8 * (i >> 1);
#pragma unroll
          for (int mi = 0; mi < 4; ++mi) od[16 * mi] = o[mi][nn][i] * ((i & 1) ? il1 : il0);
        }
      }
    }
  }
  if constexpr (NG == 3) {
    if (tid < H) prm.lsed[b * H + tid] = alf[HN + tid] + log2f(alf[2 * HN + tid]);
  } else {
    if (tid < 8 && 8 * hg + tid < H) prm.lsed[b * H + 8 * hg + tid] = alf[HN + tid] + log2f(alf[2 * HN + tid]);
  }
}

// dense CTA j (3 nd of them when the split may apply, else nd): row j % nd, head group j / nd in dense mode 3, else
// row j (all heads; j >= nd: nothing)
__device__ __noinline__ void dense_role(const DArgs prm, int j) {
#if PKI8_DSPLIT
  if (plan_dmode(prm.plan, prm.nd) == 3) {
    dense_body<1>(prm, j % prm.nd, j / prm.nd);
    return;
  }
#endif
  if (j < prm.nd) dense_body<3>(prm, j, 0);
}

__global__ void __launch_bounds__(384, 1) pki8_attn_kernel(const __grid_constant__ AParams prm) {
  using C = ACfg;
  extern __shared__ uint8_t smem_raw[];
  uint8_t* smem = smem_raw + ((1024u - (smem_u32(smem_raw) & 1023u)) & 1023u);
  const int tid = threadIdx.x, wg = tid / 128, tw = tid % 128, warp = tw / 32, lane = tid % 32, g = lane / 4,
            q = lane % 4;
  TR(0);
  if ((int)blockIdx.x < prm.ND) {
    dense_role(DArgs{prm.qn, prm.qr, prm.qn_sb, prm.qn_sh, prm.qr_sb, prm.qr_sh, prm.dpt, prm.dlen, prm.dw, prm.dense,
                     prm.od, prm.lsed, prm.scale_log2, prm.plan, prm.B},
               blockIdx.x);
    return;
  }
  const int cta = (int)blockIdx.x - prm.ND, ncta = plan_ncta(prm.plan, prm.B);
  if (cta >= ncta) return;
  const bool prod = wg == 2 && warp == 0;
  griddep_launch();
  const Plan pl(prm.plan, prm.B);
  const int U = pl.cu_u[prm.B];
  const int ua = (int)((int64_t)cta * U / ncta), ue = (int)((int64_t)(cta + 1) * U / ncta);
  if (ua >= ue) {
    griddep_wait();
    return;
  }
  const int N = ue - ua;
  const uint32_t s0 = smem_u32(smem);
  const uint32_t sq = s0 + C::OFF_Q, sst = s0 + C::OFF_ST, sp = s0 + C::OFF_P, bar = s0 + C::OFF_BAR;
  auto full = [&](int s) { return bar + 8 * s; };
  auto empty = [&](int s) { return bar + 8 * (C::NS + s); };
  const uint32_t full_q = bar + 16 * C::NS;
  int4* meta = reinterpret_cast<int4*>(smem + C::OFF_META);
  int* red_mx = reinterpret_cast<int*>(smem + C::OFF_RED);
  float* red_l = reinterpret_cast<float*>(smem + C::OFF_RED + HN * 4);
  if (tid == 0) {
    for (int s = 0; s < C::NS; ++s) {
      mbar_init(full(s), 1);
      mbar_init(empty(s), 8);  // lane 0 of each consumer warp
    }
    mbar_init(full_q, 1);
    asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
  }
  if (tid < HN) {
    red_mx[tid] = INT_MIN;
    red_l[tid] = 0.f;
  }
  __syncthreads();
  if (!prod) griddep_wait();  // the producer waits after loading its first units

  if (wg == 2) {  // producer: warp 8 loads the units (one 1152 B page per lane) and each new row's q record
    reg_dealloc<40>();
    if (warp != 0) return;
    auto load_unit = [&](int n, int u, int b, int qi) {
      const int s = n % C::NS;
      const int e0 = UE * (u - pl.cu_u[b]), ne = min(UE, pl.nc[b] - e0);
      const int64_t rrow = (int64_t)pl.rpi[b] * prm.r2t_ld + SINK + e0;
      if (lane == 0) {
        meta[s] = make_int4(b, qi, ne, u);
        mbar_expect_arrive(full(s), UE * SLOT_B);
      }
      __syncwarp();
      const int e = 4 * lane;
      const int slot = e < ne ? __ldg(prm.r2t + rrow + e) : 0;
      bulk_g2s(sst + s * C::ST_B + e * SLOT_B, prm.arena + (int64_t)(slot & ~3) * SLOT_B, 4 * SLOT_B, full(s));
    };
    // the first stages of the first row before the dependency wait (the plan, req_to_token and the arena are
    // written before qproj; the q record is qproj's)
    int cur = 0, qi = 0, n0 = 0;
    for (int b0 = 0; b0 <= prm.B; b0 += 32) {  // cur = the largest b with cu_u[b] <= ua (cu_u non-decreasing)
      const int bb = b0 + lane;
      cur += __popc(__ballot_sync(0xffffffffu, bb <= prm.B && __ldg(pl.cu_u + bb) <= ua));
    }
    --cur;
    {
      const int cu0 = __ldg(pl.cu_u + cur), cu1 = __ldg(pl.cu_u + cur + 1), nc0 = __ldg(pl.nc + cur);
      const int64_t rr0 = (int64_t)__ldg(pl.rpi + cur) * prm.r2t_ld + SINK;
      const int nf = min(min(N, C::NS), cu1 - ua), e = 4 * lane;
      int sl[C::NS];
#pragma unroll
      for (int k = 0; k < C::NS; ++k) {
        const int e0 = UE * (ua + k - cu0);
        sl[k] = k < nf && e0 + e < nc0 ? __ldg(prm.r2t + rr0 + e0 + e) : 0;
      }
#pragma unroll
      for (int k = 0; k < C::NS; ++k)
        if (k < nf) {
          const int e0 = UE * (ua + k - cu0);
          if (lane == 0) {
            meta[k] = make_int4(cur, 0, min(UE, nc0 - e0), ua + k);
            mbar_expect_arrive(full(k), UE * SLOT_B);
          }
          __syncwarp();
          bulk_g2s(sst + k * C::ST_B + e * SLOT_B, prm.arena + (int64_t)(sl[k] & ~3) * SLOT_B, 4 * SLOT_B, full(k));
        }
      n0 = nf;
    }
    griddep_wait();
    if (lane == 0) {
      mbar_expect_arrive(full_q, QREC_B);
      bulk_g2s(sq, prm.qrec + (int64_t)cur * QREC_B, QREC_B, full_q);
    }
    for (int n = n0; n < N; ++n) {
      const int u = ua + n, b = unit_row(pl.cu_u, prm.B, u), s = n % C::NS;
      if (b != cur) {  // both warpgroups finished every earlier unit before the record is replaced
        if (n >= 1) mbar_wait(empty((n - 1) % C::NS), ((n - 1) / C::NS) & 1);
        if (lane == 0) {
          mbar_expect_arrive(full_q, QREC_B);
          bulk_g2s(sq, prm.qrec + (int64_t)b * QREC_B, QREC_B, full_q);
        }
        cur = b;
        ++qi;
      }
      if (n >= C::NS) mbar_wait(empty(s), ((n / C::NS) - 1) & 1);
      load_unit(n, u, b, qi);
    }
    return;
  }
  reg_alloc<232>();
  TR(1);
  const int G = 8 * warp + g, bit0 = 640 * wg + 20 * G, wofs = 4 * (bit0 >> 5), sh0 = bit0 & 31;
  const bool wml = wg == 0 && warp == 0 && g == 0;
  float Y[10][12];
  int R[5];
  float lsg[5];
#pragma unroll
  for (int t = 0; t < 10; ++t)
#pragma unroll
    for (int i = 0; i < 12; ++i) Y[t][i] = 0.f;
#pragma unroll
  for (int kk = 0; kk < 5; ++kk) {
    R[kk] = RINV;
    lsg[kk] = 0.f;
  }
  int cur = -1, myq = -1;
#ifdef AT_PROF
  long long pf_a[5] = {0, 0, 0, 0, 0}, pf_t = 0;
#endif
  for (int n = 0; n < N; ++n) {
    PF(0);
    const int s = n % C::NS, par = n & 1;
    mbar_wait(full(s), (n / C::NS) & 1);
    if (n == 0) TR(2);
    const int4 mt = meta[s];
    const int b = mt.x, ne = mt.z;
    if (b != cur) {
      if (cur >= 0) {
        wg_wait<0>();
#pragma unroll
        for (int t = 0; t < 10; ++t) fence_acc(Y[t]);
        a_flush(prm, cta + cur, Y, R, lsg, red_l, bit0, g, q, wml, s0 + C::OFF_FL);
#pragma unroll
        for (int t = 0; t < 10; ++t)
#pragma unroll
          for (int i = 0; i < 12; ++i) Y[t][i] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 5; ++kk) {
          R[kk] = RINV;
          lsg[kk] = 0.f;
        }
      }
      cur = b;
    }
    if (mt.y != myq) {
      mbar_wait(full_q, mt.y & 1);
      myq = mt.y;
    }
    if (n == 0) TR(3);
    PF(1);
    const uint32_t st = sst + s * C::ST_B;
    // ---- score of entries 64 wg + 16 warp + g + 8 r8
    int sxi[5];
    float ik[5];
#pragma unroll
    for (int kk = 0; kk < 5; ++kk) {
      sxi[kk] = (int)lds32(sq + OFF_PRM_REC + 4 * (4 * kk + q));
      ik[kk] = __uint_as_float(lds32(sq + OFF_PRM_REC + 4 * HN + 4 * (4 * kk + q)));
    }
    int acc[24];
    float accr[12];
#pragma unroll
    for (int i = 0; i < 24; ++i) acc[i] = 0;
#pragma unroll
    for (int i = 0; i < 12; ++i) accr[i] = 0.f;
    const uint32_t rb = st + (64 * wg + 16 * warp + g) * SLOT_B + 4 * q;
#ifndef AT_NO_SCORE
#pragma unroll
    for (int j = 0; j < NQCH; ++j) {
      const uint32_t w0 = lds32(rb + 16 * j), w8 = lds32(rb + 8 * SLOT_B + 16 * j);
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        uint32_t a[4];
        a[0] = (w0 >> (2 * r)) & M01;
        a[1] = (w8 >> (2 * r)) & M01;
        a[2] = (w0 >> (2 * r + 1)) & M01;
        a[3] = (w8 >> (2 * r + 1)) & M01;
        fence_acc(acc);
        wg_arrive();
        mma_i8(a, desc_sw128(sq + j * QI_B + 32 * r), acc);
        wg_commit();
        fence_acc(acc);
        wg_wait<1>();
      }
    }
#pragma unroll
    for (int sr = 0; sr < 4; ++sr) {
      uint32_t a[4];
      const uint32_t r0 = rb + 160 + 32 * sr;
      a[0] = lds32(r0);
      a[1] = lds32(r0 + 8 * SLOT_B);
      a[2] = lds32(r0 + 16);
      a[3] = lds32(r0 + 8 * SLOT_B + 16);
      fence_acc(accr);
      wg_arrive();
      mma_bf16(a, desc_sw128(sq + OFF_QR_REC + 32 * sr), accr);
      wg_commit();
      fence_acc(accr);
      wg_wait<1>();
    }
#endif
    // ---- pv operands: W0 / W1 [kk][jj] = halfwords of entries 16 kk + 2 q + 8 jj (+ 1): window bits 0..9 at
    // positions 0..9 / bits 10..19 at positions 2..11 (bits bit0 .. bit0 + 19 of the code)
    uint32_t W0[8][2], W1[8][2];
#pragma unroll
    for (int kk = 0; kk < 8; ++kk)
#pragma unroll
      for (int jj = 0; jj < 2; ++jj) {
        const uint32_t ad = st + (16 * kk + 2 * q + 8 * jj) * SLOT_B + wofs;
        const uint32_t va = __funnelshift_r(lds32(ad), lds32(ad + 4), sh0);
        const uint32_t vb = __funnelshift_r(lds32(ad + SLOT_B), lds32(ad + SLOT_B + 4), sh0);
        W0[kk][jj] = prmt(va, vb, 0x5410);
        W1[kk][jj] = prmt(va, vb, 0x6521);
      }
    wg_wait<0>();
    fence_acc(acc);
    fence_acc(accr);
    __syncwarp();
    if (lane == 0) mbar_arrive(empty(s));  // the stage and (by the score wgmmas) the q record are no longer read
    if (n == 0) TR(4);
    PF(2);
    // ---- scores, overflow flag, P image
    float sv[2][5], ph[2][5];
    bool ov = false;
#pragma unroll
    for (int r8 = 0; r8 < 2; ++r8)
#pragma unroll
      for (int kk = 0; kk < 5; ++kk) {
        const int e = 64 * wg + 16 * warp + g + 8 * r8;
        const int x = 2 * (256 * acc[4 * kk + 2 * r8] + acc[4 * kk + 2 * r8 + 1]) - sxi[kk];
        const float v = ((float)x * ik[kk] + accr[4 * (kk >> 1) + 2 * r8 + (kk & 1)]) * prm.scale_log2;
        sv[r8][kk] = e < ne ? v : -INFINITY;
        ov |= sv[r8][kk] > (float)(R[kk] + 15);
      }
#pragma unroll
    for (int kk = 0; kk < 5; ++kk) ov |= R[kk] == RINV;
    const uint32_t pb = sp + par * (2 * PH_B) + wg * PH_B;
    auto put_p = [&]() {
#pragma unroll
      for (int r8 = 0; r8 < 2; ++r8)
#pragma unroll
        for (int kk = 0; kk < 5; ++kk) {
          const int el = 16 * warp + g + 8 * r8, nr = 8 * (kk >> 1) + 2 * q + (kk & 1);
          const __half hp = __float2half_rn(exp2f(sv[r8][kk] - (float)R[kk]));
          ph[r8][kk] = __half2float(hp);
          sts16(pb + sw128_b(nr, 2 * el), __half_as_ushort(hp));
        }
      fence_async_smem();
    };
    put_p();
    if (bar_or(2, 256, ov)) {  // first unit of the segment or an overflow: R = max(R, ceil(unit max)), rescale
#ifdef AT_PROF
      pf_a[4] += 1;
#endif
#pragma unroll
      for (int kk = 0; kk < 5; ++kk) {
        const float mu = warp_max8(fmaxf(sv[0][kk], sv[1][kk]));
        if (g == 0) atomicMax(red_mx + 4 * kk + q, f2o(mu));
      }
      named_bar(1, 256);
#pragma unroll
      for (int kk = 0; kk < 5; ++kk) {
        const int rn = max(R[kk], (int)ceilf(o2f(red_mx[4 * kk + q])));
        const float r = pow2i(R[kk] - rn);
        const int c = kk >> 1, j = kk & 1;
#pragma unroll
        for (int t = 0; t < 10; ++t) {
          Y[t][4 * c + j] *= r;
          Y[t][4 * c + 2 + j] *= r;
        }
        lsg[kk] *= r;
        R[kk] = rn;
      }
      put_p();
      named_bar(1, 256);
      if (wg == 0 && tw < HN) red_mx[tw] = INT_MIN;  // read by every consumer thread before this barrier
    }
#pragma unroll
    for (int kk = 0; kk < 5; ++kk) lsg[kk] += ph[0][kk] + ph[1][kk];
    if (n == 0) TR(5);
    PF(3);
    // ---- pv: 10 tiles x 8 k-steps of 16 entries, two wgmmas per commit group
#ifndef AT_NO_PV
    const uint32_t pc = sp + par * (2 * PH_B);
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
#pragma unroll
      for (int t = 0; t < 10; ++t) {
        const uint32_t m0 = 0x00010001u << pv_pos(t, 0), m1 = 0x00010001u << pv_pos(t, 1);
        fence_acc(Y[t]);
        wg_arrive();
#pragma unroll
        for (int kk = k0; kk < k0 + 2; ++kk) {
          const uint32_t w0 = t < 5 ? W0[kk][0] : W1[kk][0], w1 = t < 5 ? W0[kk][1] : W1[kk][1];
          const uint32_t a[4] = {w0 & m0, w0 & m1, w1 & m0, w1 & m1};
          mma_f16(a, desc_sw128(pc + (kk >> 2) * PH_B + 32 * (kk & 3)), Y[t]);
        }
        wg_commit();
        fence_acc(Y[t]);
        wg_wait<2>();
      }
    }
#endif
    wg_wait<0>();  // no wgmma in flight across the loop back-edge (ptxas C7514)
#pragma unroll
    for (int t = 0; t < 10; ++t) fence_acc(Y[t]);
    if (n == 0) TR(6);
    PF(4);
  }
  TR(7);
#ifdef AT_PROF
  if (blockIdx.x == AT_PROF && tid == 0)
    for (int i = 0; i < 5; ++i) at_pf[i] = pf_a[i];
  if (blockIdx.x == AT_PROF && tid == 0) at_pf[5] = N;
#endif
  wg_wait<0>();
#pragma unroll
  for (int t = 0; t < 10; ++t) fence_acc(Y[t]);
  if (cur >= 0) a_flush(prm, cta + cur, Y, R, lsg, red_l, bit0, g, q, wml, s0 + C::OFF_FL);
  TR(8);
}

// ---- qproj: q' = q_nope @ dec^T (fp32, mma.sync bf16) and the int8 q record of each row (layout above). A cluster of
// NQCH CTAs (rank = code chunk of 128) takes BM (row, head) pairs; the row max |q'| over the 1280 codes and sum X
// are reduced over the cluster in shared memory (fixed order).
struct QParams {
  const __nv_bfloat16* qn;  // [B][H][512]
  const __nv_bfloat16* qr;  // [B][H][64]
  const __nv_bfloat16* dec;  // [BITS][512]
  uint8_t* qrec;             // [B][QREC_B]
  int64_t qn_sb, qn_sh, qr_sb, qr_sh;
  int R;  // B * H
};
template <int BM_, int NW_>
struct QCfg {
  static constexpr int BM = BM_, NW = NW_, NT = 32 * NW, MT = BM / 16, NS = NW / MT, TPW = 16 / NS;
  static constexpr int BN = 128, KD = 512, LDS = KD + 8;  // smem rows of 1040 B
  static constexpr int OFF_A = 0;
  static constexpr int OFF_B = OFF_A + BM * LDS * 2;
  static constexpr int OFF_ST = OFF_B + BN * LDS * 2;  // [BM][256]: hi row | lo row of the chunk image
  static constexpr int OFF_RED = OFF_ST + BM * 256;    // max [NS][BM] | rmax [BM] | sum [NS][BM] | rsum [BM]
  static constexpr int OFF_G = OFF_RED + 2 * (NS + 1) * BM * 4;  // pushed: max [NQCH][BM] | sum [NQCH][BM]
  static constexpr int SMEM = OFF_G + 2 * NQCH * BM * 4;
  static_assert(MT * NS == NW && TPW * NS == 16 && TPW % 2 == 0, "qproj warps");
  static_assert(SMEM <= 227 * 1024, "qproj smem");
};
// grid (NQCH, ceil(R / BM)), cluster (NQCH, 1, 1), 32 NW threads: warp w takes rows 16 (w % MT) .. + 15 and codes
// 8 TPW (w / MT) .. of the chunk (TPW n8 tiles); code kr = 8 jt + 2 t + e of tile jt sits at chunk position
// 16 (2 t + e) + jt, so the tiles 4 jq .. 4 jq + 3 fill one 32-bit word. Rows >= R are zero-filled and not stored.
// Every (row, code) is the same mma chain for each (BM, NW) and max / sum X are exact: the record does not depend on
// the variant.
template <int BM, int NW>
__global__ void __launch_bounds__(32 * NW, 1) pki8_qproj_kernel(const __grid_constant__ QParams prm) {
  using C = QCfg<BM, NW>;
  constexpr int NT = C::NT, NS = C::NS, TPW = C::TPW, RS = C::LDS * 2;
  extern __shared__ uint8_t qsm[];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int nc = cluster_rank(), m0 = BM * blockIdx.y, mt = warp % C::MT, ns = warp / C::MT;
  const uint32_t sa = smem_u32(qsm) + C::OFF_A, sb = smem_u32(qsm) + C::OFF_B;
  asm volatile("barrier.cluster.arrive.relaxed.aligned;\n" ::: "memory");
#pragma unroll
  for (int kg = 0; kg < 4; ++kg) {  // dec (weights) before the dependency wait
#pragma unroll
    for (int i = 0; i < 2048 / NT; ++i) {
      const int idx = tid + NT * i, n = idx / 16, c = idx % 16;
      cp16(sb + n * RS + (128 * kg + 8 * c) * 2, prm.dec + (int64_t)(128 * nc + n) * C::KD + 128 * kg + 8 * c);
    }
    cp_commit();
  }
  griddep_wait();
#pragma unroll
  for (int kg = 0; kg < 4; ++kg) {
#pragma unroll
    for (int i = 0; i < (16 * BM + NT - 1) / NT; ++i) {
      const int idx = tid + NT * i, m = idx / 16, c = idx % 16;
      if (16 * BM % NT == 0 || idx < 16 * BM) {
        const int r = min(m0 + m, prm.R - 1), b = r / H, h = r % H;
        cp16z(sa + m * RS + (128 * kg + 8 * c) * 2, prm.qn + b * prm.qn_sb + h * prm.qn_sh + 128 * kg + 8 * c,
              m0 + m < prm.R);
      }
    }
    cp_commit();
  }
  griddep_launch();
  float acc[TPW][4];
#pragma unroll
  for (int j = 0; j < TPW; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[j][i] = 0.f;
  const uint32_t a_ad = sa + (16 * mt + (lane & 7) + ((lane >> 3) & 1) * 8) * RS + (lane >> 4) * 16;
  const uint32_t b_ad = sb + (8 * TPW * ns + (lane >> 4) * 8 + (lane & 7)) * RS + ((lane >> 3) & 1) * 16;
#pragma unroll
  for (int kg = 0; kg < 4; ++kg) {
    cp_wait_le<3>(3 - kg);
    __syncthreads();
#pragma unroll
    for (int ks = 0; ks < 8; ++ks) {
      const int k2 = (128 * kg + 16 * ks) * 2;
      uint32_t a[4];
      ldsm4(a_ad + k2, a);
#pragma unroll
      for (int jp = 0; jp < TPW / 2; ++jp) {
        uint32_t bb[4];
        ldsm4(b_ad + 16 * jp * RS + k2, bb);
        mma_bf16_16816(acc[2 * jp], a, bb[0], bb[1]);
        mma_bf16_16816(acc[2 * jp + 1], a, bb[2], bb[3]);
      }
    }
  }
  // ---- row max |q'| over the cluster, K = 32639 / max
  float* red = reinterpret_cast<float*>(qsm + C::OFF_RED);
  int* reds = reinterpret_cast<int*>(red + (NS + 1) * BM);
  float mxa = 0.f, mxb = 0.f;
#pragma unroll
  for (int j = 0; j < TPW; ++j) {
    mxa = fmaxf(mxa, fmaxf(fabsf(acc[j][0]), fabsf(acc[j][1])));
    mxb = fmaxf(mxb, fmaxf(fabsf(acc[j][2]), fabsf(acc[j][3])));
  }
  mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
  mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
  mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
  mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
  const int ra = 16 * mt + g, rb = ra + 8;
  if (t == 0) {
    red[ns * BM + ra] = mxa;
    red[ns * BM + rb] = mxb;
  }
  __syncthreads();
  asm volatile("barrier.cluster.wait.aligned;\n" ::: "memory");  // every CTA of the cluster has started
  if (tid < BM) {
    float m = red[tid];
#pragma unroll
    for (int s = 1; s < NS; ++s) m = fmaxf(m, red[s * BM + tid]);
    const uint32_t a = smem_u32(qsm + C::OFF_G) + 4 * (nc * BM + tid);
#pragma unroll
    for (int rk = 0; rk < NQCH; ++rk) st_dsmem(dsmem_map(a, rk), __float_as_uint(m));
  }
  cluster_sync();
  if (tid < BM) {
    const float* gmx = reinterpret_cast<const float*>(qsm + C::OFF_G);
    float gm = 0.f;
#pragma unroll
    for (int rk = 0; rk < NQCH; ++rk) gm = fmaxf(gm, gmx[rk * BM + tid]);
    red[tid] = gm > 0.f ? 32639.f / gm : 1.f;
  }
  __syncthreads();
  // ---- int8 terms, staged as the swizzled hi | lo rows of the chunk image; sum X
  uint8_t* stg = qsm + C::OFF_ST;
  int sxa = 0, sxb = 0;
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int m = half ? rb : ra, qq = ((m0 + m) % H) % 4;
    const float K = red[m];
#pragma unroll
    for (int e = 0; e < 2; ++e) {
      uint8_t* ph = stg + m * 256 + (((2 * t + e) ^ (2 * qq)) << 4) + TPW * ns;
      uint8_t* pl = stg + m * 256 + 128 + (((2 * t + e) ^ (2 * qq + 1)) << 4) + TPW * ns;
#pragma unroll
      for (int jh = 0; jh < (TPW + 3) / 4; ++jh) {
        uint32_t whi = 0, wlo = 0;
#pragma unroll
        for (int jj = 0; jj < (TPW < 4 ? TPW : 4); ++jj) {
          const int xi = __float2int_rn(acc[4 * jh + jj][2 * half + e] * K);
          const int lo = ((xi + 128) & 255) - 128, hi = (xi - lo) >> 8;
          whi |= (uint32_t)(hi & 255) << (8 * jj);
          wlo |= (uint32_t)(lo & 255) << (8 * jj);
          if (half) sxb += xi; else sxa += xi;
        }
        if constexpr (TPW >= 4) {
          *reinterpret_cast<uint32_t*>(ph + 4 * jh) = whi;
          *reinterpret_cast<uint32_t*>(pl + 4 * jh) = wlo;
        } else {
          *reinterpret_cast<uint16_t*>(ph) = (uint16_t)whi;
          *reinterpret_cast<uint16_t*>(pl) = (uint16_t)wlo;
        }
      }
    }
  }
  sxa += __shfl_xor_sync(0xffffffffu, sxa, 1);
  sxa += __shfl_xor_sync(0xffffffffu, sxa, 2);
  sxb += __shfl_xor_sync(0xffffffffu, sxb, 1);
  sxb += __shfl_xor_sync(0xffffffffu, sxb, 2);
  if (t == 0) {
    reds[ns * BM + ra] = sxa;
    reds[ns * BM + rb] = sxb;
  }
  __syncthreads();
  if (tid < BM) {
    int s = reds[tid];
#pragma unroll
    for (int k = 1; k < NS; ++k) s += reds[k * BM + tid];
    st_dsmem(dsmem_map(smem_u32(qsm + C::OFF_G) + 4 * ((NQCH + nc) * BM + tid), 0), (uint32_t)s);
  }
#pragma unroll
  for (int i = 0; i < (16 * BM + NT - 1) / NT; ++i) {  // [BM rows][256 B] -> rows 8 kk + 2 qq, + 1 of chunk nc
    const int idx = tid + NT * i, m = idx / 16, c = idx % 16, r = m0 + m;
    if ((16 * BM % NT == 0 || idx < 16 * BM) && r < prm.R) {
      const int b = r / H, h = r % H, kk = h / 4, qq = h % 4;
      *reinterpret_cast<int4*>(prm.qrec + (int64_t)b * QREC_B + nc * QI_B + (8 * kk + 2 * qq) * 128 + 16 * c) =
          *reinterpret_cast<const int4*>(stg + m * 256 + 16 * c);
    }
  }
  if (nc == 1) {  // q_rope rows 8 (kk / 2) + 2 qq + kk % 2, SW128
#pragma unroll
    for (int i = 0; i < (8 * BM + NT - 1) / NT; ++i) {
      const int idx = tid + NT * i, m = idx / 8, c = idx % 8, r = m0 + m;
      if ((8 * BM % NT == 0 || idx < 8 * BM) && r < prm.R) {
        const int b = r / H, h = r % H, kk = h / 4, qq = h % 4, nr = 8 * (kk / 2) + 2 * qq + kk % 2;
        *reinterpret_cast<int4*>(prm.qrec + (int64_t)b * QREC_B + OFF_QR_REC + nr * 128 + ((c ^ (nr & 7)) << 4)) =
            __ldg(reinterpret_cast<const int4*>(prm.qr + b * prm.qr_sb + h * prm.qr_sh) + c);
      }
    }
  }
  asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");  // the pushes to rank 0 are done
  if (nc != 0) return;  // no remote reads of this CTA (an exited CTA counts as arrived)
  asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
  if (tid < BM && m0 + tid < prm.R) {
    const int* gsx = reinterpret_cast<const int*>(qsm + C::OFF_G) + NQCH * BM;
    int sx = 0;
#pragma unroll
    for (int rk = 0; rk < NQCH; ++rk) sx += gsx[rk * BM + tid];
    const int r = m0 + tid, b = r / H, h = r % H;
    uint8_t* pr = prm.qrec + (int64_t)b * QREC_B + OFF_PRM_REC;
    reinterpret_cast<int*>(pr)[h] = sx;
    reinterpret_cast<float*>(pr + 4 * HN)[h] = 1.f / red[tid];
  }
}
// ---- combine: the segments of row b -> A'' = (2 y - L) / L (fp16, c = 2 bit - 1) and lse_c = M + log2 L (log2 units);
// rows without units: A'' = 0, lse_c = -inf. CTA (r, kc) takes bits 128 kc + 4 lane; warp w sums the segments w, w + W,
// .. (W = blockDim / 32, fixed per batch size) and warp 0 adds the W partials in order. The row's segments: for U >= G
// every CTA has >= 1 unit and they are the CTAs i0 .. i1; for U < G a CTA has <= 1 unit and unit u is in CTA
// ceil((u + 1) G / U) - 1.
struct CParams {
  const int* plan;
  int B;
  const float* yp;
  const float* mlp;
  __half* a2;    // [B][H][BITS]
  float* lsec;   // [B][H]
};
constexpr int CW = 8;
__device__ __forceinline__ int seg_cta(int p, int i0, int u0, int U, int G) {
  return U >= G ? i0 + p : ((u0 + p + 1) * G + U - 1) / U - 1;
}
template <int DEP>
__global__ void __launch_bounds__(32 * CW) pki8_combine_kernel(const __grid_constant__ CParams prm) {
  __shared__ float4 sy[CW][32];
  __shared__ float sl[CW];
  const int r = blockIdx.x, b = r / H, h = r % H, lane = threadIdx.x % 32, w = threadIdx.x / 32, W = blockDim.x / 32;
  const int k = 128 * blockIdx.y + 4 * lane;
  const Plan pl(prm.plan, prm.B);  // written before qproj
  const int U = __ldg(pl.cu_u + prm.B), u0 = __ldg(pl.cu_u + b), u1 = __ldg(pl.cu_u + b + 1);
  const int G = plan_ncta(prm.plan, prm.B);
  if (prm.B >= 16) griddep_launch();
  griddep_wait();
  __half* ao = prm.a2 + (int64_t)r * BITS + k;
  if (u0 >= u1) {
    griddep_launch();
    if (w == 0) *reinterpret_cast<uint2*>(ao) = make_uint2(0u, 0u);
    if (blockIdx.y == 0 && threadIdx.x == 0) prm.lsec[r] = -INFINITY;
    return;
  }
  const int i0 = ((u0 + 1) * G + U - 1) / U - 1;
  const int ns = U >= G ? (u1 * G + U - 1) / U - i0 : u1 - u0;
  float M = -INFINITY;
  for (int p = lane; p < ns; p += 32) M = fmaxf(M, __ldg(prm.mlp + (int64_t)(seg_cta(p, i0, u0, U, G) + b) * 2 * HN + h));
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) M = fmaxf(M, __shfl_xor_sync(0xffffffffu, M, o));
  float L = 0.f;
  float4 y = make_float4(0.f, 0.f, 0.f, 0.f);
  if constexpr (DEP == 0) {
#pragma unroll 4
    for (int p = w; p < ns; p += W) {
      const int64_t i = seg_cta(p, i0, u0, U, G) + b;
      const float* ml = prm.mlp + i * 2 * HN;
      const float wt = pow2i((int)(__ldg(ml + h) - M));
      L += wt * __ldg(ml + HN + h);
      const float4 v = __ldg(reinterpret_cast<const float4*>(prm.yp + (i * H + h) * BITS + k));
      y.x += wt * v.x;
      y.y += wt * v.y;
      y.z += wt * v.z;
      y.w += wt * v.w;
    }
  } else {
    for (int p0 = w; p0 < ns; p0 += DEP * W) {
      float mq[DEP], lq[DEP];
      float4 vq[DEP];
#pragma unroll
      for (int e = 0; e < DEP; ++e) {
        const int p = p0 + W * e;
        if (p < ns) {
          const int64_t i = seg_cta(p, i0, u0, U, G) + b;
          const float* ml = prm.mlp + i * 2 * HN;
          mq[e] = __ldg(ml + h);
          lq[e] = __ldg(ml + HN + h);
          vq[e] = __ldg(reinterpret_cast<const float4*>(prm.yp + (i * H + h) * BITS + k));
        }
      }
#pragma unroll
      for (int e = 0; e < DEP; ++e) {
        if (p0 + W * e < ns) {
          const float wt = pow2i((int)(mq[e] - M));
          L += wt * lq[e];
          const float4 v = vq[e];
          y.x += wt * v.x;
          y.y += wt * v.y;
          y.z += wt * v.z;
          y.w += wt * v.w;
        }
      }
    }
  }
  sy[w][lane] = y;
  if (lane == 0) sl[w] = L;
  griddep_launch();
  __syncthreads();
  if (w != 0) return;
  for (int j = 1; j < W; ++j) {
    const float4 v = sy[j][lane];
    y.x += v.x;
    y.y += v.y;
    y.z += v.z;
    y.w += v.w;
    L += sl[j];
  }
  const float il = 1.f / L;
  const __half2 p0 = __floats2half2_rn((2.f * y.x - L) * il, (2.f * y.y - L) * il);
  const __half2 p1 = __floats2half2_rn((2.f * y.z - L) * il, (2.f * y.w - L) * il);
  uint2 o;
  o.x = *reinterpret_cast<const uint32_t*>(&p0);
  o.y = *reinterpret_cast<const uint32_t*>(&p1);
  *reinterpret_cast<uint2*>(ao) = o;
  if (blockIdx.y == 0 && lane == 0) prm.lsec[r] = M + log2f(L);
}

// ---- o_c = A'' @ dec (fp16 mma.sync, fp32) and the lse merge with the dense part (the attn kernel's dense role: o_d
// fp32, lse_d log2 units; fa3: o_d bf16, lse_d ln units)
__device__ __forceinline__ float4 ld_dsmem4(uint32_t a) {
  float4 v;
  asm volatile("ld.shared::cluster.v4.f32 {%0, %1, %2, %3}, [%4];\n" : "=f"(v.x), "=f"(v.y), "=f"(v.z), "=f"(v.w) : "r"(a) : "memory");
  return v;
}
struct MParams {
  const __half* a2;           // [R][BITS]
  const float* lsec;          // [R]
  const __nv_bfloat16* dec;   // [BITS][512]
  const void* od;             // [B][H][512] (od_sb, od_sh): fp32 when odf, else bf16
  const float* lsed;          // lse_d (ld_h, ld_b): log2 units when odf, else ln
  __nv_bfloat16* out;         // [B][H][512] (o_sb, o_sh)
  int64_t od_sb, od_sh, ld_h, ld_b, o_sb, o_sh;
  int R, odf;
};
struct MCfg {
  static constexpr int BM = 64, BN = 64, BK = 64, LDA = BK + 8, LDB = BN + 8, NK = BITS / BK;
  static constexpr int A_B = BM * LDA * 2, B_B = BK * LDB * 2;
};
// chunk buffers of a rank: all NK / KS chunks for KS >= 4 (the A loads issued at once: one L2 round trip instead of
// one per chunk; grids of <= 4 row tiles), else a double buffer
template <int KS>
struct MBuf {
  static constexpr int NB = KS >= 4 ? MCfg::NK / KS : 2;
  static constexpr int SMEM = NB * (MCfg::A_B + MCfg::B_B);
  static_assert(KS < 4 || MCfg::NK % KS == 0, "merge chunks");
};
// grid (KS ceil(R / 64), 512 / 64), clusters of KS along x (rank k takes chunks k NK / KS ..), 256 threads: warp w takes rows 16 (w & 3) .. + 15, columns 32 (w >> 2) .. + 31
template <int KS>
__global__ void __launch_bounds__(256) pki8_merge_kernel(const __grid_constant__ MParams prm) {
  using C = MCfg;
  constexpr int NB = MBuf<KS>::NB;
  extern __shared__ __align__(128) uint8_t msm[];
  const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32, g = lane / 4, t = lane % 4;
  const int rk = (int)blockIdx.x % KS, kb = rk * C::NK / KS, ke = (rk + 1) * C::NK / KS;
  const int m0 = C::BM * ((int)blockIdx.x / KS), n0 = C::BN * blockIdx.y, mt = warp & 3, nh = warp >> 2;
  const uint32_t sA = smem_u32(msm), sB = sA + NB * C::A_B;
  // dec chunk kc: [64 k][64 n] bf16, thread: k = tid / 4, n = 16 (tid % 4) .. + 15
  const int dk = tid / 4, dn = 16 * (tid % 4);
  auto ld_dec = [&](int kc, uint4 (&v)[2]) {
    const uint4* p = reinterpret_cast<const uint4*>(prm.dec + (int64_t)(C::BK * kc + dk) * 512 + n0 + dn);
    v[0] = __ldg(p);
    v[1] = __ldg(p + 1);
  };
  auto st_dec = [&](int buf, const uint4 (&v)[2]) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t w[4] = {v[i].x, v[i].y, v[i].z, v[i].w};
      uint32_t o[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 f = __bfloat1622float2(*reinterpret_cast<const __nv_bfloat162*>(&w[j]));
        const __half2 hh = __floats2half2_rn(f.x, f.y);
        o[j] = *reinterpret_cast<const uint32_t*>(&hh);
      }
      *reinterpret_cast<uint4*>(msm + NB * C::A_B + buf * C::B_B + (dk * C::LDB + dn + 8 * i) * 2) =
          make_uint4(o[0], o[1], o[2], o[3]);
    }
  };
  auto ld_a = [&](int kc, int buf) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int idx = tid + 256 * i, m = idx / 8, c = idx % 8, r = min(m0 + m, prm.R - 1);
      cp16z(sA + buf * C::A_B + (m * C::LDA + 8 * c) * 2, prm.a2 + (int64_t)r * BITS + C::BK * kc + 8 * c,
            m0 + m < prm.R);  // rows >= R: zeros (not stored)
    }
    cp_commit();
  };
  float acc[4][4];
#pragma unroll
  for (int j = 0; j < 4; ++j)
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[j][i] = 0.f;
  const uint32_t a_off = ((16 * mt + (lane & 7) + ((lane >> 3) & 1) * 8) * C::LDA + (lane >> 4) * 8) * 2;
  const uint32_t b_off = ((((lane >> 3) & 1) * 8 + (lane & 7)) * C::LDB + 32 * nh + (lane >> 4) * 8) * 2;
  auto mma_chunk = [&](int buf) {
    const uint32_t ab = sA + buf * C::A_B + a_off, bb0 = sB + buf * C::B_B + b_off;
#pragma unroll
    for (int ks = 0; ks < 4; ++ks) {
      uint32_t a[4];
      ldsm4(ab + 32 * ks, a);
#pragma unroll
      for (int jp = 0; jp < 2; ++jp) {
        uint32_t bb[4];
        ldsm4t(bb0 + (16 * ks * C::LDB + 16 * jp) * 2, bb);
        mma_f16_16816(acc[2 * jp], a, bb[0], bb[1]);
        mma_f16_16816(acc[2 * jp + 1], a, bb[2], bb[3]);
      }
    }
  };
  // the epilogue operands of the thread's rows (rank 0), loaded after the dependency wait with the A chunks
  float2 dpre[2][4];
  float lpre[2][2];
  auto ld_epi = [&]() {
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int r = min(m0 + 16 * mt + g + 8 * half, prm.R - 1), b = r / H, h = r % H;
      const int64_t odo = b * prm.od_sb + h * prm.od_sh;
      lpre[half][0] = __ldg(prm.lsed + h * prm.ld_h + b * prm.ld_b);
      lpre[half][1] = __ldg(prm.lsec + r);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        if (j % KS != rk) continue;
        const int n = n0 + 32 * nh + 8 * j + 2 * t;
        dpre[half][j] =
            prm.odf ? *reinterpret_cast<const float2*>(static_cast<const float*>(prm.od) + odo + n)
                    : __bfloat1622float2(
                          *reinterpret_cast<const __nv_bfloat162*>(static_cast<const __nv_bfloat16*>(prm.od) + odo + n));
      }
    }
  };
  if constexpr (NB > 2) {
    {
      uint4 dv[NB][2];
#pragma unroll
      for (int i = 0; i < NB; ++i) ld_dec(kb + i, dv[i]);
#pragma unroll
      for (int i = 0; i < NB; ++i) st_dec(i, dv[i]);
    }
    griddep_wait();
#pragma unroll
    for (int i = 0; i < NB; ++i) ld_a(kb + i, i);
    ld_epi();
#pragma unroll
    for (int i = 0; i < NB; ++i) {
      cp_wait_le<NB - 1>(NB - 1 - i);
      __syncthreads();  // A of chunk i (and every dec chunk) visible
      mma_chunk(i);
    }
  } else {
    uint4 dv[2];
    ld_dec(kb, dv);
    griddep_wait();
    ld_a(kb, 0);
    ld_epi();
    st_dec(0, dv);
    for (int kc = kb; kc < ke; ++kc) {
      const int buf = (kc - kb) & 1;
      const bool more = kc + 1 < ke;
      cp_wait<0>();
      __syncthreads();  // A / dec of chunk kc visible; chunk kc - 1 (the other buffer) no longer read
      if (more) {
        ld_a(kc + 1, buf ^ 1);
        ld_dec(kc + 1, dv);
      }
      mma_chunk(buf);
      if (more) st_dec(buf ^ 1, dv);
    }
  }
  if constexpr (KS > 1) {  // partials [4 fragments][256 threads] float4, summed in rank order (deterministic)
    __syncthreads();
    float4* rs = reinterpret_cast<float4*>(msm);
#pragma unroll
    for (int j = 0; j < 4; ++j) rs[256 * j + tid] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]);
    cluster_sync();
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (j % KS != rk) continue;
      const uint32_t a = smem_u32(rs + 256 * j + tid);
      float4 v = ld_dsmem4(dsmem_map(a, 0));
#pragma unroll
      for (int k = 1; k < KS; ++k) {
        const float4 w = ld_dsmem4(dsmem_map(a, k));
        v.x += w.x;
        v.y += w.y;
        v.z += w.z;
        v.w += w.w;
      }
      acc[j][0] = v.x;
      acc[j][1] = v.y;
      acc[j][2] = v.z;
      acc[j][3] = v.w;
    }
    asm volatile("barrier.cluster.arrive.release.aligned;\n" ::: "memory");  // remote reads done (waited at exit)
  }
  griddep_launch();
#pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int r = m0 + 16 * mt + g + 8 * half;
    if (r >= prm.R) continue;
    const int b = r / H, h = r % H;
    const float lr = lpre[half][0];
    const float lc = lpre[half][1], ld = prm.odf ? lr : lr * 1.4426950408889634f;
    const float mx = fmaxf(lc, ld), wc = exp2f(lc - mx), wd = exp2f(ld - mx), iv = 1.f / (wc + wd);
    __nv_bfloat16* orow = prm.out + b * prm.o_sb + h * prm.o_sh;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      if (j % KS != rk) continue;
      const int n = n0 + 32 * nh + 8 * j + 2 * t;
      const float2 d = dpre[half][j];  // (wd d + wc acc) / l, rounding order fixed (not left to fma contraction)
      *reinterpret_cast<__nv_bfloat162*>(orow + n) =
          __floats2bfloat162_rn(__fmul_rn(__fmaf_rn(wd, d.x, __fmul_rn(wc, acc[j][2 * half])), iv),
                                __fmul_rn(__fmaf_rn(wd, d.y, __fmul_rn(wc, acc[j][2 * half + 1])), iv));
    }
  }
  if constexpr (KS > 1) asm volatile("barrier.cluster.wait.acquire.aligned;\n" ::: "memory");
}

#define PKI8_CHECK(x)                                                                                       \
  do {                                                                                                      \
    cudaError_t e_ = (x);                                                                                   \
    if (e_ != cudaSuccess) {                                                                                \
      fprintf(stderr, "pki8: %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__);                   \
      exit(1);                                                                                              \
    }                                                                                                       \
  } while (0)

// PDL launch (programmatic stream serialization) when pdl; a cluster of cl CTAs along x when cl > 1
template <typename K, typename P>
void launch_k(K kern, dim3 grid, int threads, int smem, cudaStream_t stream, bool pdl, int cl, const P& prm) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = grid;
  cfg.blockDim = dim3(threads, 1, 1);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute at[2];
  int na = 0;
  if (pdl) {
    at[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    at[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  if (cl > 1) {
    at[na].id = cudaLaunchAttributeClusterDimension;
    at[na].val.clusterDim.x = cl;
    at[na].val.clusterDim.y = 1;
    at[na].val.clusterDim.z = 1;
    ++na;
  }
  cfg.attrs = at;
  cfg.numAttrs = na;
  PKI8_CHECK(cudaLaunchKernelEx(&cfg, kern, prm));
}

// one qproj variant (BM rows x NW warps per CTA): attributes and the active clusters of one wave, once per variant
template <int BM, int NW>
struct QVar {
  static int max_clusters() {
    static const int n = [] {
      using C = QCfg<BM, NW>;
      PKI8_CHECK(cudaFuncSetAttribute(pki8_qproj_kernel<BM, NW>, cudaFuncAttributeMaxDynamicSharedMemorySize, C::SMEM));
      PKI8_CHECK(cudaFuncSetAttribute(pki8_qproj_kernel<BM, NW>, cudaFuncAttributeNonPortableClusterSizeAllowed, 1));
      cudaLaunchConfig_t cfg = {};
      cfg.gridDim = dim3(NQCH, 64, 1);
      cfg.blockDim = dim3(C::NT, 1, 1);
      cfg.dynamicSmemBytes = C::SMEM;
      cudaLaunchAttribute at[1];
      at[0].id = cudaLaunchAttributeClusterDimension;
      at[0].val.clusterDim.x = NQCH;
      at[0].val.clusterDim.y = 1;
      at[0].val.clusterDim.z = 1;
      cfg.attrs = at;
      cfg.numAttrs = 1;
      int v = 0;
      PKI8_CHECK(cudaOccupancyMaxActiveClusters(&v, pki8_qproj_kernel<BM, NW>, &cfg));
      return v;
    }();
    return n;
  }
  static void launch(const QParams& prm, cudaStream_t stream, bool pdl) {
    (void)max_clusters();
    launch_k(pki8_qproj_kernel<BM, NW>, dim3(NQCH, (unsigned)((prm.R + BM - 1) / BM)), QCfg<BM, NW>::NT,
             QCfg<BM, NW>::SMEM, stream, pdl, NQCH, prm);
  }
};

template <int KS>
void merge_launch(dim3 grid, cudaStream_t stream, bool pdl, const MParams& prm) {
  if constexpr (MBuf<KS>::SMEM > 48 * 1024) {
    static const bool init = [] {
      PKI8_CHECK(cudaFuncSetAttribute(pki8_merge_kernel<KS>, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                      MBuf<KS>::SMEM));
      return true;
    }();
    (void)init;
  }
  launch_k(pki8_merge_kernel<KS>, grid, 256, MBuf<KS>::SMEM, stream, pdl, KS, prm);
}

}  // namespace pki8

inline void pki8_qproj_dispatch(tvm::ffi::TensorView qn, tvm::ffi::TensorView qr, tvm::ffi::TensorView dec,
                                tvm::ffi::TensorView qrec, int64_t B, int64_t qn_sb, int64_t qn_sh, int64_t qr_sb,
                                int64_t qr_sh, int64_t pdl, int64_t stream) {
  namespace K = pki8;
  K::QParams prm;
  prm.qn = static_cast<const __nv_bfloat16*>(qn.data_ptr());
  prm.qr = static_cast<const __nv_bfloat16*>(qr.data_ptr());
  prm.dec = static_cast<const __nv_bfloat16*>(dec.data_ptr());
  prm.qrec = static_cast<uint8_t*>(qrec.data_ptr());
  prm.qn_sb = qn_sb;
  prm.qn_sh = qn_sh;
  prm.qr_sb = qr_sb;
  prm.qr_sh = qr_sh;
  prm.R = (int)B * K::H;
  // the smallest BM whose ceil(R / BM) clusters fit one wave (every variant is 1 CTA / SM)
  const int nm = K::QVar<64, 8>::max_clusters();
  const int qv = (prm.R + 15) / 16 <= nm ? 0 : (prm.R + 31) / 32 <= nm ? 1 : 2;  // QV
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  if (qv == 0)
    K::QVar<16, 8>::launch(prm, st, pdl != 0);
  else if (qv == 1)
    K::QVar<32, 8>::launch(prm, st, pdl != 0);
  else
    K::QVar<64, 8>::launch(prm, st, pdl != 0);
}

inline void pki8_attn_dispatch(tvm::ffi::TensorView qrec, tvm::ffi::TensorView arena, tvm::ffi::TensorView r2t,
                               tvm::ffi::TensorView plan, tvm::ffi::TensorView yp, tvm::ffi::TensorView mlp, int64_t B,
                               int64_t r2t_ld, double scale_log2, int64_t G, tvm::ffi::TensorView qn,
                               tvm::ffi::TensorView qr, int64_t qn_sb, int64_t qn_sh, int64_t qr_sb, int64_t qr_sh,
                               tvm::ffi::TensorView dpt, tvm::ffi::TensorView dlen, int64_t dw,
                               tvm::ffi::TensorView dense, tvm::ffi::TensorView od, tvm::ffi::TensorView lsed, int64_t nd,
                               int64_t pdl, int64_t stream) {
  namespace K = pki8;
  K::AParams prm;
  prm.qrec = static_cast<const uint8_t*>(qrec.data_ptr());
  prm.arena = static_cast<const uint8_t*>(arena.data_ptr());
  prm.r2t = static_cast<const int*>(r2t.data_ptr());
  prm.r2t_ld = r2t_ld;
  prm.plan = static_cast<const int*>(plan.data_ptr());
  prm.B = (int)B;
  prm.yp = static_cast<float*>(yp.data_ptr());
  prm.mlp = static_cast<float*>(mlp.data_ptr());
  prm.scale_log2 = (float)scale_log2;
  prm.qn = static_cast<const __nv_bfloat16*>(qn.data_ptr());
  prm.qr = static_cast<const __nv_bfloat16*>(qr.data_ptr());
  prm.qn_sb = qn_sb;
  prm.qn_sh = qn_sh;
  prm.qr_sb = qr_sb;
  prm.qr_sh = qr_sh;
  prm.dpt = static_cast<const int*>(dpt.data_ptr());
  prm.dlen = static_cast<const int*>(dlen.data_ptr());
  prm.dw = (int)dw;
  prm.dense = static_cast<const __nv_bfloat16*>(dense.data_ptr());
  prm.od = static_cast<float*>(od.data_ptr());
  prm.lsed = static_cast<float*>(lsed.data_ptr());
  prm.ND = (int)nd;
  if (nd > 0 && (dw > K::DCfg::MAXDW || (nd != B && (!PKI8_DSPLIT || nd != 3 * B)))) {
    fprintf(stderr, "pki8: dense role needs dw <= %d and nd == B (or 3 B with PKI8_DSPLIT) (dw %lld nd %lld B %lld)\n", K::DCfg::MAXDW,
            (long long)dw, (long long)nd, (long long)B);
    exit(1);
  }
  static const bool init = [] {
    PKI8_CHECK(cudaFuncSetAttribute(K::pki8_attn_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, K::ACfg::SMEM));
    return true;
  }();
  (void)init;
  K::launch_k(K::pki8_attn_kernel, dim3((unsigned)(nd + G)), 384, K::ACfg::SMEM, reinterpret_cast<cudaStream_t>(stream),
              pdl != 0, 1, prm);
}

inline void pki8_combine_dispatch(tvm::ffi::TensorView plan, tvm::ffi::TensorView yp, tvm::ffi::TensorView mlp,
                                  tvm::ffi::TensorView a2, tvm::ffi::TensorView lsec, int64_t B, int64_t pdl,
                                  int64_t stream) {
  namespace K = pki8;
  K::CParams prm;
  prm.plan = static_cast<const int*>(plan.data_ptr());
  prm.B = (int)B;
  prm.yp = static_cast<const float*>(yp.data_ptr());
  prm.mlp = static_cast<const float*>(mlp.data_ptr());
  prm.a2 = static_cast<__half*>(a2.data_ptr());
  prm.lsec = static_cast<float*>(lsec.data_ptr());
  const int W = B <= 2 ? 8 : B <= 8 ? 4 : B <= 16 ? 2 : 1;  // ~G / B + 1 segments per row
  K::launch_k(B <= 4 ? K::pki8_combine_kernel<8> : K::pki8_combine_kernel<0>, dim3((unsigned)(B * K::H), K::BITS / 128), 32 * W, 0, reinterpret_cast<cudaStream_t>(stream),
              pdl != 0, 1, prm);
}

inline void pki8_merge_dispatch(tvm::ffi::TensorView a2, tvm::ffi::TensorView lsec, tvm::ffi::TensorView dec,
                                tvm::ffi::TensorView od, tvm::ffi::TensorView lsed, tvm::ffi::TensorView out, int64_t B,
                                int64_t od_sb, int64_t od_sh, int64_t ld_h, int64_t ld_b, int64_t o_sb, int64_t o_sh,
                                int64_t ks, int64_t odf, int64_t pdl, int64_t stream) {
  namespace K = pki8;
  K::MParams prm;
  prm.a2 = static_cast<const __half*>(a2.data_ptr());
  prm.lsec = static_cast<const float*>(lsec.data_ptr());
  prm.dec = static_cast<const __nv_bfloat16*>(dec.data_ptr());
  prm.od = od.data_ptr();
  prm.lsed = static_cast<const float*>(lsed.data_ptr());
  prm.out = static_cast<__nv_bfloat16*>(out.data_ptr());
  prm.od_sb = od_sb;
  prm.od_sh = od_sh;
  prm.ld_h = ld_h;
  prm.ld_b = ld_b;
  prm.o_sb = o_sb;
  prm.o_sh = o_sh;
  prm.R = (int)B * K::H;
  prm.odf = (int)odf;
  const int mtl = (prm.R + K::MCfg::BM - 1) / K::MCfg::BM;
  const int KS = ks > 0 ? (int)ks : (mtl <= 4 ? 4 : mtl <= 24 ? 2 : 1);
  const dim3 grid((unsigned)(KS * mtl), 512 / K::MCfg::BN);
  const cudaStream_t st = reinterpret_cast<cudaStream_t>(stream);
  if (KS == 4)
    K::merge_launch<4>(grid, st, pdl != 0, prm);
  else if (KS == 2)
    K::merge_launch<2>(grid, st, pdl != 0, prm);
  else
    K::merge_launch<1>(dim3((unsigned)mtl, 512 / K::MCfg::BN), st, pdl != 0, prm);
}

}  // namespace sglang
