// Tensor-core tail main kernel (included inside cuda_kernel.cu's anonymous
// namespace, after the popc runtime-unrolled kernel it mirrors).
//
// Same contract, grid, arguments, quantizer and float expressions as
// picokv_v1_expbits_atomic2_runtime_unrolled_kernel<4, KEY_T, EB, 0, LB, 5>
// (GQA-4, Ck=320, unpruned); only the two integer inner products move from
// __popcll to mma.sync b1 AND+POPC:
//
// score  dot(h,l) = sum_i q_i (1 - 2 k_i) = Sq_h - 2 sum_i q_i k_i with the
//        int8 query (sign + 7 magnitude planes) rewritten as two's-complement
//        planes T_0..T_7 (T_7 weight -128): sum_i q_i k_i =
//        sum_b w_b popc(T_b & k). Per 16 tokens: m16n8k256 + m16n8k128 (320
//        key bits) for each n-tile j = 0..3, B column n = (head n>>1,
//        plane 2j + (n&1)), combined by Horner over j (x4 per step, plane 7
//        negated). The lane ends with head tig's dots of tokens g and g+8.
// value  pos(h,c,G) = sum_e 2^e popc(v_{c,G} & P_{h,G,e}): m16n8k128 per
//        (16 codes, group pair, plane e); A row = code c's bits of the two
//        64-token groups, B column n = (head n>>1, group n&1) block-diagonal
//        (zero outside its own group), Horner over e (x2 per step).
//
// Both inner products are exact integers, and every float expression and the
// per-(head, code) summation order (tile by tile, group 0..3) match the popc
// kernel, so single-writer outputs are bitwise identical.
//
// Memory: each tile's keys (256 tokens x 5 words) and values (4 groups x Cv
// codes) are staged in shared memory with cp.async -- keys of tile t+1 load
// during the quantizer/value phases of tile t, values of tile t during its
// score phase -- so the BMMA phases only read shared memory. Token rows past
// L_b and dead/out-of-range groups read stale shared data: every BMMA output
// row depends only on its own token/code, other-group B segments are zero,
// and those outputs are discarded, as with the zero-filled loads they replace.

__device__ __forceinline__ void picokv_mma_b1_k256(
  int (&d)[4], const unsigned int a0, const unsigned int a1, const unsigned int a2,
  const unsigned int a3, const unsigned int b0, const unsigned int b1
) {
  asm("mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void picokv_mma_b1_k128(
  int (&d)[4], const unsigned int a0, const unsigned int a1, const unsigned int b0
) {
  asm("mma.sync.aligned.m16n8k128.row.col.s32.b1.b1.s32.and.popc "
      "{%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a0), "r"(a1), "r"(b0));
}

__device__ __forceinline__ void picokv_cp_async8(void *smem_ptr, const void *gmem_ptr) {
  const unsigned int smem_addr = (unsigned int)__cvta_generic_to_shared(smem_ptr);
  asm volatile("cp.async.ca.shared.global [%0], [%1], 8;\n" :: "r"(smem_addr), "l"(gmem_ptr) : "memory");
}

__device__ __forceinline__ void picokv_cp_async_commit() {
  asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int Pending>
__device__ __forceinline__ void picokv_cp_async_wait() {
  asm volatile("cp.async.wait_group %0;\n" :: "n"(Pending) : "memory");
}

// Order-preserving int key of a float (monotone for non-NaN values): a warp
// max over keys selects the same element as the fmaxf shuffle tree.
__device__ __forceinline__ int picokv_float_order_key(const float x) {
  const int bits = __float_as_int(x);
  return bits ^ ((bits >> 31) & 0x7fffffff);
}

__device__ __forceinline__ float picokv_float_from_order_key(const int key) {
  return __int_as_float(key ^ ((key >> 31) & 0x7fffffff));
}

template <bool KeyTransposedFixed, int ExpBits, int CvT = 320, int MinBlocksPerSm = 4, int BlockTilesT = 2>
__global__ void __launch_bounds__(256, MinBlocksPerSm) picokv_v1_expbits_atomic2_tc_kernel(
  const long *__restrict__ packed_projected_query,
  const long *__restrict__ packed_key,
  const long *__restrict__ packed_value_sign,
  const float *__restrict__ query_scale,
  const float *__restrict__ dense_lse,
  float *__restrict__ atomic_code_output, // [B,QH,S,Cv]
  float *__restrict__ atomic_den,         // [B,QH,S]
  const int B,
  const int QH,
  const int KVH,
  const int S,
  const int L,
  const int LGroups,
  const int KBlocks,
  const int Cv,
  const int query_scale_numel,
  const float attention_scale,
  const float prune_log_eps,
  const int *__restrict__ global_lens,
  const int *__restrict__ sched_req,
  const int *__restrict__ sched_tile
) {
  constexpr int Threads = 256;
  constexpr int GroupSize = 4;
  constexpr int KB = 5;                       // Ck = 320 key bits = 256 + 64
  constexpr int PlaneSegs = KB * 2;           // 32-bit segments per query plane
  constexpr int BlockTiles = BlockTilesT;
  constexpr int TileGroups = kTileSize / 64;
  constexpr int QuantWarps = 8;
  constexpr int QuantPasses = (GroupSize * TileGroups + QuantWarps - 1) / QuantWarps;
  constexpr int ExpMaxCode = (1 << ExpBits) - 1;
  constexpr int TokenBlocks = kTileSize / 16;  // m16 score tiles per 256-token tile
  constexpr int ScorePasses = TokenBlocks / QuantWarps;
  constexpr int CodeTiles = CvT / 16;          // m16 value tiles
  constexpr int WarpCodeTiles = (CodeTiles + QuantWarps - 1) / QuantWarps;
  // The score epilogue stores (4 heads x 8 tokens) per instruction; an 8-word
  // row pad puts them in 32 distinct banks.
  constexpr int WeightStride = kTileSize + 8;
  // Staged tiles in u32 words. Row strides are 8 (keys, u64 fragment reads)
  // and 16 (values, u32 fragment reads) mod 32 words: conflict-free loads.
  constexpr int KeyStride = 2 * kTileSize + 8;   // per key word kb
  constexpr int ValueStride = 2 * CvT + 16;      // per 64-token group
  constexpr int KeyChunks = KB * kTileSize;      // 8-byte cp.async chunks
  constexpr int ValueChunks = TileGroups * CvT;
  static_assert(CvT % 16 == 0, "tc tail: Cv must be a multiple of 16");
  static_assert(kQMagBits == 7, "tc tail: int8 query (sign + 7 magnitude planes)");
  (void)KBlocks;
  (void)Cv;
  (void)prune_log_eps;

  const int _sched_bx = blockIdx.x;
  const int tile_block_idx = (sched_tile != nullptr) ? sched_tile[_sched_bx] : _sched_bx;
  const int S_idx = blockIdx.y;
  const int BKVH_idx = blockIdx.z;
  const int B_idx = (sched_req != nullptr) ? sched_req[_sched_bx] : (BKVH_idx / KVH);
  const int KVH_idx = (sched_req != nullptr) ? BKVH_idx : (BKVH_idx - B_idx * KVH);
  const int thread_idx = threadIdx.x;
  const int qh_base = KVH_idx * GroupSize;
  const int lane_idx = thread_idx & 31;
  const int warp_idx = thread_idx >> 5;
  const int g = lane_idx >> 2;    // mma groupID
  const int tig = lane_idx & 3;   // mma thread-in-group

  const int L_b = (global_lens != nullptr) ? global_lens[B_idx] : L;
  if (tile_block_idx * (BlockTiles * kTileSize) >= L_b) return;

  __shared__ int query_sum[GroupSize];        // sum_i q_i (signed int8 query)
  __shared__ float scale_cache[GroupSize];
  __shared__ float lse_cache[GroupSize];
  __shared__ unsigned int query_planes[GroupSize][8][PlaneSegs];
  __shared__ float weights[GroupSize][WeightStride];
  __shared__ unsigned int exp_planes[GroupSize][TileGroups][ExpBits][2];
  __shared__ float exp_scale2[GroupSize][TileGroups];
  __shared__ float exp_bias[GroupSize][TileGroups];
  __shared__ unsigned int live_bits;          // per tile: groups with any scale2 > 0
  __shared__ __align__(16) unsigned int key_stage[KB * KeyStride];
  __shared__ __align__(16) unsigned int value_stage[TileGroups * ValueStride];

  // Chunk i = r * Threads + thread of a tile: keys are (kb = r, token =
  // thread) when transposed, else the tile's contiguous [token][kb] words;
  // values are the tile's contiguous [group][code] words.
  const long kv_head = (long)(B_idx * KVH + KVH_idx);
  const long *__restrict__ key_src = packed_key + kv_head * L * KB + thread_idx;
  const long *__restrict__ value_src = packed_value_sign + kv_head * LGroups * CvT + thread_idx;
  const int group_end = min(LGroups, (L_b + 63) >> 6);  // groups holding a token < L_b
  // Keys of `tile` with token < L_b.
  auto stage_keys = [&](const int tile) {
    #pragma unroll
    for (int r = 0; r < (KeyChunks + Threads - 1) / Threads; r++) {
      const int i = r * Threads + thread_idx;
      if constexpr (KeyTransposedFixed) {
        if (tile * kTileSize + thread_idx < L_b) {
          picokv_cp_async8(&key_stage[r * KeyStride + 2 * thread_idx], key_src + (long)r * L + tile * kTileSize);
        }
      } else {
        const int tok = i / KB;
        const int kb = i - tok * KB;
        if (i < KeyChunks && tile * kTileSize + tok < L_b) {
          picokv_cp_async8(&key_stage[kb * KeyStride + 2 * tok], key_src + (long)tile * KeyChunks + r * Threads);
        }
      }
    }
  };
  // Values of `tile`'s groups below group_end.
  auto stage_values = [&](const int tile) {
    #pragma unroll
    for (int r = 0; r < (ValueChunks + Threads - 1) / Threads; r++) {
      const int i = r * Threads + thread_idx;
      const int lg = i / CvT;
      if (i < ValueChunks && tile * TileGroups + lg < group_end) {
        picokv_cp_async8(&value_stage[2 * i + (ValueStride - 2 * CvT) * lg],
                         value_src + (long)tile * ValueChunks + r * Threads);
      }
    }
  };

  const int first_tile = tile_block_idx * BlockTiles;
  stage_keys(first_tile);
  picokv_cp_async_commit();
  stage_values(first_tile);
  picokv_cp_async_commit();
  // Two's-complement query planes, bit-sliced: thread (head, 32-dim segment)
  // computes T = (M ^ S) + S per dimension lane with a ripple carry (M_7 = 0),
  // half-warp per head. Segment s holds key word s>>1, half s&1.
  if (thread_idx < GroupSize * 16) {  // warps 0-1
    const int hrel = thread_idx >> 4;
    const int seg = thread_idx & 15;
    int part = 0;
    if (seg < PlaneSegs) {
      const int qh = qh_base + hrel;
      const unsigned int *__restrict__ q32 = reinterpret_cast<const unsigned int *>(packed_projected_query) +
        ((((long)(B_idx * QH + qh) * S + S_idx) * KB + (seg >> 1)) * 8) * 2 + (seg & 1);
      const unsigned int sign = q32[0];
      unsigned int carry = sign;  // +1 on negated dimensions
      #pragma unroll
      for (int plane = 0; plane < kQMagBits; plane++) {
        const unsigned int x = q32[2 * (plane + 1)] ^ sign;
        const unsigned int t = x ^ carry;
        carry &= x;
        query_planes[hrel][plane][seg] = t;
        part += __popc(t) << plane;
      }
      const unsigned int t7 = sign ^ carry;
      query_planes[hrel][7][seg] = t7;
      part -= __popc(t7) << 7;
    }
    #pragma unroll
    for (int offset = 8; offset > 0; offset >>= 1) {
      part += __shfl_xor_sync(0xffffffffu, part, offset);
    }
    if (seg == 0) {
      query_sum[hrel] = part;
    }
  }
  if (thread_idx < GroupSize) {
    const int qh = qh_base + thread_idx;
    const long lse_idx = (long)(B_idx * QH + qh) * S + S_idx;
    const float query_scale_value = query_scale_numel == 1 ? query_scale[0] : query_scale[lse_idx];
    scale_cache[thread_idx] = query_scale_value * attention_scale;
    lse_cache[thread_idx] = dense_lse[lse_idx];
  }
  __syncthreads();

  // Score B fragments are block-constant: column n = g -> (head g>>1, plane
  // 2j + (g&1)). k256 segment tig = low half of key word tig, segment 4+tig =
  // its high half; the k128 chunk carries key word 4 in segments 0-1.
  unsigned int score_b256[4][2];
  unsigned int score_b128[4];
  #pragma unroll
  for (int j = 0; j < 4; j++) {
    const int plane = 2 * j + (g & 1);
    score_b256[j][0] = query_planes[g >> 1][plane][2 * tig];
    score_b256[j][1] = query_planes[g >> 1][plane][2 * tig + 1];
    score_b128[j] = tig < 2 ? query_planes[g >> 1][plane][8 + tig] : 0u;
  }

  const unsigned long long *key_stage64 = reinterpret_cast<const unsigned long long *>(key_stage);

  float acc[WarpCodeTiles][2];
  #pragma unroll
  for (int t = 0; t < WarpCodeTiles; t++) {
    acc[t][0] = 0.0f;
    acc[t][1] = 0.0f;
  }
  float den_acc = 0.0f;

  #pragma unroll
  for (int block_tile = 0; block_tile < BlockTiles; block_tile++) {
    const int tile_idx = first_tile + block_tile;
    // A tile entirely past L_b contributes exactly nothing in the popc kernel
    // (weights 0 / -inf -> scale 0, bias -0); block-uniform exit.
    if (tile_idx * kTileSize >= L_b) break;
    if (block_tile > 0) {
      stage_values(tile_idx);
      picokv_cp_async_commit();
    }
    picokv_cp_async_wait<1>();  // this tile's keys (its values may pend)
    __syncthreads();

    #pragma unroll
    for (int pass = 0; pass < ScorePasses; pass++) {
      const int token_block = pass * QuantWarps + warp_idx;
      const int local0 = token_block * 16 + g;
      const int l0 = tile_idx * kTileSize + local0;
      const int l1 = l0 + 8;
      const unsigned long long key0 = key_stage64[tig * (KeyStride / 2) + local0];
      const unsigned long long key1 = key_stage64[tig * (KeyStride / 2) + local0 + 8];
      // tig >= 2 reads a neighbour's word; its B segment is zero.
      const unsigned int key0_tail = key_stage[4 * KeyStride + 2 * local0 + tig];
      const unsigned int key1_tail = key_stage[4 * KeyStride + 2 * (local0 + 8) + tig];
      const unsigned int a0 = (unsigned int)key0;
      const unsigned int a1 = (unsigned int)key1;
      const unsigned int a2 = (unsigned int)(key0 >> 32);
      const unsigned int a3 = (unsigned int)(key1 >> 32);

      // Horner over plane pairs, high to low: d = 4 d + popc(T_{2j}, T_{2j+1}).
      int d[4] = {0, 0, 0, 0};
      picokv_mma_b1_k256(d, a0, a1, a2, a3, score_b256[3][0], score_b256[3][1]);
      picokv_mma_b1_k128(d, key0_tail, key1_tail, score_b128[3]);
      d[1] = -d[1];  // plane 7 carries weight -128
      d[3] = -d[3];
      #pragma unroll
      for (int j = 2; j >= 0; j--) {
        #pragma unroll
        for (int r = 0; r < 4; r++) {
          d[r] *= 4;
        }
        picokv_mma_b1_k256(d, a0, a1, a2, a3, score_b256[j][0], score_b256[j][1]);
        picokv_mma_b1_k128(d, key0_tail, key1_tail, score_b128[j]);
      }
      // d[0]/d[2]: even planes, d[1]/d[3]: odd planes of head tig, tokens l0/l1.
      const int dot0 = query_sum[tig] - 2 * (d[0] + 2 * d[1]);
      const int dot1 = query_sum[tig] - 2 * (d[2] + 2 * d[3]);

      float out0 = 0.0f;
      float out1 = 0.0f;
      if constexpr (ExpBits <= 2) {
        out0 = -3.402823466e38f;
        out1 = -3.402823466e38f;
      }
      if (l0 < L_b) {
        const float score0 = (float)dot0 * scale_cache[tig];
        if constexpr (ExpBits <= 2) {
          out0 = score0;
        } else {
          out0 = picokv_fast_expf(score0 - lse_cache[tig]);
        }
      }
      if (l1 < L_b) {
        const float score1 = (float)dot1 * scale_cache[tig];
        if constexpr (ExpBits <= 2) {
          out1 = score1;
        } else {
          out1 = picokv_fast_expf(score1 - lse_cache[tig]);
        }
      }
      weights[tig][local0] = out0;
      weights[tig][local0 + 8] = out1;
    }
    if (thread_idx == 0) {
      live_bits = 0u;
    }
    __syncthreads();
    // key_stage is free: prefetch the next tile's keys under quantizer + value.
    if (block_tile + 1 < BlockTiles && (tile_idx + 1) * kTileSize < L_b) {
      stage_keys(tile_idx + 1);
    }
    picokv_cp_async_commit();
    // Group quantizer: verbatim from the popc kernel (PruneMode 0).
    #pragma unroll
    for (int quant_pass = 0; quant_pass < QuantPasses; quant_pass++) {
      const int quant_task = warp_idx + quant_pass * QuantWarps;
      const int hrel = quant_task / TileGroups;
      const int local_group = quant_task - hrel * TileGroups;
      const int local_base = local_group << 6;
      if (quant_task >= GroupSize * TileGroups) {
        continue;
      }

      const float w0 = weights[hrel][local_base + lane_idx];
      const float w1 = weights[hrel][local_base + lane_idx + 32];
      // == the popc kernel's fmaxf shuffle tree (max is a selection; weights
      // are never NaN and signed zeros cannot change any quantizer output).
      const float group_max = picokv_float_from_order_key(
        __reduce_max_sync(0xffffffff, picokv_float_order_key(fmaxf(w0, w1))));

      int q0 = 0;
      int q1 = 0;
      float scale = 0.0f;
      if constexpr (ExpBits <= 2) {
        const float max_score = group_max;
        const float max_weight = max_score > -3.0e38f ? picokv_fast_expf(max_score - lse_cache[hrel]) : 0.0f;
        if (max_weight > 0.0f) {
          const float d0 = w0 - max_score;
          const float d1 = w1 - max_score;
          if constexpr (ExpBits == 1) {
            constexpr float t0 = -0.6931471805599453f;
            q0 = d0 >= t0 ? 1 : 0;
            q1 = d1 >= t0 ? 1 : 0;
          } else {
            constexpr float t0 = -1.7917594692280550f;
            constexpr float t1 = -0.6931471805599453f;
            constexpr float t2 = -0.1823215567939546f;
            q0 = (d0 >= t0) + (d0 >= t1) + (d0 >= t2);
            q1 = (d1 >= t0) + (d1 >= t1) + (d1 >= t2);
          }
          scale = max_weight / (float)ExpMaxCode;
        }
      } else {
        const float max_weight = group_max;
        const float inv_scale = max_weight > 0.0f ? (float)ExpMaxCode / max_weight : 0.0f;
        if (max_weight > 0.0f) {
          q0 = (int)(w0 * inv_scale + 0.5f);
          q1 = (int)(w1 * inv_scale + 0.5f);
          q0 = q0 > ExpMaxCode ? ExpMaxCode : q0;
          q1 = q1 > ExpMaxCode ? ExpMaxCode : q1;
          scale = max_weight / (float)ExpMaxCode;
        }
      }

      const int qsum = __reduce_add_sync(0xffffffff, q0 + q1);

      #pragma unroll
      for (int bit_idx = 0; bit_idx < ExpBits; bit_idx++) {
        const unsigned int low = __ballot_sync(0xffffffff, ((q0 >> bit_idx) & 1) != 0);
        const unsigned int high = __ballot_sync(0xffffffff, ((q1 >> bit_idx) & 1) != 0);
        if (lane_idx == 0) {
          exp_planes[hrel][local_group][bit_idx][0] = low;
          exp_planes[hrel][local_group][bit_idx][1] = high;
        }
      }
      if (lane_idx == 0) {
        exp_scale2[hrel][local_group] = scale + scale;
        exp_bias[hrel][local_group] = -scale * (float)qsum;
        if (scale > 0.0f) {
          atomicOr(&live_bits, 1u << local_group);
        }
      }
    }
    picokv_cp_async_wait<1>();  // this tile's values (next keys may pend)
    __syncthreads();

    if (thread_idx < GroupSize) {
      #pragma unroll
      for (int local_group = 0; local_group < TileGroups; local_group++) {
        den_acc -= exp_bias[thread_idx][local_group];
      }
    }
    // Bit lg: some head has scale2 > 0 in group lg (implies lg < group_end).
    const unsigned int live_mask = live_bits;

    // Value: per group pair (2p, 2p+1) and m16 code tile. Lane (g, tig) owns
    // head tig, codes 16 mt + g and +8; accumulator columns 2tig / 2tig+1 are
    // groups 2p / 2p+1 of head tig, applied in group order like the popc loop.
    #pragma unroll
    for (int pair = 0; pair < TileGroups / 2; pair++) {
      const int lg_first = 2 * pair;
      const unsigned int pair_live = (live_mask >> lg_first) & 3u;
      if (pair_live == 0u) continue;  // block-uniform

      // B: column g = (head g>>1, group g&1); its two segments of the other
      // group stay zero.
      const int b_head = g >> 1;
      const int b_group = g & 1;
      unsigned int value_b[ExpBits];
      #pragma unroll
      for (int e = 0; e < ExpBits; e++) {
        value_b[e] = (tig >> 1) == b_group ? exp_planes[b_head][lg_first + b_group][e][tig & 1] : 0u;
      }
      // A head with scale2 0 skips the group, as in the popc kernel.
      float scale2[2];
      float bias[2];
      #pragma unroll
      for (int q = 0; q < 2; q++) {
        scale2[q] = exp_scale2[tig][lg_first + q];
        bias[q] = exp_bias[tig][lg_first + q];
      }
      // A: segment tig = half (tig&1) of group lg_first + (tig>>1)'s word.
      const unsigned int *a_stage = value_stage + (lg_first + (tig >> 1)) * ValueStride + 2 * g + (tig & 1);

      #pragma unroll
      for (int t = 0; t < WarpCodeTiles; t++) {
        const int mt = warp_idx + t * QuantWarps;
        if (mt < CodeTiles) {  // warp-uniform
          const unsigned int va0 = a_stage[mt * 32];
          const unsigned int va1 = a_stage[mt * 32 + 16];
          int d[4] = {0, 0, 0, 0};
          picokv_mma_b1_k128(d, va0, va1, value_b[ExpBits - 1]);
          #pragma unroll
          for (int e = ExpBits - 2; e >= 0; e--) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
              d[r] *= 2;
            }
            picokv_mma_b1_k128(d, va0, va1, value_b[e]);
          }
          #pragma unroll
          for (int q = 0; q < 2; q++) {
            const bool use = scale2[q] > 0.0f;
            const float sum0 = acc[t][0] + (scale2[q] * (float)d[q] + bias[q]);
            const float sum1 = acc[t][1] + (scale2[q] * (float)d[2 + q] + bias[q]);
            acc[t][0] = use ? sum0 : acc[t][0];
            acc[t][1] = use ? sum1 : acc[t][1];
          }
        }
      }
    }
    __syncthreads();
  }
  picokv_cp_async_wait<0>();

  if (thread_idx < GroupSize) {
    const int qh = qh_base + thread_idx;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx);
    atomicAdd(&atomic_den[base], den_acc);
  }

  {
    const int qh = qh_base + tig;
    const long base = ((long)(B_idx * QH + qh) * S + S_idx) * CvT;
    #pragma unroll
    for (int t = 0; t < WarpCodeTiles; t++) {
      const int mt = warp_idx + t * QuantWarps;
      if (mt < CodeTiles) {
        atomicAdd(&atomic_code_output[base + mt * 16 + g], acc[t][0]);
        atomicAdd(&atomic_code_output[base + mt * 16 + g + 8], acc[t][1]);
      }
    }
  }
}
