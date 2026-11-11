// Explicit m16n8k16 BF16 Tensor Core fragments. Ampere or later.
// A: row-major16x16 (four uint32 registers); B: row-major16x8
// (two uint32 registers); C: four FP32 registers per lane.
struct MmaAcc {
  float x[4];
  __device__ __forceinline__ void clear() {
    #pragma unroll
    for (int i=0;i<4;++i) x[i]=0;
  }
};
__device__ __forceinline__ void load_a(uint32_t (&a)[4], const bf16* base, int stride) {
  const int lane=threadIdx.x%32;
  const unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(base+(lane%16)*stride+(lane/16)*8));
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(a[0]),"=r"(a[1]),"=r"(a[2]),"=r"(a[3]):"r"(addr));
}
__device__ __forceinline__ void load_b(uint32_t (&b)[2], const bf16* base, int stride) {
  const int lane=threadIdx.x%32;
  const unsigned addr=static_cast<unsigned>(__cvta_generic_to_shared(base+(lane%16)*stride));
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];"
               :"=r"(b[0]),"=r"(b[1]):"r"(addr));
}
__device__ __forceinline__ void mma(MmaAcc& c,const uint32_t (&a)[4],const uint32_t (&b)[2]) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
               "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               :"+f"(c.x[0]),"+f"(c.x[1]),"+f"(c.x[2]),"+f"(c.x[3])
               :"r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}

__device__ __forceinline__ uint32_t code_pair(unsigned n0,unsigned n1) {
  return 0x3f803f80U|(n0<<15)|(n1<<31);
}
// Assemble MMA operands directly in registers from packed bits. The B fragment
// maps lane/4 to N and lane%4 to a K pair, with the second register at K+8.
__device__ __forceinline__ void load_code_b(uint32_t (&b)[2],const uint64_t* pk,const uint64_t* pv,
    int row,int start,int length,int bits,int cb,int n,int k,bool key) {
  const int lane=threadIdx.x%32,token=start+n+lane/4,c=cb+k+(lane%4)*2;
  #pragma unroll
  for(int i=0;i<2;++i) {
    unsigned n0=0,n1=0;
    if(token<length) {
      if(key) {
        const auto* ptr=reinterpret_cast<const uint32_t*>(pk+(int64_t(row)*(bits/64)+cb/64)*length+token);
        const uint32_t word=ptr[k/32];
        n0=(word>>(c%32+i*8))&1U;n1=(word>>(c%32+i*8+1))&1U;
      } else {
        const auto* ptr=pv+(int64_t(row)*((length+63)/64)+token/64)*bits+c+i*8;
        const unsigned shift=token%32;
        const auto* halves=reinterpret_cast<const uint32_t*>(ptr)+token%64/32;
        n0=1U^((halves[0]>>shift)&1U);n1=1U^((halves[2]>>shift)&1U);
      }
    }
    b[i]=token<length?code_pair(n0,n1):0;
  }
}

template<int BT,int Warps,int CodeBits=0,bool PackedDecoder=false>
__device__ __forceinline__ void reconstruct_tile_ptx(
    const uint64_t* pk,const uint64_t* pv,const bf16* dt,bf16* stage,bf16* kv,
    int row,int start,int length,int runtime_bits,bool key) {
  const int bits=CodeBits ? CodeBits : runtime_bits;
  constexpr int TS=BT+8, Threads=Warps*32;
  constexpr int N=128/16*(BT/8), PerWarp=(N+Warps-1)/Warps;
  bf16* dec=stage;
  const int warp=threadIdx.x/32,lane=threadIdx.x%32;
  MmaAcc acc[PerWarp];
  #pragma unroll
  for(int j=0;j<PerWarp;++j)acc[j].clear();
  for(int cb=0;cb<bits;cb+=64) {
    if constexpr(!PackedDecoder) {
    for(int i=threadIdx.x;i<128*8;i+=Threads) {
      const int d=i/8,c=(i%8)*8;
      *reinterpret_cast<uint4*>(dec+d*72+c)=
          *reinterpret_cast<const uint4*>(dt+(int64_t(row)*128+d)*bits+cb+c);
    }
    __syncthreads();
    }
    #pragma unroll
    for(int k=0;k<64;k+=16) {
      constexpr int BFrags=(BT/8+Warps-1)/Warps;
      uint32_t b[BFrags][2];
      #pragma unroll
      for(int f=0;f<BFrags;++f) {
        const int n=((warp+f*Warps)%(BT/8))*8;
        load_code_b(b[f],pk,pv,row,start,length,bits,cb,n,k,key);
      }
      #pragma unroll
      for(int j=0;j<PerWarp;++j) {
        const int task=warp+j*Warps,m=(task/(BT/8))*16;
        if(task<N) {
          uint32_t a[4];
          if constexpr(PackedDecoder) {
            const uint4 v=*reinterpret_cast<const uint4*>(dt+(((int64_t(row)*(bits/16)+(cb+k)/16)*8+m/16)*32+lane)*8);
            a[0]=v.x;a[1]=v.y;a[2]=v.z;a[3]=v.w;
          } else load_a(a,dec+m*72+k,72);
          mma(acc[j],a,b[j%BFrags]);
        }
      }
    }
    if constexpr(!PackedDecoder) __syncthreads();
  }
  #pragma unroll
  for(int j=0;j<PerWarp;++j) {
    const int task=warp+j*Warps,m=(task/(BT/8))*16,n=(task%(BT/8))*8;
    if(task<N) {
      #pragma unroll
      for(int r=0;r<4;++r) {
        const int d=m+lane/4+(r/2)*8,t=n+(lane%4)*2+r%2;
        const bf16 val=__float2bfloat16_rn(acc[j].x[r]);
        // K is consumed as [128,BT]; V as [BT,128]. No FP32 smem epilogue.
        if(key)kv[d*TS+t]=val;
        else kv[t*136+d]=val;
      }
    }
  }
  __syncthreads();
}

template<int BT,int Warps,int BlockTiles,int CodeBits=0,bool PackedDecoder=false>
__global__ void reconstructed_bf16_attention_kernel_ptx(
    const bf16* __restrict__ q,const uint64_t* __restrict__ pk,const uint64_t* __restrict__ pv,
    const bf16* __restrict__ dkt,const bf16* __restrict__ dvt,const float* __restrict__ lse,
    float* __restrict__ output,float* __restrict__ den,int length,int runtime_ck,int runtime_cv,float scale) {
  const int ck=CodeBits ? CodeBits : runtime_ck,cv=CodeBits ? CodeBits : runtime_cv;
  constexpr int Threads=Warps*32,TS=BT+8;
  constexpr int Stage=(PackedDecoder ? 16*136 : 128*72)*2;
  constexpr int KVSize=128*TS>BT*136?128*TS:BT*136;
  extern __shared__ __align__(32) unsigned char storage[];
  auto* stage=reinterpret_cast<bf16*>(storage);
  auto* kv=reinterpret_cast<bf16*>(storage+Stage);
  auto* weight=kv+KVSize;
  auto* scores=reinterpret_cast<float*>(stage);
  const int row=blockIdx.y,warp=threadIdx.x/32,lane=threadIdx.x%32;
  for(int tile=0;tile<BlockTiles;++tile) {
    const int start=(blockIdx.x*BlockTiles+tile)*BT;
    reconstruct_tile_ptx<BT,Warps,CodeBits,PackedDecoder>(pk,pv,dkt,stage,kv,row,start,length,ck,true);
    for(int i=threadIdx.x;i<16*128;i+=Threads) {
      const int h=i/128,d=i%128;
      stage[h*136+d]=h<4?q[(row*4+h)*128+d]:__float2bfloat16_rn(0.0f);
    }
    __syncthreads();
    constexpr int ScoreTiles=(BT/8+Warps-1)/Warps;
    MmaAcc acc[ScoreTiles];
    #pragma unroll
    for(int j=0;j<ScoreTiles;++j) {
      acc[j].clear();
      const int n=(warp+j*Warps)*8;
      if(n<BT) {
        #pragma unroll
        for(int k=0;k<128;k+=16) {
          uint32_t a[4],b[2];load_a(a,stage+k,136);load_b(b,kv+k*TS+n,TS);mma(acc[j],a,b);
        }
      }
    }
    __syncthreads();
    #pragma unroll
    for(int j=0;j<ScoreTiles;++j) {
      const int n=(warp+j*Warps)*8;
      if(n<BT && lane/4<4) {
        scores[(lane/4)*BT+n+(lane%4)*2]=acc[j].x[0];
        scores[(lane/4)*BT+n+(lane%4)*2+1]=acc[j].x[1];
      }
    }
    __syncthreads();
    if(warp<4) {
      float sum=0;
      for(int t=lane;t<BT;t+=32) {
        const float w=start+t<length?expf(scores[warp*BT+t]*scale-lse[row*4+warp]):0;
        weight[warp*TS+t]=__float2bfloat16_rn(w);sum+=w;
      }
      sum=warp_sum(sum);if(!lane)atomicAdd(den+row*4+warp,sum);
    }
    for(int i=threadIdx.x;i<12*BT;i+=Threads)weight[(i/BT+4)*TS+i%BT]=__float2bfloat16_rn(0.0f);
    __syncthreads();
    reconstruct_tile_ptx<BT,Warps,CodeBits,PackedDecoder>(pk,pv,dvt,stage,kv,row,start,length,cv,false);
    #pragma unroll
    for(int j=0;j<(16+Warps-1)/Warps;++j) {
      const int n=(warp+j*Warps)*8;
      if(n<128) {
        MmaAcc out;out.clear();
        #pragma unroll
        for(int k=0;k<BT;k+=16) {
          uint32_t a[4],b[2];load_a(a,weight+k,TS);load_b(b,kv+k*136+n,136);mma(out,a,b);
        }
        if(lane/4<4) {
          atomicAdd(output+(row*4+lane/4)*128+n+(lane%4)*2,out.x[0]);
          atomicAdd(output+(row*4+lane/4)*128+n+(lane%4)*2+1,out.x[1]);
        }
      }
    }
    __syncthreads();
  }
}

template<int BT,int Warps,int BlockTiles,int CodeBits=0>
__global__ void unpacked_bf16_attention_kernel_ptx(
    const bf16* __restrict__ qp,const uint64_t* __restrict__ pk,const uint64_t* __restrict__ pv,
    const float* __restrict__ lse,float* __restrict__ output,float* __restrict__ den,
    int length,int runtime_ck,int runtime_cv,float scale) {
  const int ck=CodeBits ? CodeBits : runtime_ck,cv=CodeBits ? CodeBits : runtime_cv;
  constexpr int Threads=Warps*32,TS=BT+8;
  __shared__ __align__(32) bf16 weight[16*TS];
  __shared__ __align__(32) float scores[4*BT];
  const int row=blockIdx.y,warp=threadIdx.x/32,lane=threadIdx.x%32;
  constexpr int ScoreTiles=(BT/8+Warps-1)/Warps;
  for(int tile=0;tile<BlockTiles;++tile) {
    const int start=(blockIdx.x*BlockTiles+tile)*BT;
    MmaAcc acc[ScoreTiles];
    #pragma unroll
    for(int j=0;j<ScoreTiles;++j)acc[j].clear();
    for(int cb=0;cb<ck;cb+=64) {
      #pragma unroll
      for(int j=0;j<ScoreTiles;++j) {
        const int n=(warp+j*Warps)*8;
        if(n<BT) {
          #pragma unroll
          for(int k=0;k<64;k+=16) {
            uint32_t a[4],b[2];
            const int h=lane/4,c=cb+k+(lane%4)*2;
            a[0]=h<4?*reinterpret_cast<const uint32_t*>(qp+(row*4+h)*ck+c):0;
            a[1]=0;
            a[2]=h<4?*reinterpret_cast<const uint32_t*>(qp+(row*4+h)*ck+c+8):0;
            a[3]=0;
            load_code_b(b,pk,pv,row,start,length,ck,cb,n,k,true);mma(acc[j],a,b);
          }
        }
      }
    }
    #pragma unroll
    for(int j=0;j<ScoreTiles;++j) {
      const int n=(warp+j*Warps)*8;
      if(n<BT && lane/4<4) {
        scores[(lane/4)*BT+n+(lane%4)*2]=acc[j].x[0];
        scores[(lane/4)*BT+n+(lane%4)*2+1]=acc[j].x[1];
      }
    }
    __syncthreads();
    if(warp<4) {
      float sum=0;
      for(int t=lane;t<BT;t+=32) {
        const float s=__bfloat162float(__float2bfloat16_rn(scores[warp*BT+t]))*scale;
        const bf16 w=__float2bfloat16_rn(start+t<length?expf(s-lse[row*4+warp]):0.0f);
        weight[warp*TS+t]=w;sum+=__bfloat162float(w);
      }
      sum=warp_sum(sum);if(!lane)atomicAdd(den+row*4+warp,sum);
    }
    for(int i=threadIdx.x;i<12*BT;i+=Threads)weight[(i/BT+4)*TS+i%BT]=__float2bfloat16_rn(0.0f);
    __syncthreads();
    for(int cb=0;cb<cv;cb+=64) {
      #pragma unroll
      for(int j=0;j<(8+Warps-1)/Warps;++j) {
        const int n=(warp+j*Warps)*8;
        if(n<64) {
          MmaAcc out;out.clear();
          #pragma unroll
          for(int k=0;k<BT;k+=16) {
            uint32_t a[4],b[2];load_a(a,weight+k,TS);
            // B is [token,code]. A packed V word supplies all token pairs
            // for this code; decode directly into the MMA B registers.
            const int token=start+k+(lane%4)*2,c=cb+n+lane/4;
            const auto* ptr=reinterpret_cast<const uint32_t*>(pv+(int64_t(row)*((length+63)/64)+token/64)*cv+c);
            const uint32_t word=token<length?ptr[token%64/32]:0;
            #pragma unroll
            for(int r=0;r<2;++r) {
              const int t=token+r*8;
              const unsigned bit=t%32;
              const unsigned n0=1U^((word>>bit)&1U),n1=1U^((word>>(bit+1))&1U);
              b[r]=code_pair(n0,n1);
              if(t>=length)b[r]=0;
              else if(t+1>=length)b[r]&=0x0000ffffU;
            }
            mma(out,a,b);
          }
          if(lane/4<4 && cb+n+(lane%4)*2+1<cv) {
            atomicAdd(output+(row*4+lane/4)*cv+cb+n+(lane%4)*2,out.x[0]);
            atomicAdd(output+(row*4+lane/4)*cv+cb+n+(lane%4)*2+1,out.x[1]);
          }
        }
      }
    }
    __syncthreads();
  }
}
