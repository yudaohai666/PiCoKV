import sys,json,statistics
from pathlib import Path
import torch
import torch.nn.functional as F
from picokv.picokv_standalone.cache import PreparedKVCache

def _make_cache(length, k_bits, v_bits, device):
    generator = torch.Generator().manual_seed(23)
    key_codes = (torch.randint(2, (2, 2, length, k_bits), generator=generator) * 2 - 1).to(torch.bfloat16)
    value_codes = (torch.randint(2, (2, 2, length, v_bits), generator=generator) * 2 - 1).to(torch.bfloat16)
    weights = torch.ones(64, dtype=torch.int64) << torch.arange(64)
    packed_key = ((key_codes < 0).reshape(2, 2, length, k_bits // 64, 64).long() * weights).sum(-1).transpose(-1, -2).contiguous()
    value_binary = F.pad((value_codes > 0).long(), (0, 0, 0, -length % 64))
    packed_value = (value_binary.reshape(2, 2, -1, 64, v_bits) * weights[:, None]).sum(-2).contiguous()
    key_decoder = (torch.randn(2, 2, k_bits, 128, generator=generator) / k_bits ** 0.5).to(device=device, dtype=torch.bfloat16)
    value_decoder = (torch.randn(2, 2, v_bits, 128, generator=generator) / v_bits ** 0.5).to(device=device, dtype=torch.bfloat16)
    cache = PreparedKVCache(dense_key=torch.randn(2, 2, 5, 128, generator=generator).to(device=device, dtype=torch.bfloat16), dense_value=torch.randn(2, 2, 5, 128, generator=generator).to(device=device, dtype=torch.bfloat16), key_projection=key_decoder, value_projection=value_decoder, key_projection_t=key_decoder.transpose(-1, -2).contiguous(), packed_key=packed_key.to(device), packed_value=packed_value.to(device), exp_bits=4)
    return (cache, key_codes.to(device), value_codes.to(device))

from rebuttal import reconstruct_cuda_attention as rc,unpacked_kernel_attention as uc
from rebuttal import reconstruct_cuda_attention_triton as rt,unpacked_kernel_attention_triton as ut
rows=[]
for length,kbits,vbits in [(0,128,192),(1,128,192),(31,256,320),(32,320,256),(33,320,320),(63,320,320),(64,320,320),(65,320,320),(127,320,320),(128,320,320),(129,320,320),(511,512,128),(512,320,320),(513,128,512),(32768,320,320)]:
 with torch.inference_mode():
  torch.manual_seed(71)
  cache,_,_=_make_cache(max(length,1),kbits,vbits,'cuda')
  if not length:
   cache.packed_key=cache.packed_key[...,:0].contiguous();cache.packed_value=cache.packed_value[:,:,:0].contiguous()
  q=torch.randn(2,8,1,128,device='cuda',dtype=torch.bfloat16)
  for reconstruct,new,old in [(False,uc.decode_attention_fused,ut.decode_attention_fused),(True,rc.decode_attention,rt.decode_attention)]:
   reference=old(q,cache,attention_scale=.07).clone();out=new(q,cache,attention_scale=.07).clone()
   assert torch.isfinite(out).all()
   torch.testing.assert_close(out,reference,atol=.008,rtol=.02)
   error=(out.float()-reference.float())
   row={'length':length,'k_bits':kbits,'v_bits':vbits,'reconstruct':reconstruct,'relative_rmse':(error.norm()/reference.float().norm().clamp_min(1e-12)).item(),'max_abs':error.abs().max().item()}
   rows.append(row);print(row,flush=True)
Path('rebuttal/results/uv_cuda_32k_20261110/cuda_vs_triton_correctness.json').write_text(json.dumps({'cases':len(rows),'results':rows},indent=2)+'\n')
print('30 CUDA/Triton boundary checks passed',flush=True)
