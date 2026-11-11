"""Check that measurement instrumentation preserves production cache tensors."""
from pathlib import Path
import sys,json
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from picokv.picokv_standalone import transformers as adapter
from rebuttal.latency.instrument import Recorder

torch.manual_seed(19)
results=[]
for length,ck,cv in [(513,128,192),(2048,320,320)]:
 k=torch.randn(1,2,length,128,device='cuda',dtype=torch.bfloat16)
 v=torch.randn_like(k)
 kwargs=dict(key_bits=ck,value_bits=cv,sink_size=64,local_size=256,eas=.01,zerorope=20,direction_sample_size=1024)
 ref=adapter.prepare_kv_cache(k,v,**kwargs)
 rec=Recorder().install()
 try:out=adapter.prepare_kv_cache(k,v,**kwargs)
 finally:rec.restore()
 trace=rec.summarize()
 for name in ['dense_key','dense_value','key_projection','value_projection','key_projection_t','packed_key','packed_value']:
  torch.testing.assert_close(getattr(ref,name),getattr(out,name),rtol=0,atol=0)
 categories=set(trace['stage_sum_across_streams_ms'])
 assert categories=={'complete_cache_prepare','projection_fitting','gpc_encoding_residual','packing','eas_selection','gather_compact_packed_cache'}
 rows=trace['intervals']
 assert {r['side'] for r in rows if r['category']=='projection_fitting'}=={'key','value'}
 assert len({r['stream'] for r in rows if r['category']=='projection_fitting'})==2
 results.append({'length':length,'k_bits':ck,'v_bits':cv,'bit_identical_all_cache_tensors':True,'categories':sorted(categories),'two_cuda_streams':True})
 print(results[-1],flush=True)
Path(__file__).with_name('validation.json').write_text(json.dumps({'checks':results,'status':'passed'},indent=2)+'\n')
