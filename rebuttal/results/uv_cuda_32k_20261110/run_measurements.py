from pathlib import Path
import subprocess,sys,json,datetime,os,hashlib,importlib.metadata,platform
import torch
assert Path(sys.prefix).resolve()==Path("/home/zxh/zxh/PiCoKV/.venv").resolve(), sys.prefix
root=Path('/data/zxh/PiCoKV2');out=root/'rebuttal/results/uv_cuda_32k_20261110'
mode=sys.argv[1]
order=[('dense','cuda'),('packed','cuda'),('unpacked_fused','cuda'),('reconstruct_cuda','cuda')]
source_files=['rebuttal/cuda/attention.cu','rebuttal/cuda/attention_ptx.cuh','rebuttal/cuda/attention.cpp','rebuttal/cuda_attention.py','rebuttal/reconstruct_cuda_attention.py','rebuttal/unpacked_kernel_attention.py','rebuttal/reconstruct_cuda_attention_triton.py','rebuttal/unpacked_kernel_attention_triton.py','rebuttal/reconstruct_attention.py','rebuttal/measure_attention.py','rebuttal/run_efficiency.py','benchmarks/Efficiency/run_efficiency.py','picokv/picokv_standalone/decode.py','picokv/picokv_cuda_extension/cuda_kernel.cu']
config={'status':'running','started_at':datetime.datetime.now().astimezone().isoformat(),'mode':mode,'physical_gpu':int(os.environ['CUDA_VISIBLE_DEVICES']),'gpu_model':'RTX3090 24GB','run_mode':'sequential on same physical GPU; fresh process/model for each method','model':'/home/zxh/zxh/models/Llama-3.1-8B-Instruct','prompt_len':32768,'batch_size':1,'decode_tokens':64,'warmup':2,'repeats':5,'profile_tokens':8,'seed':42,'k_bits':320,'v_bits':320,'sink_size':64,'local_size':256,'exp_bits':4,'eas':.01,'zerorope':20,'pruning':False,'source_sha256':{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in source_files},'environment':{k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','PYTORCH_CUDA_ALLOC_CONF','TORCH_EXTENSIONS_DIR','TRITON_CACHE_DIR','TORCH_CUDA_ARCH_LIST','MAX_JOBS','OMP_NUM_THREADS','TOKENIZERS_PARALLELISM']},'processes':{}}
config['python']={'executable':sys.executable,'prefix':sys.prefix,'version':sys.version}
config['packages']={p:importlib.metadata.version(p) for p in ['torch','triton','transformers','flash-attn','accelerate','numpy']}
config['runtime']={'gpu_name':torch.cuda.get_device_name(),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'platform':platform.platform()}
config['environment'].update({k:os.environ.get(k) for k in ['VIRTUAL_ENV','PICOKV_DISABLE_PREBUILT','UV_CACHE_DIR','PYTHONPATH']})
config['uv_project_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path('/home/zxh/zxh/PiCoKV/pyproject.toml'),Path('/home/zxh/zxh/PiCoKV/uv.lock')]}
def save(): (out/f'{mode}_run_config.json').write_text(json.dumps(config,indent=2)+'\n')
save()
for backend,implementation in order:
 tag=backend+('_triton' if implementation=='triton' else '')
 output=f'rebuttal/results/uv_cuda_32k_20261110/{mode}_{tag}.json'
 if mode=='attention':
  command=[sys.executable,'-u','rebuttal/measure_attention.py','--backend',backend,'--control-implementation',implementation,'--length','32768','--decode-tokens','64','--repeats','5','--profile-tokens','8','--output',output]
 else:
  common=['--model',config['model'],'--lengths','32768','--decode-tokens','64','--batch-size','1','--warmup','2','--repeats','5','--dataset','benchmarks/Efficiency/datasets/PaulGrahamEssays.jsonl','--output',output]
  if backend=='dense':command=[sys.executable,'-u','benchmarks/Efficiency/run_efficiency.py','--method','flash_attention_2',*common]
  else:
   args=['--method','picokv','--k-bits','320','--v-bits','320','--sink-size','64','--local-size','256','--exp-bits','4','--eas','.01','--zerorope','20','--no-pruning','--decode-backend',backend,*common]
   if implementation=='triton':
    module='reconstruct_cuda_attention' if backend=='reconstruct_cuda' else 'unpacked_kernel_attention'
    fn='decode_attention' if backend=='reconstruct_cuda' else 'decode_attention_fused'
    code=f'from rebuttal import {module} as current,{module}_triton as reference; current.{fn}=reference.{fn}; from rebuttal.run_efficiency import main;main()'
    command=[sys.executable,'-u','-c',code,*args]
   else:command=[sys.executable,'-u','rebuttal/run_efficiency.py',*args]
 print('starting',mode,tag,flush=True)
 config['processes'][tag]={'command':command,'started_at':datetime.datetime.now().astimezone().isoformat()}
 with (out/f'{mode}_{tag}.log').open('w') as log:
  p=subprocess.Popen(command,cwd=root,stdout=log,stderr=subprocess.STDOUT)
  config['processes'][tag]['pid']=p.pid;save();return_code=p.wait()
 config['processes'][tag].update(exit_code=return_code,finished_at=datetime.datetime.now().astimezone().isoformat());save()
 if return_code:config['status']='failed';save();raise SystemExit(return_code)
 r=json.loads((root/output).read_text())
 print('finished',mode,tag,r['kernel_profile']['attention']['mean_kernel_ms_per_token_all_layers'] if mode=='attention' else r['results']['32768'],flush=True)
config['sources_unchanged_during_measurement']=all(hashlib.sha256((root/f).read_bytes()).hexdigest()==sha for f,sha in config['source_sha256'].items())
assert config['sources_unchanged_during_measurement']
config.update(status='completed',finished_at=datetime.datetime.now().astimezone().isoformat());save()
