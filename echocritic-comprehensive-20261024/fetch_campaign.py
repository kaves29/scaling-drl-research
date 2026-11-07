"""Read-only W&B export. Raw files are published once, never overwritten."""
from pathlib import Path
import json,hashlib,concurrent.futures,time
import wandb
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'raw_api';RAW.mkdir(exist_ok=True)
api=wandb.Api(overrides={'base_url':'https://api.forge.coreweave.com'},timeout=90)
runs=list(api.runs('s_kaveti23-seckinger/echocritic_official_exp1',per_page=200))
print('records',len(runs),flush=True)
def save(p,obj):
 if not p.exists():p.write_text(json.dumps(obj,default=str,indent=2))
for r in runs:
 d=RAW/r.id;d.mkdir(exist_ok=True)
 save(d/'metadata.json',{'id':r.id,'name':r.name,'state':r.state,'config':r.config,'summary':dict(r.summary),'attrs':r._attrs,'url':r.url})
def fetch(r):
 d=RAW/r.id
 try:
  if not (d/'history.jsonl').exists():
   tmp=d/'history.partial'
   with tmp.open('w') as f:
    count=0
    for row in r.scan_history(page_size=1100000):f.write(json.dumps(row,default=str)+'\n');count+=1
   tmp.rename(d/'history.jsonl')
  fs=list(r.files(per_page=500));save(d/'files.json',[{'name':f.name,'size':f.size,'md5':f.md5} for f in fs])
  for f in fs:
   if (f.name in ('wandb-metadata.json','config.yaml','output.log','wandb-summary.json') or f.name.endswith(('.csv','.npz'))) and f.size<50000000:
    target=d/'files'/f.name
    if not target.exists():f.download(root=str(d/'files'),replace=False)
  return r.id,'ok'
 except Exception as e:
  save(d/'fetch_error.json',{'type':type(e).__name__,'message':str(e)[:500]});return r.id,type(e).__name__
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 for result in pool.map(fetch,runs):print(*result,flush=True)
manifest=[]
for p in RAW.rglob('*'):
 if p.is_file():manifest.append({'path':str(p.relative_to(RAW)),'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
save(ROOT/'raw_api_manifest.json',manifest)
print('EXPORT_COMPLETE',len(manifest),flush=True)
