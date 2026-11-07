"""Reproduce crash timing and configuration evidence directly from raw export."""
from pathlib import Path
import json,datetime
import pandas as pd
R=Path(__file__).resolve().parent;rows=[];configs=[]
for p in sorted((R/'raw_api').glob('*/metadata.json')):
 m=json.loads(p.read_text());c=m['config'];a=c['agent'];configs.append({'id':m['id'],'source':m['attrs']['commit'],'environment':c['env_name'],'seed':c['seed'],'actor_depth':a['actor_num_blocks'],'actor_width':a['actor_hidden_dim'],'critic_depth':a['critic_num_blocks'],'critic_width':a['critic_hidden_dim'],'critic_lr':a['critic_learning_rate'],'critic_decay':a['critic_weight_decay'],'actor_lr':a['actor_learning_rate'],'actor_decay':a['actor_weight_decay'],'gamma':a['gamma'],'tau':a['target_tau'],'single_critic':not a['critic_use_cdq'],'raw_budget':c['num_env_steps'],'interaction_budget':c['num_interaction_steps'],'probe_rounds':c['probe']['rounds'],'probe_checks':c['probe']['checks'],'fit_updates':c['probe']['steps']})
 if m['state']=='crashed':
  h=[json.loads(x) for x in p.with_name('history.jsonl').read_text().splitlines()];runtime=json.loads((p.parent/'files/wandb-metadata.json').read_text());t=h[-1]['_timestamp'];rows.append({'id':m['id'],'env':c['env_name'],'seed':c['seed'],'last_timestamp':t,'last_iso':datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat(),'runtime':h[-1].get('_runtime'),'start':runtime.get('startedAt'),'host':runtime.get('host')})
pd.DataFrame(rows).to_csv(R/'derived/crash_timing.csv',index=False);pd.DataFrame(configs).to_csv(R/'derived/configuration_audit.csv',index=False)
print('crash end spread seconds',max(x['last_timestamp'] for x in rows)-min(x['last_timestamp'] for x in rows))
