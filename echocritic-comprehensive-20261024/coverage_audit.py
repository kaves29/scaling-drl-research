"""Audit actual local export coverage; no campaign membership inferred from names."""
from pathlib import Path
import json,csv,hashlib
ROOT=Path(__file__).resolve().parent
SOURCE=Path('/Users/shouryakaveti/echocritic_wandb_export/full_export')
rows=[];files=[]
for p in sorted(SOURCE.glob('*/metadata.json')):
    m=json.loads(p.read_text());c=m['config'];h=p.with_name('history.jsonl')
    data=[json.loads(x) for x in h.read_text().splitlines() if x.strip()]
    for f in (p,h):files.append({'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
    agent=c['agent'];probe=c['probe']
    rows.append({'id':m['id'],'name':m['name'],'state':m['state'],'created_at':m['created_at'],
      'configured_project':c['project_name'],'environment':c['env_name'],'seed':c['seed'],
      'architecture':f"D{agent['critic_num_blocks']}W{agent['critic_hidden_dim']}",
      'probe_rounds':probe['rounds'],'probe_checks':probe['checks'],'fitting_updates':probe['steps'],
      'raw_budget':c['num_env_steps'],'interaction_budget':c['num_interaction_steps'],
      'history_rows':len(data),'last_wandb_step':data[-1].get('_step'),
      'eval_observations':sum('eval/avg_return' in r for r in data),
      'last_eval_wandb_step':next((r.get('_step') for r in reversed(data) if 'eval/avg_return' in r),None),
      'critic_lr':agent['critic_learning_rate'],'critic_decay':agent['critic_weight_decay'],
      'training_source_verified':False,'canonical_membership_verified':False})
with (ROOT/'available_records.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
(ROOT/'input_hashes.json').write_text(json.dumps(files,indent=2))
(ROOT/'coverage.json').write_text(json.dumps({'available_records':len(rows),'expected_owner_reported_records':103,
    'missing_relative_to_reported_count':103-len(rows),'states':{s:sum(r['state']==s for r in rows) for s in sorted({r['state'] for r in rows})},
    'configured_projects':{s:sum(r['configured_project']==s for r in rows) for s in sorted({r['configured_project'] for r in rows})},
    'status':'INCOMPLETE_INPUT_CANNOT_EXECUTE_FULL_CAMPAIGN_INFERENCE'},indent=2))
print((ROOT/'coverage.json').read_text())
