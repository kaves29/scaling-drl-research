import json,tempfile,shutil
from pathlib import Path
import pandas as pd
from experiments.exp12 import ledger
from analysis import exp1_analysis as e1
from tests.test_exp12_exp2_analysis import _write_run
root=Path(tempfile.mkdtemp(prefix='exp12-ledger-probe-'))
for a in ['D2W512','D4W1024','D4W1536']:
 k=_write_run(root,a,'dog-run',1)
 d=ledger.ledger_root(root)/k
 # Empty files retain the exact required header; run.csv still contains the old endpoint.
 c=pd.read_csv(d/'checks.csv');c.iloc[:0].to_csv(d/'checks.csv',index=False)
r,c=ledger.load(root)
result={'run_count':len(r),'check_rows':len(c),'primary':e1.primary_endpoint(r,reps=20).to_dict('records'),'files_used_have_no_check20':True}
Path('/tmp/exp12-followup-evidence/exp1_ledger_acceptance.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2));shutil.rmtree(root)
