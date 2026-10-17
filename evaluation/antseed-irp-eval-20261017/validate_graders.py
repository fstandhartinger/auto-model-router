"""Self-test of the graders: every coding task must pass with its dataset reference solution and fail a stub; every other
task must pass an answer built from its key and fail a wrong one. Needs the HF cache from build_tasks.py."""
import json,sys,glob
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
import grade, pandas as pd
T=[json.loads(l) for l in open(Path(__file__).resolve().parent/'tasks-public.jsonl')]
he=pd.concat([pd.read_parquet(f) for f in glob.glob('/home/flori/.cache/huggingface/hub/datasets--evalplus--humanevalplus/snapshots/*/data/*.parquet')]).set_index('task_id')
mb=pd.concat([pd.read_parquet(f) for f in glob.glob('/home/flori/.cache/huggingface/hub/datasets--evalplus--mbppplus/snapshots/*/data/*.parquet')]).set_index('task_id')
bad=0
for t in T:
    if t['category']=='coding':
        if t['source'].startswith('HumanEval'):
            r=he.loc[t['source'].split()[1]]; code=r.prompt+r.canonical_solution
        else:
            r=mb.loc[int(t['source'].split()[1])]; code=r.code
        ok,d=grade.grade(t,"```python\n"+code+"\n```"); ok2,d2=grade.grade(t,"```python\ndef nothing(): pass\n```")
        print(t['id'],ok,ok2, d if not ok else '')
        bad+= (not ok) or ok2
    else:
        e=t['expected']
        if t['grader']=='json': ans="```json\n"+json.dumps(e)+"\n```"
        elif t['grader']=='letter': ans="blah\nAnswer: "+e
        else: ans="x\nFinal answer: **"+str(e)+"**"
        ok,d=grade.grade(t,ans)
        neg={'json':'{"a":1}','letter':'Answer: Z','number':'Final answer: 123456789','string':'Final answer: nope'}[t['grader']]
        ok2,_=grade.grade(t,neg)
        if not ok or ok2: print('BAD',t['id'],ok,ok2,d); bad+=1
print('bad',bad)
