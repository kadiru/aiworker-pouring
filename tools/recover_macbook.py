#!/usr/bin/env python3
"""Read a scoped AIWorker snapshot over SSH; never executes recovered code."""
import base64
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "recovered/macbook-2026-10-28"
REMOTE = r'''
from pathlib import Path
import base64, hashlib, json, sqlite3, subprocess
base=Path.home()
root=base/'Documents/aiworker'
selected=set()
for name in ('AGENTS.md','CODEX_HANDOFF.md','README_IMPORT.md',
             'AIWORKER_QUEST_XR_RUNBOOK.md','FLUID_CUP_EXAMPLE_RUNBOOK.md',
             'ISAACSIM6_FLUID_RUNBOOK.md','BREV_L40S_ISAACSIM_REMOTE_RUNBOOK.md'):
    selected.add(root/name)
for directory in ('scripts/fluid','patches','ideation','tmp_pouring_assets/pouring'):
    selected.update(p for p in (root/directory).iterdir()
                    if p.is_file() and p.suffix in ('.py','.md','.json','.sh','.usda'))
for name in ('backup_scope.md','fluid_and_pouring_files.txt','rog_inventory.txt',
             'removed_output_dataset_dirs.txt','removed_model_weight_files.txt'):
    selected.add(root/'backups/rog-pairog2-20260630_180523/manifests'/name)
for rel in ('outputs/aiworker_training_analysis/current_training_analysis_20260622.md',
            'outputs/aiworker_training_analysis/aiworker_training_report_20260622_013736.md',
            'outputs/fluid_sandbox_ui_transfer/ROG_UI_TRANSFER_INSTRUCTIONS.md'):
    selected.add(root/rel)
files=[]
for p in ([] if HISTORY_ONLY else sorted(selected)):
    if p.is_file() and not p.is_symlink() and p.stat().st_size < 2000000:
        raw=p.read_bytes()
        files.append(dict(path=str(p.relative_to(root)),source=str(p),
                          bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),
                          mtime=p.stat().st_mtime,base64=base64.b64encode(raw).decode()))
env={'PATH':'/usr/bin:/bin:/opt/homebrew/bin','GIT_OPTIONAL_LOCKS':'0'}
git={}
for name,args in (() if HISTORY_ONLY else (('commit',['rev-parse','HEAD']),('status',['status','--short']),
                  ('recent_commits',['log','-10','--format=%H %ad %s','--date=iso-strict']))):
    r=subprocess.run(['git','-C',str(root)]+args,capture_output=True,text=True,env=env)
    git[name]=dict(returncode=r.returncode,output=r.stdout,stderr=r.stderr)
inventory=[] if HISTORY_ONLY else subprocess.run(['/opt/homebrew/bin/rg','--files','--hidden','-g','!.git',
                          '-g','!tmp','-g','!node_modules','-g','!.venv',str(root)],
                         capture_output=True,text=True).stdout.splitlines()
ids=('019f1a11-9363-7bd2-bae5-c1f39f836edc','019f1959-6099-7ad1-bd15-c018f13a7bc0',
     '019f0c59-1551-70f1-b342-15d6582c5183','019eedcf-fc48-77a0-a9f7-301c8e28b412',
     '019ea8a9-9cc5-7df0-b3e9-729d0053742b')
con=sqlite3.connect('file:'+str(base/'.codex/state_5.sqlite')+'?mode=ro',uri=True)
con.execute('PRAGMA query_only=ON')
threads=[]
for id in ids:
    row=con.execute('SELECT id,title,cwd,rollout_path,created_at,updated_at FROM threads WHERE id=?',(id,)).fetchone()
    if not row: continue
    t=dict(zip(('id','title','cwd','rollout_path','created_at','updated_at'),row))
    finals=[]
    p=Path(t['rollout_path'])
    compressed=Path(str(p)+'.zst')
    proc=None
    stream=None
    if p.is_file():
        stream=p.open()
    elif compressed.is_file():
        proc=subprocess.Popen(['/opt/homebrew/bin/zstd','-dc',str(compressed)],stdout=subprocess.PIPE,text=True)
        stream=proc.stdout
        t['compressed_rollout_path']=str(compressed)
    if stream is not None:
        for number,line in enumerate(stream,1):
            try: item=json.loads(line)
            except ValueError: continue
            payload=item.get('payload',{})
            if item.get('type')=='response_item' and payload.get('type')=='message' and payload.get('role')=='assistant' and (payload.get('channel')=='final' or payload.get('phase') in ('final','final_answer')):
                text='\n'.join(c.get('text','') for c in payload.get('content',[]) if c.get('type') in ('output_text','text'))
                if text:
                    finals.append(dict(line=number,timestamp=item.get('timestamp'),text=text))
        stream.close()
        if proc is not None and proc.wait()!=0:
            raise RuntimeError('Could not decompress '+str(compressed))
    t['final_count']=len(finals)
    # Selected public-facing responses only: no hidden reasoning or full chat dump.
    t['finals']=finals[:2]+finals[-12:] if len(finals)>14 else finals
    threads.append(t)
con.close()
print(json.dumps(dict(source_root=str(root),files=files,git=git,inventory=inventory,threads=threads)))
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history-only', action='store_true', help='Add history supplement, including compressed rollouts')
    args=parser.parse_args()
    if not args.history_only and DEST.exists():
        raise SystemExit("Snapshot already exists; preserve it rather than overwrite.")
    if args.history_only and (DEST / 'history.json').exists():
        raise SystemExit('History supplement already exists.')
    command = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8",
               "-o", "StrictHostKeyChecking=yes", "-o", "ForwardAgent=no",
               "-o", "UserKnownHostsFile=" + str(ROOT / "evidence/macbook-known-hosts"),
               "kadirumacbook@192.168.68.59", "python3 -"]
    result = subprocess.run(command, input='HISTORY_ONLY='+repr(args.history_only)+'\n'+REMOTE, text=True, capture_output=True, timeout=60)
    if result.returncode:
        raise SystemExit(result.stderr)
    payload = json.loads(result.stdout)
    DEST.mkdir(parents=True, exist_ok=args.history_only)
    if args.history_only:
        (DEST / 'history.json').write_text(json.dumps(payload['threads'],indent=2)+'\n')
        print(json.dumps([dict(id=t['id'],final_count=t['final_count'],saved=len(t['finals'])) for t in payload['threads']]))
        return
    metadata = []
    for entry in payload.pop("files"):
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("unsafe relative path")
        raw = base64.b64decode(entry.pop("base64"))
        if hashlib.sha256(raw).hexdigest() != entry["sha256"]:
            raise ValueError("hash mismatch: " + str(relative))
        # Preserve prior instructions as reference text, not active instructions.
        if relative.name == "AGENTS.md":
            relative = relative.with_name("ORIGINAL_AGENTS.txt")
        output = DEST / "source" / relative
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(raw)
        entry["local_path"] = str(output.relative_to(DEST))
        metadata.append(entry)
    payload["files"] = metadata
    (DEST / "manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(dict(snapshot=str(DEST),files=len(metadata),
                          bytes=sum(x["bytes"] for x in metadata),
                          thread_excerpts=len(payload["threads"]))))


if __name__ == "__main__":
    main()
