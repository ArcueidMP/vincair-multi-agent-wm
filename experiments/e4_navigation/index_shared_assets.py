"""Read-only artifact inventory; does not upload data or claim unverified downloads."""
import argparse,csv,hashlib,json
from pathlib import Path

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    if a.output.exists():raise RuntimeError('Preserve existing inventory')
    a.output.mkdir(parents=True);rows=[];groups=[]
    for domain in ['e0','e1','e4']:
        base=a.root/'results'/domain
        for run in sorted(p for p in base.iterdir() if p.is_dir()):
            declared={}
            for name in ['dataset_sha256.json','trajectory_sha256.json']:
                path=run/name
                if not path.exists():continue
                obj=json.loads(path.read_text())
                if isinstance(obj,dict):
                    declared.update({k:v for k,v in obj.items() if isinstance(v,str) and len(v)==64})
            files=[]
            for p in sorted(run.rglob('*')):
                if not p.is_file() or p.suffix not in ['.npz','.pt','.json','.csv','.md','.patch']:continue
                rel=p.relative_to(run).as_posix();size=p.stat().st_size
                # Hash small artifacts; retain declared hashes for large arrays without pretending a fresh check.
                actual=sha(p) if size<=1024*1024 else None;expected=declared.get(rel)
                if actual is not None and expected is not None:assert actual==expected,(p,expected,actual)
                row=dict(relative_path=p.relative_to(a.root).as_posix(),bytes=size,sha256=actual or expected,
                    checksum_status='freshly_hashed' if actual is not None else ('existing_manifest_only' if expected else 'not_hashed'),
                    storage='local_only',shared_download_url=None)
                rows.append(row);files.append(row)
            groups.append(dict(run=run.relative_to(a.root).as_posix(),files=len(files),bytes=sum(f['bytes'] for f in files),
                checksum_index_files=[n for n in ['dataset_sha256.json','trajectory_sha256.json'] if (run/n).exists()],
                consumption='Historical run-specific protocol/consumption markers govern reuse; never infer untouched test from directory name',storage='local_only'))
    weights=[];manifest=json.loads((a.root/'results/phase1/checkpoint_manifest.json').read_text())
    for row in manifest['files']:
        p=Path(row['local_path']);assert p.exists();digest=sha(p);assert digest==row['sha256']
        weights.append(dict(relative_path=p.relative_to(a.root).as_posix(),bytes=p.stat().st_size,sha256=digest,checksum_status='freshly_hashed',
            model_task='VMAS PushT' if 'vmas_pusht' in row['remote_path'] else 'MultiPushT',
            upstream_url=f"https://huggingface.co/{manifest['repository']}/resolve/{manifest['revision']}/{row['remote_path']}",
            storage='original upstream plus local copy; NOT uploaded by this inventory',navigation_compatible=False))
    for p in sorted((a.root/'data/checkpoints').glob('*/readout/cost_probe.pt')):
        weights.append(dict(relative_path=p.relative_to(a.root).as_posix(),bytes=p.stat().st_size,sha256=sha(p),checksum_status='freshly_hashed',
            model_task='MultiPushT readout; see original intake provenance',upstream_url=None,storage='local copy; source in readout intake report',navigation_compatible=False))
    doc=dict(schema='lewm_external_asset_index_v1',scan_scope='Scientific .npz/.pt/.json/.csv/.md/.patch in results/e0,e1,e4; excludes caches/private images and other directories',
        files=len(rows),bytes=sum(r['bytes'] for r in rows),freshly_hashed_files=sum(r['checksum_status']=='freshly_hashed' for r in rows),
        declared_only_files=sum(r['checksum_status']=='existing_manifest_only' for r in rows),unhashed_files=sum(r['checksum_status']=='not_hashed' for r in rows),
        run_groups=groups,author_weight_inventory=weights,navigation_checkpoint='Not provided; do not substitute any listed PushT weight',
        shared_storage='No new HF/dataset storage created. Team must choose destination and access; null download links are genuinely pending.',
        files_inventory='files.csv',privacy='No private chat images, credentials, HF cache or WSL disk included')
    (a.output/'ASSET_INDEX.json').write_text(json.dumps(doc,indent=2,ensure_ascii=False),encoding='utf-8')
    with (a.output/'files.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
    print(json.dumps({k:doc[k] for k in ['files','bytes','freshly_hashed_files','declared_only_files','unhashed_files']}))
if __name__=='__main__':main()
