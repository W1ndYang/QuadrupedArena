"""Portable entry points. Offline commands never load credentials or call a model."""
import argparse,copy,json,shutil,subprocess,sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def new_directory(path):
    path=path.expanduser().resolve()
    if path.exists() and any(path.iterdir()):raise ValueError(f'Output must be empty: {path}')
    path.mkdir(parents=True,exist_ok=True)
    return path

def replay(output,numbers):
    from ds_fixed_goal import scenario,read,dump,digest,save_match
    from peek_braking_fix import SequenceProbe
    out=new_directory(output);cases=read(ROOT/'cases.json');records=[]
    dump(out/'cases.json',cases)
    for number in numbers:
        case=cases[number-1];source=ROOT/f'run_{number}';folder=out/f'run_{number}';folder.mkdir()
        scenario(case['layout']);cfg=read(source/'trial_config.json');seen=[]
        def saved(packet):
            index=packet['observation_id'];expected=read(source/f'api_calls/{index:03d}_request.json')['observation']
            if digest(packet)!=digest(expected):raise AssertionError(f'Public observation changed at run {number}, step {index}; stop replay')
            seen.append(index);return copy.deepcopy(read(source/f'accepted/{index:03d}.json')['action'])
        probe=SequenceProbe(cfg['base'],cfg['rules'],saved,folder)
        result=probe.run(f"saved_{case['layout']}_v041")
        assert result['states']==read(source/'timeline.json'),'Trajectory changed'
        assert result['events']==read(source/'events.json'),'Referee events changed'
        original=read(source/'result.json')
        for key in ('winner','reason','seconds','collision_stop_ticks','distance_m'):
            assert result['summary'][key]==original[key],key
        save_match(folder,result);dump(folder/'timeline.json',result['states']);dump(folder/'events.json',result['events'])
        dump(folder/'trial_config.json',cfg);dump(folder/'case.json',case)
        summary=dict(original,id=result['summary']['id'],completion_mode='saved_decision_replay',paid_calls_this_replay=0)
        dump(folder/'result.json',summary)
        records.append(dict(run=number,layout=case['layout'],public_packets_matched=len(seen),
            states_identical=True,events_identical=True,winner=summary['winner'],seconds=summary['seconds']))
        print('REPLAY_MATCHED',case['layout'],len(seen),'decisions',summary['seconds'],'seconds',flush=True)
    # Rendering the replay uses the same geometry and no new model call.
    shutil.copytree(ROOT/'source_snapshot',out/'source_snapshot',ignore=shutil.ignore_patterns('__pycache__'))
    dump(out/'replay_validation.json',dict(status='passed',network_calls=0,games=records))
    print('Output:',out)

def prepare(output):
    out=new_directory(output)
    for name in ('ds_fixed_goal.py','qwen_fixed_goal.py','sequence_skills.py','peek_geometry.py','peek_braking_fix.py',
                 'run_batch.py','run_goal_only.py','run_study.py','v04_model.py','prepare.py',
                 'render_native.py','build_report.py','report_style.css','audit_results.py',
                 'layouts.json','cases.json','waypoints.json','api_profile.json','design_basis.json'):
        shutil.copy2(ROOT/name,out/name)
    for name in ('source_snapshot','render_helpers'):
        shutil.copytree(ROOT/name,out/name,ignore=shutil.ignore_patterns('__pycache__'))
    subprocess.run([sys.executable,str(out/'prepare.py')],cwd=out,check=True)
    import hashlib
    files=[*out.glob('*.py'),*out.glob('prompts/*.txt'),*out.glob('source_snapshot/scripts/*.py'),*out.glob('render_helpers/*.py')]
    manifest={'files':{str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
    (out/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('Prepared without API calls:',out)
    print('Explicit paid command:',sys.executable,str(out/'run_batch.py'),'--execute-paid')

def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    rp=s.add_parser('replay',help='Re-execute archived decisions, checking every public input and state')
    rp.add_argument('--output',type=Path,required=True);rp.add_argument('--run',type=int,choices=(1,2,3))
    pp=s.add_parser('prepare',help='Prepare an empty directory for your own paid experiment')
    pp.add_argument('--output',type=Path,required=True)
    s.add_parser('audit',help='Audit shipped observations, rules, usage and source hashes')
    a=p.parse_args()
    if a.command=='replay':replay(a.output,[a.run] if a.run else [1,2,3])
    elif a.command=='prepare':prepare(a.output)
    else:subprocess.run([sys.executable,str(ROOT/'audit_results.py')],cwd=ROOT,check=True)

if __name__=='__main__':main()
