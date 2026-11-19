"""Test one-change ablations of recorded direct-policy successes."""
import argparse
import io
import json
from pathlib import Path
import tarfile

from llm_compare_ubuntu import verify_comparison
from llm_option_policy import LLMOptionPolicy
from llm_run_episode import option_log
from llm_study_data import emit
from online_env import ACTION_NAMES
from option_env import run_episode


class FixedOptionSequence:
    def __init__(self, episode_id, stages):
        self.observer=LLMOptionPolicy(episode_id,None,48.)
        self.stages=stages
        self.offsets=[0,0]

    def observe(self,observation,mask):
        self.observer.observe(observation,mask)

    def __call__(self,observation,mask):
        stage=self.observer.stage
        offset=self.offsets[stage]
        if offset>=len(self.stages[stage]):
            raise ValueError('The fixed Option sequence ended before the referee')
        name=self.stages[stage][offset]
        action=ACTION_NAMES.index(name)
        if not mask[action]:
            raise ValueError(f'The fixed Option {name} is unavailable in the changed state')
        self.offsets[stage]+=1
        return action,0.,0.


def variants(stages):
    yield 'recorded_sequence',stages
    for stage,sequence in enumerate(stages):
        changes={}
        peeks=[i for i,skill in enumerate(sequence) if skill.startswith('PEEK_')]
        if len(peeks)>1:
            changes['remove_extra_peek']=sequence[:peeks[-1]]+sequence[peeks[-1]+1:]
        for i,skill in enumerate(sequence):
            if skill.endswith('_GATE') or skill.startswith('SMOKE_'):
                other=skill.replace('SOUTH','NORTH') if 'SOUTH' in skill else skill.replace('NORTH','SOUTH')
                changes['switch_entrance']=sequence[:i]+[other]+sequence[i+1:]
                if skill.startswith('SMOKE_'):
                    changes['omit_smoke']=sequence[:i]+[skill[len('SMOKE_'):]+'_GATE']+sequence[i+1:]
                break
        for name,change in changes.items():
            updated=list(stages)
            updated[stage]=change
            yield f'stage_{stage+1}_{name}',updated


def audit(path,output):
    verify_comparison(path)
    results=[]
    with tarfile.open(path) as outer:
        report=json.load(outer.extractfile('llm_comparison/comparison.json'))
        if report['fixture_only']:
            raise ValueError('Requires recorded live-policy results')
        for row in report['rows']:
            if (row['method']!='D2' or row['status']!='completed' or
                    row['summary']['winner']!='attacker'):
                continue
            data=outer.extractfile(f"llm_comparison/{row['index']:03d}-D2.tar.gz").read()
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as episode:
                manifest=json.load(episode.extractfile('llm_run/run_manifest.json'))
                existing=json.load(episode.extractfile(f"llm_run/{manifest['episode_id']}.json"))
            original=existing['options']
            stages=[[o['action'] for o in original if o['stage']==stage] for stage in (0,1)]
            for variant,choices in variants(stages):
                policy=FixedOptionSequence(manifest['episode_id'],choices)
                try:
                    outcome,records=run_episode(row['first_seed'],row['second_seed'],policy)
                    options=option_log(records)
                    if variant=='recorded_sequence' and (options!=original or
                            outcome['winner']!=row['summary']['winner']):
                        raise AssertionError('Original direct-policy trajectory failed to reproduce')
                    result=dict(index=row['index'],variant=variant,status='completed',
                                winner=outcome['winner'],seconds=outcome['seconds'],
                                reason=outcome['reason'], options=options)
                except ValueError as error:
                    if variant=='recorded_sequence':
                        raise
                    result=dict(index=row['index'],variant=variant,status='sequence_unavailable',
                                error=str(error))
                results.append(result)
                print(json.dumps({k:v for k,v in result.items() if k!='options'}),flush=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    emit(output,dict(rows=results,scope='Post-hoc one-variable changes to the recorded successful '
         'Option sequence. Future Options are kept fixed and cannot adapt to changed observations.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.archive,args.output)
