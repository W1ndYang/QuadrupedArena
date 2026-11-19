"""Diagnose formal failures with one-Option counterfactuals on recorded prefixes."""
import argparse
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

from llm_bundle import verify_bundle
from llm_compare_ubuntu import verify_comparison
from llm_study_data import emit, read_jsonl
from llm_study_snapshots import replay_option
from online_env import ACTION_NAMES


def neighboring_skills(skill, legal, plan=None):
    controls = {}
    if skill.startswith('PEEK_'):
        controls['observation_direction'] = ('PEEK_NORTH' if skill.endswith('SOUTH')
                                             else 'PEEK_SOUTH')
    if skill in ('SOUTH_GATE','NORTH_GATE','SMOKE_SOUTH','SMOKE_NORTH'):
        south = 'SOUTH' in skill
        other, side = ('NORTH','SOUTH') if south else ('SOUTH','NORTH')
        smoked = skill.startswith('SMOKE_')
        controls['route_side'] = 'SMOKE_'+other if smoked else other+'_GATE'
        controls['smoke_use'] = side+'_GATE' if smoked else 'SMOKE_'+side
    if skill != 'WAIT':
        controls['one_option_delay'] = 'WAIT'
    elif 'NEXT_COVER_OR_GOAL' in legal:
        controls['continue_now'] = 'NEXT_COVER_OR_GOAL'
    elif plan:
        side = plan['side'].upper()
        controls['leave_immediately_same_plan'] = (
            'SMOKE_'+side if plan['smoke'] else side+'_GATE')
    return {name: value for name, value in controls.items() if value in legal}


def diagnose(data, row, events):
    summary = data.get('summary', {})
    last = next(iter(reversed(data.get('options', []))), None)
    if summary.get('reason') != 'tag' or last is None:
        return dict(formal_reason=summary.get('reason'),
                    policy_failure=row.get('failure_category'),
                    options=data.get('options_before_failure', []))
    event = next((event for event in events if event['kind']=='decision' and
                  abs(event['request']['time']-last['start']) < 1e-5 and
                  event['request']['stage']==last['stage']), None)
    report = dict(formal_reason='tag', terminal_option=last,
                  exposure=summary.get('referee_telemetry'))
    if event is None:
        report['counterfactual_status'] = 'No service decision at the final Option start'
        return report
    request = event['request']
    legal = [name for name, present in zip(ACTION_NAMES, request['public']['action_mask'])
             if present]
    variants = dict(recorded=last['action'], **neighboring_skills(
        last['action'], legal, event['response'].get('plan')))
    provenance = dict(source_request_id=request['request_id'],
                      seed_pair=[row['first_seed'],row['second_seed']])
    results = {}
    for name, skill in variants.items():
        try:
            results[name] = replay_option(events, provenance, skill)
        except (ValueError, RuntimeError) as error:
            results[name] = dict(skill=skill,error=str(error))
    original = results['recorded']
    reproduced = (original.get('winner') == 'defender' and
                  original.get('formal_reason') == 'tag' and
                  abs(original['end']-summary['seconds']) < 1e-5)
    report.update(reproduced_recorded_tag=reproduced, variants=results,
                  survived_selected_option=[
                      name for name, result in results.items()
                      if name!='recorded' and reproduced and 'error' not in result and
                      result.get('winner')!='defender'])
    return report


def study_failures(path, output, allow_fixture=False):
    verify_comparison(path)
    findings = []
    with tarfile.open(path) as outer, TemporaryDirectory() as directory:
        report = json.load(outer.extractfile('llm_comparison/comparison.json'))
        if report['fixture_only'] and not allow_fixture:
            raise ValueError('Fixture results are for engineering verification')
        for row in report['rows']:
            if row['status']=='completed' and row['summary']['winner']=='attacker':
                continue
            name = f"{row['index']:03d}-{row['method']}.tar.gz"
            content = outer.extractfile('llm_comparison/'+name).read()
            inner_path = Path(directory)/'episode.tar.gz'
            inner_path.write_bytes(content)
            verify_bundle(inner_path)
            with tarfile.open(fileobj=io.BytesIO(content),mode='r:gz') as inner:
                manifest = json.load(inner.extractfile('llm_run/run_manifest.json'))
                episode_path = f"llm_run/{manifest['episode_id']}.json"
                if episode_path not in inner.getnames():
                    findings.append(dict(method=row['method'],index=row['index'],
                                         status=row['status'],diagnosis='Episode log unavailable'))
                    continue
                episode = json.load(inner.extractfile(episode_path))
                log = 'llm_run/decision_events.jsonl'
                events = read_jsonl(inner.extractfile(log).read().decode()) if log in inner.getnames() else []
            findings.append(dict(method=row['method'],index=row['index'],status=row['status'],
                                 diagnosis=diagnose(episode,row,events)))
    output.parent.mkdir(parents=True,exist_ok=True)
    emit(output,dict(fixture_only=report['fixture_only'],phase=report['phase'],failures=findings,
                     scope='Counterfactuals end at the next Option boundary or formal terminal. '
                           'Surviving the changed Option leaves later outcomes open.'))
    print(json.dumps(dict(failures=len(findings),output=str(output))))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--fixture-for-tests',action='store_true')
    args=parser.parse_args()
    study_failures(args.archive,args.output,args.fixture_for_tests)
