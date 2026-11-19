"""Isolate the D3 hard check by replaying identical recorded direct proposals."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import tarfile
from tempfile import TemporaryDirectory

from llm_bundle import verify_bundle
from llm_compare_ubuntu import verify_comparison
from llm_study_architecture import screen
from llm_study_data import emit, read_jsonl
from llm_study_protocol import parse_direct
from online_env import ACTION_NAMES


def read_case(outer,temporary,index,method):
    name=f'llm_comparison/{index:03d}-{method}.tar.gz'
    data=outer.extractfile(name).read()
    temporary.write_bytes(data)
    verify_bundle(temporary)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        config=json.load(archive.extractfile('llm_run/config_effective.json'))
        events=read_jsonl(archive.extractfile('llm_run/decision_events.jsonl').read().decode())
    if config['study_architecture']!=method:
        raise ValueError('Archived arm differs from its filename and comparison plan')
    return events


def switch_check(events,source,target):
    count=0
    for event in events:
        if event['kind']!='decision' or not event.get('llm_request'):
            continue
        request=event['request']
        actual=event['response']
        reasoning=actual['reasoning']
        decision=json.loads(event['llm_request'][-1]['content'])['decision']
        parsed=parse_direct(reasoning['raw_response']['text'],list(ACTION_NAMES[1:]))
        if parsed!=reasoning['direct'] or actual['action']!=ACTION_NAMES.index(
                reasoning['safety_check']['executed']):
            raise ValueError('The recorded raw response, parsed proposal or executed action differs')
        own_check=screen(parsed,request,decision,enabled=source=='D3')
        if own_check!=reasoning['safety_check']:
            raise ValueError('Recomputed source check differs from the archived response')
        alternate=screen(parsed,request,decision,enabled=target=='D3')
        count+=1
        if alternate['executed']!=own_check['executed']:
            return dict(first_difference=dict(time=request['time'],stage=request['stage'],
                        recorded_check=own_check,replacement_check=alternate),
                        source=source,target=target,checked_decisions=count)
    return dict(first_difference=None,source=source,target=target,checked_decisions=count)


def audit(path,output):
    verify_comparison(path)
    cases=[]
    with tarfile.open(path) as outer, TemporaryDirectory() as directory:
        report=json.load(outer.extractfile('llm_comparison/comparison.json'))
        if report['fixture_only'] or report['status']!='completed':
            raise ValueError('Expected a completed real experiment')
        summary={(row['index'],row['method']):row for row in report['rows']}
        temporary=Path(directory)/'episode.tar.gz'
        for index,_ in enumerate(report['seed_pairs']):
            logs={}
            first={}
            diagnostics=[]
            for source,target in (('D2','D3'),('D3','D2')):
                if summary[index,source]['status']!='completed':
                    raise ValueError('Direct replay needs a complete recorded model exchange')
                logs[source]=read_case(outer,temporary,index,source)
                first[source]=next(row for row in logs[source] if row.get('llm_request'))
                diagnostics.append(switch_check(logs[source],source,target))
            first_skills={method: row['response']['reasoning']['direct']['skill']
                          for method,row in first.items()}
            shared=first['D2']['llm_request']==first['D3']['llm_request']
            cases.append(dict(index=index,first_model_request_shared=shared,
                              first_proposed_skills=first_skills,switch_checks=diagnostics))
    counts={}
    for source in ('D2','D3'):
        rows=[v for case in cases for v in case['switch_checks'] if v['source']==source]
        counts[source]=dict(cases=len(rows),changed=sum(r['first_difference'] is not None for r in rows),
                            checked_decisions=sum(r['checked_decisions'] for r in rows))
    result=dict(archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                scope='Same-recorded-output decision replay; a changed decision ends the comparison.',
                shared_first_requests=sum(row['first_model_request_shared'] for row in cases),
                different_first_skill_with_shared_request=sum(
                    row['first_model_request_shared'] and len(set(row['first_proposed_skills'].values()))>1
                    for row in cases),counts=counts,cases=cases)
    output.parent.mkdir(parents=True,exist_ok=True)
    emit(output,result)
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    audit(args.archive,args.output)
