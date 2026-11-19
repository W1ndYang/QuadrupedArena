"""Verify and combine disjoint completed batches of one frozen holdout."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

from llm_compare_ubuntu import comparison_bundle, verify_comparison
from llm_study_data import emit
from llm_study_online import aggregate, batch_indices
from llm_study_protocol import digest


MATCHED_FIELDS = ('seed_plan', 'seed_plan_sha256', 'methods', 'direct_model',
                  'decision_runtime_version', 'decision_config_sha256',
                  'git_head', 'runner_sha256', 'duration_seconds', 'decision_clock')


def verify_batch(report, reference):
    if (report['phase']!='holdout' or report['fixture_only'] or
            report['status']!='completed' or not report.get('holdout_batch')):
        raise ValueError('Expected completed live holdout batches')
    if any(report[key]!=reference[key] for key in MATCHED_FIELDS):
        raise ValueError('Holdout batches use different frozen plans, runners, or methods')
    plan = report['seed_plan']
    if report['seed_plan_sha256']!=digest(plan):
        raise ValueError('Frozen plan hash mismatch')
    indices=batch_indices(len(plan['seed_pairs']),report['holdout_batch'])
    if indices!=report['seed_indices'] or report['seed_pairs']!=[
            plan['seed_pairs'][index] for index in indices]:
        raise ValueError('Batch seeds differ from the prescribed partition')
    methods=report['methods']
    expected={(i,method) for i in range(len(indices)) for method in methods}
    seen=set()
    for row in report['rows']:
        key=(row['index'],row['method'])
        if key not in expected or key in seen:
            raise ValueError('Unexpected or repeated paired episode')
        seen.add(key)
        index=key[0]
        if (row['seed_index']!=indices[index] or
                [row['first_seed'],row['second_seed']]!=report['seed_pairs'][index] or
                row['status'] not in ('completed','policy_failure','api_failure')):
            raise ValueError('Wrong seed or incomplete experiment row')
    if seen!=expected or report['row_count_planned']!=len(expected):
        raise ValueError('Batch lacks planned episode rows')
    return indices


def join(paths,output):
    reports=[]
    for path in paths:
        verify_comparison(path)
        with tarfile.open(path) as archive:
            report=json.load(archive.extractfile('llm_comparison/comparison.json'))
        reports.append((path,report))
    if not reports:
        raise ValueError('Provide at least one completed holdout batch')
    reference=reports[0][1]
    verify_batch(reference,reference)
    used=set()
    total=reference['holdout_batch'][1]
    for _,report in reports:
        indices=verify_batch(report,reference)
        if report['holdout_batch'][1]!=total or used.intersection(indices):
            raise ValueError('Batch indices overlap or use different partitions')
        used.update(indices)
    full_pairs=reference['seed_plan']['seed_pairs']
    if used!=set(range(len(full_pairs))) or len(reports)!=total:
        raise ValueError('Wait for all frozen holdout batches before combining')
    output=output.resolve()
    bundle=output.parent/(output.name+'.tar.gz')
    if bundle.exists():
        raise FileExistsError(bundle)
    output.mkdir(parents=True,exist_ok=False)
    rows=[]
    provenance=[]
    for path,report in sorted(reports,key=lambda item:item[1]['seed_indices'][0]):
        provenance.append(dict(name=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                               batch=report['holdout_batch'],
                               seed_count=len(report['seed_indices'])))
        with tarfile.open(path) as source:
            for row in report['rows']:
                local,global_index,method=row['index'],row['seed_index'],row['method']
                member=f'llm_comparison/{local:03d}-{method}.tar.gz'
                with source.extractfile(member) as stream, (
                        output/f'{global_index:03d}-{method}.tar.gz').open('xb') as dest:
                    shutil.copyfileobj(stream,dest)
                rows.append(dict(row,index=global_index))
    result=dict(reference,seed_pairs=full_pairs,seed_indices=list(range(len(full_pairs))),
                holdout_batch=None,source_batches=provenance,rows=rows,
                row_count_planned=len(rows),
                maximum_live_calls=sum(report['maximum_live_calls'] for _,report in reports),
                started_utc=min(report['started_utc'] for _,report in reports),
                finished_utc=max(report['finished_utc'] for _,report in reports),
                aggregate=aggregate(rows,reference['methods'],len(full_pairs)))
    emit(output/'comparison.json',result)
    comparison_bundle(output,bundle)
    print(json.dumps(dict(upload_this=str(bundle),rows=len(rows),seeds=len(full_pairs))),flush=True)
    return bundle


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,action='append',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    join(args.batch,args.output)
