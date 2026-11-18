"""One independent simulation episode using a running HTTP decision service."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import secrets
import traceback

from llm_option_policy import DecisionServiceClient, LLMOptionPolicy
from option_env import run_episode
from online_env import ACTION_NAMES


def run(first_seed,second_seed,service_url,output_dir,duration=48.,episode_id=None):
    output_dir=Path(output_dir);output_dir.mkdir(parents=True,exist_ok=True)
    episode_id=episode_id or 'ep-'+secrets.token_hex(8)
    client=DecisionServiceClient(service_url)
    policy=LLMOptionPolicy(episode_id,client,duration)
    started=datetime.now(timezone.utc).isoformat()
    try:
        summary,records=run_episode(first_seed,second_seed,policy,duration)
        evaluation=policy.finish(summary)
        result=dict(status='completed',started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            episode_id=episode_id,first_seed=first_seed,second_seed=second_seed,
            summary=summary,service_evaluation=evaluation,
            own_trajectory=policy.own_trajectory,
            decisions=policy.responses,
            options=[dict(stage=row['stage'],action=ACTION_NAMES[row['action']],
                          start=row['time'],end=row['end_time'],
                          duration_seconds=row['elapsed_seconds'],
                          termination=row['termination_reason']) for row in records])
    except Exception as exc:
        result=dict(status='failed',started_utc=started,
            finished_utc=datetime.now(timezone.utc).isoformat(),
            episode_id=episode_id,first_seed=first_seed,second_seed=second_seed,
            error_type=type(exc).__name__,error=str(exc),
            traceback=traceback.format_exc(),decisions=policy.responses,
            own_trajectory=policy.own_trajectory)
    path=output_dir/f'{episode_id}.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status=result['status'],episode_id=episode_id,
                          output=str(path),summary=result.get('summary'),
                          error=result.get('error')),ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--service-url',default='http://127.0.0.1:8765')
    parser.add_argument('--first-seed',type=int,default=810000)
    parser.add_argument('--second-seed',type=int,default=820000)
    parser.add_argument('--duration',type=float,default=48.)
    parser.add_argument('--episode-id')
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    result=run(args.first_seed,args.second_seed,args.service_url,
               args.output_dir,args.duration,args.episode_id)
    if result['status']!='completed':raise SystemExit(1)


if __name__=='__main__':main()
