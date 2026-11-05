"""Independent-seed evaluation for learned and fixed Skill policies."""
import argparse
import datetime
import json
from pathlib import Path
import statistics
import time

import numpy as np

from arena_paths import workspace_root
from rl_skill_env import ACTION_NAMES,SkillEnv,load_rl_config
from rl_skill_ppo import PPOAgent
from semantic_skills import VisualMemory


PROJECT=Path(__file__).resolve().parents[1]


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def metrics(episodes):
    scores=[x['score'] for x in episodes]
    return dict(episodes=len(episodes),mean_score=float(np.mean(scores)),score_std=float(np.std(scores)),
        score_min=float(np.min(scores)),score_max=float(np.max(scores)),
        time_limit_rate=float(np.mean([x['finish']=='time_limit' for x in episodes])),
        elimination_rate=float(np.mean([x['finish']=='quadruped_eliminated' for x in episodes])),
        mean_simulation_seconds=float(np.mean([x['simulation_seconds'] for x in episodes])),
        mean_blocked_rate=float(np.mean([x['blocked_actions']/max(1,x['decisions']) for x in episodes])),
        mean_committed_rate=float(np.mean([x['committed_actions']/max(1,x['decisions']) for x in episodes])))


def run_episode(cfg,seed,policy,trajectory_dir,name):
    env=SkillEnv(cfg,seed=seed,record_trajectory=True);observation,_=env.reset(seed)
    if name=='ppo_no_visual':
        env.engine.memory=VisualMemory(cfg['start_b'])
        env.sensor.sample=lambda match,camera_pan_degrees:None
        observation=env._observation()
    inference_ns=[]
    while not env.match.ref.finished:
        started=time.perf_counter_ns();action=policy(observation,env);inference_ns.append(time.perf_counter_ns()-started)
        observation,_,done,_,_=env.step(action)
        if done:break
    summary=env.episode_summary();trajectory=summary.pop('trajectory')
    dump(trajectory_dir/f'{name}_seed_{seed}.json',dict(summary=summary,steps=trajectory))
    summary['mean_policy_inference_ms']=statistics.fmean(inference_ns)/1e6
    return summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/rl_skill_v1.json')
    parser.add_argument('--output-dir',type=Path);parser.add_argument('--seeds',nargs='*',type=int)
    args=parser.parse_args();cfg=load_rl_config(args.config);seeds=args.seeds or cfg['evaluation']['seeds']
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output=(args.output_dir or workspace_root()/'experiments/rl-skill-eval'/stamp).resolve();output.mkdir(parents=True,exist_ok=False)
    trajectory_dir=output/'trajectories';trajectory_dir.mkdir()
    rng=np.random.default_rng(0);agent,metadata=PPOAgent.load(args.checkpoint,rng,restore_optimizer=False)
    probe=SkillEnv(cfg,seed=seeds[0]);observation,_=probe.reset(seeds[0])
    if agent.observation_size!=len(observation) or agent.action_size!=probe.action_size:raise ValueError('Checkpoint/environment contract mismatch')
    stochastic_rngs={}
    policies={
        'ppo':lambda obs,env:agent.act(obs,rng,deterministic=True)[0],
        'ppo_stochastic':lambda obs,env:agent.act(obs,stochastic_rngs.setdefault(('ppo_stochastic',env.cfg['seed']),
            np.random.default_rng(900000+env.cfg['seed'])),deterministic=False)[0],
        'ppo_no_visual':lambda obs,env:agent.act(obs,stochastic_rngs.setdefault(('ppo_no_visual',env.cfg['seed']),
            np.random.default_rng(900000+env.cfg['seed'])),deterministic=False)[0],
        'capture_safe':lambda obs,env:1,
        'capture_fast':lambda obs,env:2,
        'observe_then_capture':lambda obs,env:5 if env.decision_index%20==0 else 1,
        'alternate_capture_1s':lambda obs,env:1 if (env.decision_index//10)%2==0 else 2,
    }
    results={name:[] for name in policies}
    for seed in seeds:
        for name,policy in policies.items():
            episode=run_episode(cfg,int(seed),policy,trajectory_dir,name);results[name].append(episode)
            print(json.dumps(dict(policy=name,**episode),allow_nan=False),flush=True)
    summary=dict(status='evaluated',contract_version=cfg['contract_version'],checkpoint=str(args.checkpoint.resolve()),
        checkpoint_metadata=metadata,seeds=list(seeds),policies={name:metrics(episodes) for name,episodes in results.items()},
        paired={name:{} for name in ('ppo','ppo_stochastic')})
    for learned in ('ppo','ppo_stochastic'):
        learned_scores={x['seed']:x['score'] for x in results[learned]}
        for baseline in ('capture_safe','capture_fast','observe_then_capture','alternate_capture_1s'):
            other={x['seed']:x['score'] for x in results[baseline]};deltas=[learned_scores[s]-other[s] for s in seeds]
            summary['paired'][learned][baseline]=dict(mean_score_delta=float(np.mean(deltas)),delta_std=float(np.std(deltas)),
                wins=sum(d>0 for d in deltas),ties=sum(d==0 for d in deltas),losses=sum(d<0 for d in deltas))
    summary['model']=dict(parameter_count=agent.parameter_count(),checkpoint_bytes=args.checkpoint.stat().st_size,
        forward_benchmark=agent.benchmark(observation))
    dump(output/'episodes.json',results);dump(output/'evaluation_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))


if __name__=='__main__':main()
