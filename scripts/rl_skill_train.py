"""Train the v1 10 Hz Skill policy and write a reproducible run directory."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from arena_paths import workspace_root
from rl_skill_env import ACTION_NAMES,SkillEnv,load_rl_config
from rl_skill_ppo import PPOAgent,compute_gae


PROJECT=Path(__file__).resolve().parents[1]


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def git_metadata():
    def run(*args):
        result=subprocess.run(args,cwd=PROJECT,text=True,capture_output=True)
        return result.stdout.strip() if result.returncode==0 else None
    return dict(commit=run('git','rev-parse','HEAD'),branch=run('git','branch','--show-current'),diff_sha256=hashlib.sha256((run('git','diff','--binary') or '').encode()).hexdigest())


def collect(agent,env,observation,steps,rng,episode_returns,episode_scores):
    data={k:[] for k in ('observations','actions','log_probabilities','values','rewards','dones')}
    completed=[];running_return=episode_returns[-1] if episode_returns else 0.
    for _ in range(steps):
        action,logp,value=agent.act(observation,rng)
        next_observation,reward,done,_,info=env.step(action)
        for key,value_item in (('observations',observation),('actions',action),('log_probabilities',logp),
                               ('values',value),('rewards',reward),('dones',done)):
            data[key].append(value_item)
        running_return+=reward;observation=next_observation
        if done:
            completed.append(dict(seed=env.cfg['seed'],episode_return=running_return,score=info['official_score'],
                finish=info['finish'],simulation_seconds=info['time']))
            episode_scores.append(info['official_score']);episode_returns.append(0.);running_return=0.
            observation,_=env.reset()
    if not episode_returns:episode_returns.append(running_return)
    else:episode_returns[-1]=running_return
    last_value=agent.values(observation)[0]
    arrays={key:np.asarray(value) for key,value in data.items()}
    advantages,returns=compute_gae(arrays['rewards'],arrays['dones'],arrays['values'],last_value,
        env.cfg['training']['gamma'],env.cfg['training']['gae_lambda'])
    arrays.update(advantages=advantages,returns=returns)
    return arrays,observation,completed


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',type=Path,default=PROJECT/'configs/rl_skill_v1.json')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--total-steps',type=int)
    parser.add_argument('--rollout-steps',type=int)
    parser.add_argument('--resume',type=Path)
    args=parser.parse_args();cfg=load_rl_config(args.config);training=cfg['training']
    if args.total_steps is not None:training['total_steps']=args.total_steps
    if args.rollout_steps is not None:training['rollout_steps']=args.rollout_steps
    stamp=datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    output=(args.output_dir or workspace_root()/'experiments/rl-skill'/stamp).resolve();output.mkdir(parents=True,exist_ok=False)
    (output/'checkpoints').mkdir();dump(output/'resolved_config.json',cfg)
    source_snapshot=output/'source_snapshot'
    sources=[PROJECT/'requirements-cpu.txt',PROJECT/'requirements-rl.txt',PROJECT/'docs/RL_SKILL_V1.md',
        PROJECT/'scripts/run_rl_skill_v1.sh',*sorted((PROJECT/'scripts').glob('*.py')),*sorted((PROJECT/'configs').glob('*.json'))]
    for source in sources:
        destination=source_snapshot/source.relative_to(PROJECT);destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    dump(output/'environment.json',dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,git=git_metadata(),arguments=sys.argv[1:]))
    seed=int(training['seed']);rng=np.random.default_rng(seed);env=SkillEnv(cfg,seed=seed)
    observation,reset_info=env.reset(seed)
    if args.resume:
        agent,metadata=PPOAgent.load(args.resume,rng,restore_optimizer=True);global_step=int(metadata['global_step'])
        if agent.observation_size!=len(observation) or agent.action_size!=env.action_size:raise ValueError('Resume checkpoint contract mismatch')
        # Checkpoints intentionally restart at an episode boundary; advance the
        # deterministic episode-seed sequence instead of replaying early seeds.
        env.episode_index=int(metadata.get('next_episode_index',env.episode_index))
        observation,reset_info=env.reset()
    else:
        agent=PPOAgent(len(observation),env.action_size,training['hidden_size'],rng,
            training['actor_learning_rate'],training['critic_learning_rate']);global_step=0
    metadata_base=dict(contract_version=cfg['contract_version'],observation_size=len(observation),action_size=env.action_size,
        action_names=list(ACTION_NAMES),hidden_size=agent.hidden_size,actor_learning_rate=training['actor_learning_rate'],
        critic_learning_rate=training['critic_learning_rate'],repeat_logit=agent.repeat_logit,training_seed=seed)
    episode_returns=[];episode_scores=[];last_checkpoint=global_step;started=time.perf_counter();update_index=0
    log_path=output/'training.jsonl'
    with log_path.open('a') as log:
        while global_step<int(training['total_steps']):
            count=min(int(training['rollout_steps']),int(training['total_steps'])-global_step)
            rollout_started=time.perf_counter()
            batch,observation,completed=collect(agent,env,observation,count,rng,episode_returns,episode_scores)
            metrics=agent.update(batch,training,rng);global_step+=count;update_index+=1
            record=dict(update=update_index,global_step=global_step,rollout_seconds=time.perf_counter()-rollout_started,
                elapsed_seconds=time.perf_counter()-started,episodes_completed=len(completed),episodes=completed,
                rollout_mean_return=None if not completed else float(np.mean([x['episode_return'] for x in completed])),
                recent_mean_score=None if not episode_scores else float(np.mean(episode_scores[-20:])),**metrics)
            log.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');log.flush()
            print(json.dumps({k:record[k] for k in ('update','global_step','rollout_seconds','episodes_completed','recent_mean_score','policy_loss','value_loss','entropy')},allow_nan=False),flush=True)
            if global_step-last_checkpoint>=int(training['checkpoint_interval_steps']):
                agent.save(output/'checkpoints'/f'step_{global_step:09d}.npz',dict(**metadata_base,global_step=global_step,
                    next_episode_index=env.episode_index));last_checkpoint=global_step
    final=output/'checkpoints'/'final.npz';agent.save(final,dict(**metadata_base,global_step=global_step,
        next_episode_index=env.episode_index))
    summary=dict(status='trained',global_step=global_step,episodes_completed=len(episode_scores),
        mean_training_score=None if not episode_scores else float(np.mean(episode_scores)),
        recent_mean_training_score=None if not episode_scores else float(np.mean(episode_scores[-20:])),
        wall_seconds=time.perf_counter()-started,checkpoint=str(final),inference=agent.benchmark(observation))
    dump(output/'training_summary.json',summary);print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
