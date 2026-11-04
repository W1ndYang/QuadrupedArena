"""Dependency-light NumPy PPO for the discrete Skill policy."""
import json
from pathlib import Path
import time

import numpy as np


def softmax(logits):
    shifted=logits-np.max(logits,axis=-1,keepdims=True)
    values=np.exp(shifted)
    return values/np.sum(values,axis=-1,keepdims=True)


class Adam:
    def __init__(self,parameters,learning_rate):
        self.learning_rate=float(learning_rate);self.step_count=0
        self.m={k:np.zeros_like(v) for k,v in parameters.items()}
        self.v={k:np.zeros_like(v) for k,v in parameters.items()}

    def step(self,parameters,gradients,max_norm):
        norm=math_norm(gradients)
        scale=min(1.,float(max_norm)/(norm+1e-8))
        self.step_count+=1
        for key in parameters:
            g=gradients[key]*scale
            self.m[key]=.9*self.m[key]+.1*g
            self.v[key]=.999*self.v[key]+.001*g*g
            mh=self.m[key]/(1-.9**self.step_count);vh=self.v[key]/(1-.999**self.step_count)
            parameters[key]-=self.learning_rate*mh/(np.sqrt(vh)+1e-8)
        return norm

    def arrays(self,prefix):
        result={prefix+'_step':np.asarray(self.step_count,dtype=np.int64)}
        for key,value in self.m.items():result[prefix+'_m_'+key]=value
        for key,value in self.v.items():result[prefix+'_v_'+key]=value
        return result

    def restore(self,arrays,prefix):
        self.step_count=int(arrays[prefix+'_step'])
        for key in self.m:
            self.m[key][...]=arrays[prefix+'_m_'+key]
            self.v[key][...]=arrays[prefix+'_v_'+key]


def math_norm(gradients):
    return float(np.sqrt(sum(float(np.sum(g*g)) for g in gradients.values())))


class MLP:
    def __init__(self,input_size,hidden_size,output_size,rng,output_scale=1.):
        self.parameters={
            'w1':(rng.normal(0,1,(input_size,hidden_size))/np.sqrt(input_size)).astype(np.float32),
            'b1':np.zeros(hidden_size,dtype=np.float32),
            'w2':(rng.normal(0,1,(hidden_size,output_size))*output_scale/np.sqrt(hidden_size)).astype(np.float32),
            'b2':np.zeros(output_size,dtype=np.float32),
        }

    def forward(self,x):
        hidden=np.tanh(x@self.parameters['w1']+self.parameters['b1'])
        return hidden@self.parameters['w2']+self.parameters['b2'],hidden

    def backward(self,x,hidden,output_gradient):
        dh=(output_gradient@self.parameters['w2'].T)*(1-hidden*hidden)
        return {
            'w2':hidden.T@output_gradient,
            'b2':np.sum(output_gradient,axis=0),
            'w1':x.T@dh,
            'b1':np.sum(dh,axis=0),
        }


class PPOAgent:
    def __init__(self,observation_size,action_size,hidden_size,rng,actor_lr=3e-4,critic_lr=1e-3,repeat_logit=1.5):
        self.observation_size=int(observation_size);self.action_size=int(action_size);self.hidden_size=int(hidden_size)
        self.repeat_logit=float(repeat_logit)
        self.actor=MLP(observation_size,hidden_size,action_size,rng,output_scale=.1)
        self.critic=MLP(observation_size,hidden_size,1,rng,output_scale=1.)
        # A mild capture prior avoids an idle initial argmax without preventing
        # observation-conditional actions from becoming deterministic.
        self.actor.parameters['b2'][1]=.5
        self.actor_optimizer=Adam(self.actor.parameters,actor_lr)
        self.critic_optimizer=Adam(self.critic.parameters,critic_lr)

    def policy(self,observations):
        x=np.atleast_2d(observations).astype(np.float32);logits,hidden=self.actor.forward(x)
        logits=logits+self.repeat_logit*x[:,-self.action_size:]
        return softmax(logits),hidden,logits

    def values(self,observations):
        x=np.atleast_2d(observations).astype(np.float32);values,_=self.critic.forward(x)
        return values[:,0]

    def act(self,observation,rng,deterministic=False):
        probabilities,_,_=self.policy(observation)
        p=probabilities[0]
        action=int(np.argmax(p) if deterministic else rng.choice(self.action_size,p=p))
        return action,float(np.log(max(p[action],1e-12))),float(self.values(observation)[0])

    def update(self,batch,cfg,rng):
        advantages=batch['advantages'].astype(np.float32)
        advantages=(advantages-advantages.mean())/(advantages.std()+1e-8)
        n=len(advantages);indices=np.arange(n);metrics=[]
        for _ in range(int(cfg['epochs'])):
            rng.shuffle(indices)
            for start in range(0,n,int(cfg['minibatch_size'])):
                take=indices[start:start+int(cfg['minibatch_size'])]
                x=batch['observations'][take];actions=batch['actions'][take];adv=advantages[take]
                old_logp=batch['log_probabilities'][take];returns=batch['returns'][take]
                probabilities,hidden,_=self.policy(x)
                selected=np.maximum(probabilities[np.arange(len(take)),actions],1e-12)
                logp=np.log(selected);ratio=np.exp(logp-old_logp)
                clip=float(cfg['clip_ratio']);active=((adv>=0)&(ratio<=1+clip))|((adv<0)&(ratio>=1-clip))
                dlogp=np.where(active,-adv*ratio/len(take),0.)
                output_gradient=-probabilities*dlogp[:,None]
                output_gradient[np.arange(len(take)),actions]+=dlogp
                log_probabilities=np.log(np.maximum(probabilities,1e-12))
                mean_plogp=np.sum(probabilities*log_probabilities,axis=1,keepdims=True)
                output_gradient+=float(cfg['entropy_coefficient'])*probabilities*(log_probabilities-mean_plogp)/len(take)
                actor_gradients=self.actor.backward(x,hidden,output_gradient.astype(np.float32))
                actor_norm=self.actor_optimizer.step(self.actor.parameters,actor_gradients,cfg['max_grad_norm'])

                predicted,value_hidden=self.critic.forward(x);error=predicted[:,0]-returns
                value_gradient=(error/len(take))[:,None]
                critic_gradients=self.critic.backward(x,value_hidden,value_gradient.astype(np.float32))
                critic_norm=self.critic_optimizer.step(self.critic.parameters,critic_gradients,cfg['max_grad_norm'])
                unclipped=ratio*adv;clipped=np.clip(ratio,1-clip,1+clip)*adv
                metrics.append(dict(policy_loss=-float(np.mean(np.minimum(unclipped,clipped))),
                    value_loss=.5*float(np.mean(error*error)),entropy=-float(np.mean(np.sum(probabilities*log_probabilities,axis=1))),
                    approx_kl=float(np.mean(old_logp-logp)),clip_fraction=float(np.mean(np.abs(ratio-1)>clip)),
                    actor_gradient_norm=actor_norm,critic_gradient_norm=critic_norm))
        return {key:float(np.mean([m[key] for m in metrics])) for key in metrics[0]}

    def checkpoint_arrays(self,metadata):
        arrays={}
        for key,value in self.actor.parameters.items():arrays['actor_'+key]=value
        for key,value in self.critic.parameters.items():arrays['critic_'+key]=value
        arrays.update(self.actor_optimizer.arrays('actor_adam'));arrays.update(self.critic_optimizer.arrays('critic_adam'))
        arrays['metadata_json']=np.asarray(json.dumps(metadata,sort_keys=True))
        return arrays

    def save(self,path,metadata):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix(path.suffix+'.tmp')
        with temporary.open('wb') as stream:np.savez_compressed(stream,**self.checkpoint_arrays(metadata))
        temporary.replace(path)

    @classmethod
    def load(cls,path,rng,restore_optimizer=True):
        with np.load(path,allow_pickle=False) as arrays:
            metadata=json.loads(str(arrays['metadata_json']))
            agent=cls(metadata['observation_size'],metadata['action_size'],metadata['hidden_size'],rng,
                metadata['actor_learning_rate'],metadata['critic_learning_rate'],metadata.get('repeat_logit',1.5))
            for key in agent.actor.parameters:agent.actor.parameters[key][...]=arrays['actor_'+key]
            for key in agent.critic.parameters:agent.critic.parameters[key][...]=arrays['critic_'+key]
            if restore_optimizer:
                agent.actor_optimizer.restore(arrays,'actor_adam');agent.critic_optimizer.restore(arrays,'critic_adam')
        return agent,metadata

    def parameter_count(self):
        return sum(v.size for v in self.actor.parameters.values())+sum(v.size for v in self.critic.parameters.values())

    def benchmark(self,observation,iterations=10000):
        started=time.perf_counter_ns()
        for _ in range(iterations):self.policy(observation)
        elapsed=time.perf_counter_ns()-started
        return dict(iterations=iterations,total_seconds=elapsed/1e9,mean_milliseconds=elapsed/iterations/1e6,
            ppo_parameter_count=self.parameter_count())


def compute_gae(rewards,dones,values,last_value,gamma,gae_lambda):
    rewards=np.asarray(rewards,dtype=np.float32);dones=np.asarray(dones,dtype=np.float32);values=np.asarray(values,dtype=np.float32)
    advantages=np.zeros_like(rewards);gae=0.
    for t in reversed(range(len(rewards))):
        next_value=last_value if t==len(rewards)-1 else values[t+1]
        nonterminal=1.-dones[t]
        delta=rewards[t]+gamma*next_value*nonterminal-values[t]
        gae=delta+gamma*gae_lambda*nonterminal*gae;advantages[t]=gae
    return advantages,advantages+values
