"""DashScope model on the existing public-RGB semantic protocol, not old omniscient Qwen."""
import copy
import json
from pathlib import Path
from deepseek_trial import Planner as PublicPlanner,PROJECT
from qwen_client import call_api,endpoint,load_key


def settings(mode):
    config=json.loads((PROJECT/'configs/qwen_vision_trial.json').read_text())
    modes=config.pop('modes');config.update(modes[mode]);config['requested_mode']=mode
    endpoint(config)
    if config['model']!='qwen3.6-27b' or config['automatic_retries']!=0 or config['automatic_strategy_fallback']:
        raise ValueError('Unexpected Qwen model or retries')
    if config['max_calls_per_match']!=20 or (config['decision_period_seconds'],config['frame_interval_seconds'])!=(2,.5):
        raise ValueError('Unexpected call budget or cadence')
    return config


def vision_client(config,key,messages):
    current=copy.deepcopy(messages)
    for message in current:
        if not isinstance(message['content'],list):continue
        for block in message['content']:
            if block['type']=='image_url':
                # Same PNG pixels; DashScope uses pixel limits, not DeepSeek's original detail.
                block['image_url'].pop('detail',None)
    return call_api(config,key,current)


class Planner(PublicPlanner):
    def __init__(self,config,cfg,key,root,client=vision_client):
        super().__init__(config,cfg,key,root,client)
