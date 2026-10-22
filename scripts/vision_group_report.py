"""Posthoc comparison hub; never invokes models or alters original run records."""
import argparse
import datetime
import hashlib
import html
import json
import os
from pathlib import Path
from llm_trial import dump


def build(runs,output,incident=None):
    output.mkdir(parents=True,exist_ok=False);rows=[]
    for root in runs:
        result=json.loads((root/'result.json').read_text())
        protocol=json.loads((root/'protocol.json').read_text())
        summary=json.loads((root/'test_summary.json').read_text()) if (root/'test_summary.json').exists() else {}
        prompt=root/'planner_context.txt'
        replay=json.loads((root/'third_person_replay.json').read_text()) if (root/'third_person_replay.json').exists() else None
        rows.append(dict(directory=str(root),report=os.path.relpath(root/'report.html',output),
            onboard_gif=os.path.relpath(root/'onboard.gif',output),
            third_person_gif=os.path.relpath(root/replay['gif'],output) if replay else None,
            model=protocol.get('planner','unknown'),effort=protocol.get('reasoning_effort','unknown'),
            status=result['status'],finish=result.get('finish','未完成'),score=result.get('score'),
            simulation_seconds=result.get('simulation_seconds'),decisions=result.get('decisions',summary.get('completed_decisions')),
            api_mean_seconds=summary.get('api_mean_seconds',summary.get('api_completed_mean_seconds')),
            decision_wait_mean_seconds=summary.get('decision_wait_mean_seconds'),
            total_tokens=(summary.get('usage') or summary.get('all_requests_usage') or {}).get('total_tokens'),
            prompt_sha256=hashlib.sha256(prompt.read_bytes()).hexdigest() if prompt.exists() else None))
    incident_data=json.loads((incident/'test_summary.json').read_text()) if incident else None
    compared_files=('trial_config.json','planner_context.txt',*(f'source_snapshot/scripts/{name}.py' for name in ('semantic_skills','vision_core','trial_rules','adaptive_motion','llm_trial')))
    fingerprints={name:[hashlib.sha256((root/name).read_bytes()).hexdigest() if (root/name).exists() else None for root in runs] for name in compared_files}
    shared_audit={name:dict(sha256=values,identical=all(values) and len(set(values))==1) for name,values in fingerprints.items()}
    dump(output/'comparison.json',dict(written_after_all_trials=True,not_planner_input=True,trials=rows,integration_incident=incident_data,
        shared_files_audit=shared_audit,
        limits=['Single seed, one attempt per condition. Not statistically controlled model ranking.',
            'Qwen max/min are enabled-default-maximum-budget versus thinking-off, not native reasoning_effort tiers.',
            'Assistant inherits development context and old trials; only current-match live observations are whitelisted. Not a fresh blinded context.',
            'Assistant tokens unavailable. Assistant end-to-end waiting and cloud API latency are different measures.']))
    esc=lambda value:html.escape(str(value))
    number=lambda value:'—' if value is None else f'{value:.2f}' if isinstance(value,float) else str(value)
    labels={'maximum_default_budget':'思考开启 / 默认最大预算','thinking_off':'思考关闭','session_setting_not_api_controlled':'当前助手会话','current_assistant_session':'助手','quadruped_eliminated':'四足淘汰'}
    display=lambda value:labels.get(value,value) if isinstance(value,str) else value
    table=''.join('<tr>'+''.join('<td>'+esc(number(display(row[key])))+'</td>' for key in ('model','effort','finish','score','simulation_seconds','decisions','api_mean_seconds','decision_wait_mean_seconds','total_tokens'))+'</tr>' for row in rows)
    options=''.join(f'<option value="{esc(row["report"])}">{i+1}. {esc(display(row["model"]))} / {esc(display(row["effort"]))}</option>' for i,row in enumerate(rows))
    links=' '.join(f'<a href="{esc(row["report"])}" target="_blank">单独打开第{i+1}轮</a>' for i,row in enumerate(rows))
    media=''.join(f'<p>第{i+1}轮：<a href="{esc(row["onboard_gif"])}">机载 GIF</a>'+ (f'<a href="{esc(row["third_person_gif"])}">第三视角 GIF</a>' if row['third_person_gif'] else '第三视角尚未关联')+'</p>' for i,row in enumerate(rows))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三轮视觉决策对照</title>
<style>body{margin:0;background:#101923;color:#e9f0f7;font:16px system-ui,sans-serif}main{max-width:1440px;margin:auto;padding:22px}p{line-height:1.7}a{color:#81cff2;margin-right:20px}select{padding:10px;background:#233747;color:white;max-width:100%}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #344956;text-align:left}iframe{width:100%;height:950px;border:1px solid #344956;margin-top:20px}.scroll{overflow:auto}.note{border-left:4px solid #e5ad59;padding-left:15px}</style>
<main><h1>三轮非全知视觉决策测试</h1><p>同地图、规则、种子、五技能接口、2秒决策与0.5秒机载采样。均保留逐帧依据、原始JSON、外部时间和机载GIF；Python运动学＋Isaac在线RGB，无接触动力学。</p>
<p class="note">Qwen“最高/最低”实际采用开启思考＋服务端默认最大思考预算 / 关闭思考，非原生low/max档位。两轮回答上限16384、请求截止600秒。助手继承工程上下文，Token不可获取；其等待时间含工具和看图开销，不能直接当成API推理时间。每组仅一次，不作普遍性能排名。</p>
<p>参数依据：<a href="https://help.aliyun.com/zh/model-studio/qwen-api-via-openai-chat-completions">百炼官方接口文档</a>；<a href="comparison.json">原始对照统计</a></p>__INCIDENT__
<div class="scroll"><table><thead><tr><th>决策模型</th><th>模式</th><th>结束原因</th><th>最终分</th><th>比赛/s</th><th>决策数</th><th>API均值/s</th><th>端到端等待均值/s</th><th>总Token</th></tr></thead><tbody>__TABLE__</tbody></table></div>
<h2>阅读提示</h2><p>在下方切换各轮“结果与复盘”，查看占点、受击及脚点标注核查。助手保留工程背景、历史结果，不构成严格同上下文模型排名。</p><p>各轮含逐帧视觉依据、公开状态依据、简短决策摘要和不确定性；这些是可核验解释，不是内部思维链，也不是模型读图正确的保证。所有第三视角信息仅用于赛后展示。</p>
<h2>GIF 下载</h2>__MEDIA__
<h2>选择测试记录</h2><select id="trial" aria-label="选择一轮">__OPTIONS__</select><p>__LINKS__</p><iframe id="viewer" title="逐帧截图与决策报告"></iframe></main>
<script>const s=document.getElementById('trial'),v=document.getElementById('viewer');s.onchange=()=>v.src=s.value;s.onchange();</script></html>'''
    note=''
    if incident:
        note=f'<p class="note">另保留一次适配失败：首次Qwen第2步返回代码围栏包装的完整JSON，旧解析器拒绝，止于2秒。补齐单个完整围栏的格式解析后从头重启；不修改决策字段，不隐藏失败。额外请求2次，总Token {incident_data.get("full_run_token_total")}。<a href="{esc(os.path.relpath(incident/"report.html",output))}">查看失败原始记录</a></p>'
    (output/'index.html').write_text(page.replace('__TABLE__',table).replace('__OPTIONS__',options).replace('__LINKS__',links).replace('__INCIDENT__',note).replace('__MEDIA__',media))
    print(output/'index.html')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('runs',nargs=3,type=Path);p.add_argument('--output',type=Path);p.add_argument('--incident',type=Path)
    a=p.parse_args();build(a.runs,a.output or a.runs[0].parent/(datetime.datetime.now().strftime('%Y%m%d-%H%M%S')+'_三轮对照_Qwen思考开关_助手'),a.incident)
