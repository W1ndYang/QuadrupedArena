"""Posthoc public-data-only report for an interrupted API trial. No replay/rerun."""
import argparse
import csv
import hashlib
import html
import json
import statistics
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from deepseek_trial import public_observation
from qwen_client import usage_total


def dump(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def build(root):
    read=lambda p:json.loads((root/p).read_text())
    original=read('result.json');assert original['status']=='failed'
    decisions=[json.loads(p.read_text()) for p in sorted((root/'decisions').glob('*.json'))]
    if not decisions:
        protocol=read('protocol.json');profile=read('api_profile.json');observation=read('observations/000.json')
        failure=read('api_calls/000_failure.json')
        response=read('api_calls/000_response.json') if (root/'api_calls/000_response.json').exists() else None
        summary=dict(status='interrupted_not_finished',model=profile['model'],reasoning_effort=profile['reasoning_effort'],
            completed_decisions=0,attempted_requests=1,simulation_seconds=observation['simulation_seconds'],
            score_at_interruption=observation['public_state']['score'],own_hp_at_interruption=observation['self_state']['hp'],
            failed_request=failure,failed_request_usage=response.get('usage') if response else None,full_match_completed=False,automatic_retries=0)
        dump(root/'test_summary.json',summary);dump(root/'decisions.json',[])
        body=html.escape(json.dumps(summary,ensure_ascii=False,indent=2))
        frame=html.escape(observation['frames'][-1]['file'])
        (root/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>首步请求失败，未执行比赛</title><h1>'+html.escape(profile['model'])+'：首步请求失败</h1><p>没有产生已接受决策，没有伪造回放或自动重试。原始观测、请求和失败记录保留。</p><img width="768" src="'+frame+'"><pre>'+body+'</pre></html>')
        print(json.dumps(summary,ensure_ascii=False));return
    dump(root/'decisions.json',decisions)  # Aggregate existing accepted records; no invented action.
    observations=[read(f'observations/{i:03d}.json') for i in range(len(decisions)+1)]
    for o in observations:public_observation(o)
    frames=read('frame_index.json');protocol=read('protocol.json');profile=read('api_profile.json')
    failure=read(f'api_calls/{len(decisions):03d}_failure.json');last=observations[-1]
    records=[read(f'api_calls/{i:03d}_response.json') for i in range(len(decisions))];usage=usage_total(records)
    response_path=root/f'api_calls/{len(decisions):03d}_response.json'
    failed_response=json.loads(response_path.read_text()) if response_path.exists() else None
    capped=bool(failed_response and failed_response.get('finish_reason')=='length')
    invalid=bool(failed_response and failed_response.get('finish_reason')=='stop')
    cause='输出预算耗尽' if capped else '输出格式或动作校验失败' if invalid else failure['error_type']
    all_usage=usage_total(records+[failed_response]) if failed_response else None
    token_note=(f"全部请求总Token {all_usage['total_tokens']}（包括未产出决策的末次请求）。" if all_usage and all_usage['usage_complete'] else '失败请求usage未知，可能计费，不能当成0或宣称已知用量为完整费用。')
    times=[d['api']['api_wall_seconds'] for d in decisions]
    summary=dict(status='interrupted_not_finished',model=profile['model'],reasoning_effort=profile['reasoning_effort'],
        completed_decisions=len(decisions),attempted_requests=len(decisions)+1,simulation_seconds=last['simulation_seconds'],
        score_at_interruption=last['public_state']['score'],own_hp_at_interruption=last['self_state']['hp'],
        api_completed_mean_seconds=statistics.mean(times),api_completed_median_seconds=statistics.median(times),
        api_completed_min_seconds=min(times),api_completed_max_seconds=max(times),failed_request=failure,
        completed_requests_usage=usage,failed_request_usage=failed_response.get('usage') if failed_response else None,
        full_run_token_total=all_usage['total_tokens'] if all_usage else None,all_requests_usage=all_usage,failure_cause=cause,
        selected_skills=dict(Counter(d['skill'] for d in decisions)),visual_annotations=sum('sighting' in d['params'] for d in decisions),
        frames=len(frames),automatic_retries=0,full_match_completed=False,
        caveats=['Partial score is not a final match result',token_note,
            'Provider returned finish_reason=length; not a network failure' if capped else 'Complete response failed output validation; not a transport failure' if invalid else 'Error record alone does not establish provider versus network root cause',
            f"Answer/output budget {profile['max_tokens']}, timeout {profile['timeout_seconds']}; only one sample per setting"])
    dump(root/'test_summary.json',summary)
    hashes=[]
    for f in frames:
        assert f['simulation_seconds']<=last['simulation_seconds']
        assert Image.open(root/f['file']).size==(768,432)
        hashes.append(dict(file=f['file'],sha256=hashlib.sha256((root/f['file']).read_bytes()).hexdigest()))
    for i,d in enumerate(decisions):
        action=read(f'actions/{i:03d}.json')
        assert set(action)==set(protocol['intent_schema'])
        assert action=={k:d[k] for k in action}
    dump(root/'vision_audit.json',dict(status='partial_records_checked',not_complete_match=True,
        public_observation_whitelist=True,raw_decisions_match=True,frames=hashes,
        partial_truth_timeline_available=(root/'timeline.json').exists(),full_truth_timeline_available=False))
    with (root/'decision_timing.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['decision','status','simulation_seconds','api_started_utc','api_finished_utc','api_seconds','tokens'])
        for d in decisions:
            a=d['api'];w.writerow([d['observation_id'],'completed',d['simulation_seconds'],a['api_started_utc'],a['api_finished_utc'],a['api_wall_seconds'],d['tokens']])
        w.writerow([len(decisions),'failed_'+cause,last['simulation_seconds'],failure['call_start_utc'],failure['call_failed_utc'],failure['call_until_failure_seconds'],(failed_response.get('usage') or {}).get('total_tokens') if failed_response else 'unknown'])
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18);images=[];durations=[]
    for i,f in enumerate(frames):
        im=Image.new('RGB',(768,482),'#302018');im.paste(Image.open(root/f['file']).convert('RGB'),(0,50))
        ImageDraw.Draw(im).text((12,10),f"未完赛 · 中断前机载记录  t={f['simulation_seconds']:.2f}s · 止于{last['simulation_seconds']:g}秒",font=font,fill='white')
        images.append(im.quantize(colors=192));durations.append(round(1000*(frames[i+1]['simulation_seconds']-f['simulation_seconds'])) if i+1<len(frames) else 1500)
    images[0].save(root/'onboard.gif',save_all=True,append_images=images[1:],duration=durations,loop=0,disposal=2)
    review=dict(written_after_match=False,written_after_interruption=True,not_planner_input=True,
        findings=[f"本轮未完赛：第{len(decisions)+1}次请求在{failure['call_until_failure_seconds']:.2f}秒后发生{cause}，未收到完整决策，没有重试。",
            f"中断时比赛{last['simulation_seconds']:.2f}秒、得分{last['public_state']['score']:.2f}、四足剩{last['self_state']['hp']}HP；这不是淘汰或最终成绩。",
            f"已完成请求平均{statistics.mean(times):.2f}秒、中位数{statistics.median(times):.2f}秒；失败请求不包含在此均值中。",
            f"已完成请求总Token {usage['total_tokens']}，推理Token {usage['reasoning_tokens']}；"+token_note,
            f"技能选择：{summary['selected_skills']}，视觉目标标注{summary['visual_annotations']}次。未展示出新增躲藏/瞄准意图。"],
        next_steps=['先确定推理预算和失败处理，再单独启动新试验；本次不擅自续跑或重试。'],
        limits=[f"原始result.json和API失败记录未改写；本页仅展示已保存的公开观测与{len(decisions)}条已执行决策。",
            '接口返回length，输出未完整生成；并非HTTP断流。' if capped else '接口返回stop，正文未通过格式或动作校验；不是网络中断或超时。' if invalid else '错误记录本身不能区分提供方、网络和本地原因，见原始失败类型。',
            f"模型{profile['model']}，思考配置{profile['reasoning_effort']}；开始结束时间由外部记录，不是模型自报。",
            '没有完整终局轨迹，不能提供本轮完整第三视角回放或与low最终得分作胜负比较。'])
    if (root/'visual_spotcheck.json').exists():
        audit=read('visual_spotcheck.json');review['findings'].extend(audit['findings']);review['limits'].append('视觉核对仅是抽查，不是对所有帧的完整识别准确率评测；模型自述不能证明内部推理质量。')
    dump(root/'post_run_review.json',review)
    # Presentation-only view model; original result.json remains untouched.
    result=dict(status='failed',mode='qwen_semantic_trial' if profile['model']=='qwen3.6-27b' else 'deepseek_semantic_trial',semantic_skills=True,score=last['public_state']['score'],
        simulation_seconds=last['simulation_seconds'],finish=cause+'，未完赛',
        decision_wait_seconds=sum(d['decision_wait_seconds'] for d in decisions),tokens=None)
    data=dict(result=result,protocol=protocol,decisions=decisions,observations=observations,frames=frames,review=review)
    template=(Path(__file__).parent/'report_templates/vision_viewer.html').read_text()
    template=template.replace('<main>',f'<main><p class="warn">本轮未完成：以下分数是{last["simulation_seconds"]:g}秒中断时的暂时分数，不是终局成绩。第{len(decisions)+1}次请求{cause}，{len(decisions)}次决策已执行；本页不展示虚构的后续比赛。</p>')
    (root/'report.html').write_text(template.replace('__VISION_DATA__',json.dumps(data,ensure_ascii=False).replace('<','\\u003c')))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);build(p.parse_args().directory)
