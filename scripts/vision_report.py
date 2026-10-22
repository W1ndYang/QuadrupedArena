"""Post-match viewer for the actual online RGB inputs, not replay substitutes."""
import argparse
import csv
import hashlib
import html
import json
import math
import statistics
from collections import Counter
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from llm_trial import dump


def build(root):
    read=lambda name:json.loads((root/name).read_text())
    result=read('result.json');assert result['status']=='passed'
    assert result['mode'] in ('assistant_visual_trial','assistant_semantic_trial','deepseek_semantic_trial','qwen_semantic_trial','assistant_assessment_trial'),'Do not present engineering smoke as an assistant match'
    semantic=bool(result.get('semantic_skills'))
    decisions=read('decisions.json');frames=read('frame_index.json');protocol=read('protocol.json')
    observations=[read(f'observations/{i:03d}.json') for i in range(len(decisions)+1)]
    timeline=read('timeline.json');states={s['t']:s for s in timeline}
    checks=[]
    for i,obs in enumerate(observations):
        assert set(obs)==({'observation_id','simulation_seconds','self_state','public_state','frames','perception'}|({'opponent_initial_xy'} if i==0 else set())|({'skills'} if semantic else set()))
        assert set(obs['self_state'])=={'xy','yaw','hp','speed','cooldown','skill_status'}
        assert set(obs['public_state'])=={'score','active_target','target_generation','target_remaining','dwell','finished'}
        assert obs['observation_id']==i
        assert len(obs['frames'])<=5
        assert obs['frames'][-1]['simulation_seconds']==obs['simulation_seconds']
        if i:assert obs['frames'][0]==observations[i-1]['frames'][-1]
        for f in obs['frames']:assert f['simulation_seconds']<=obs['simulation_seconds']
    for i,f in enumerate(frames):
        s=states[f['simulation_seconds']]
        assert f['own_xy']==s['xy'][0]
        assert math.isclose(f['body_yaw_degrees'],math.degrees(s['yaw'][0]),abs_tol=1e-8)
        assert math.isclose(f['camera_world_yaw_degrees'],f['body_yaw_degrees']+f['camera_pan_degrees'],abs_tol=1e-8)
        assert -90-1e-8<=f['camera_pan_degrees']<=90+1e-8
        assert Image.open(root/f['file']).size==(768,432)
        if i:
            dt=f['simulation_seconds']-frames[i-1]['simulation_seconds']
            assert 0<dt<=.5+1e-8
            assert abs(f['camera_pan_degrees']-frames[i-1]['camera_pan_degrees'])<=120*dt+1e-8
        checks.append(dict(file=f['file'],sha256=hashlib.sha256((root/f['file']).read_bytes()).hexdigest()))
    for i,d in enumerate(decisions):
        assert d['observation_id']==i and d['simulation_seconds']==observations[i]['simulation_seconds']
        action=d['compiled_action'] if semantic else d['action']
        assert action['skill']!='AIM' and action.get('facing')!='enemy'
        if semantic:
            request=read(f'actions/{i:03d}.json')
            if result['mode'] in ('deepseek_semantic_trial','qwen_semantic_trial','assistant_assessment_trial'):assert set(request)==set(protocol['intent_schema'])
            else:assert set(request)=={'observation_id','skill','params','reason'} and len(request['reason'])<=100
            assert 'goal' not in request['params'] and 'via' not in request['params']
        assert 0<observations[i+1]['simulation_seconds']-d['simulation_seconds']<=2+1e-8
    dump(root/'vision_audit.json',dict(status='passed',public_field_whitelist=True,initial_enemy_only=True,
        no_privileged_tracking_actions=True,frame_times_and_own_poses_match_timeline=True,
        bounded_gimbal_rate=True,raw_frame_count=len(frames),observations=len(observations),frames=checks))
    with (root/'decision_timing.csv').open('w',newline='') as stream:
        writer=csv.writer(stream);writer.writerow(['decision','sim_start','sim_end','input_frames','wait_seconds','skill_compile_seconds','tokens','api_wall_seconds','first_output_seconds','first_content_seconds','reasoning_tokens','api_started_utc','api_finished_utc'])
        for i,d in enumerate(decisions):
            a=d.get('api',{})
            writer.writerow([i,d['simulation_seconds'],observations[i+1]['simulation_seconds'],len(observations[i]['frames']),d['decision_wait_seconds'],d.get('skill_compile_seconds',''),d.get('tokens'),a.get('api_wall_seconds'),a.get('time_to_first_output_seconds'),a.get('time_to_first_visible_content_seconds'),a.get('reasoning_tokens'),a.get('api_started_utc'),a.get('api_finished_utc')])
    font=ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',18)
    gif=[];durations=[]
    for i,f in enumerate(frames):
        im=Image.new('RGB',(768,482),'#132333');im.paste(Image.open(root/f['file']).convert('RGB'),(0,50))
        s=states[f['simulation_seconds']]
        ImageDraw.Draw(im).text((12,10),f"机载采样回放  t={f['simulation_seconds']:.2f}s  云台 {f['camera_pan_degrees']:+.0f}°  HP {s['hp'][0]}  得分 {s['score']:.2f}",font=font,fill='white')
        gif.append(im.quantize(colors=192))
        durations.append(round((frames[i+1]['simulation_seconds']-f['simulation_seconds'])*1000) if i+1<len(frames) else 1500)
    gif[0].save(root/'onboard.gif',save_all=True,append_images=gif[1:],duration=durations,loop=0,optimize=False,disposal=2)
    review=dict(written_after_match=True,not_planner_input=True,
        conclusion='在线非全知视觉闭环已跑通；本轮7.82分、30.62秒淘汰，不证明策略优于旧方案。',
        findings=[
            '多帧有效：7.5秒侧向帧出现人形，8秒末帧已因视角转动而看不到；决策004使用了该中间帧。',
            '扫描并非越多越好：贴墙时多帧拍到同一墙面；发现人形后改成侧向保持，近距离时改成前向保持。',
            '早期风险估计不足：在8秒已看到人形后仍走较长开阔转移，赛后裁判记录8.62和11.24秒受击。',
            '18秒提前离开High3北侧，20秒坍塌画面同步正确；放弃P2末尾0.36分，保留当时1HP。',
            '26秒发现墙南端肢体并调整入点；28秒近距离威胁使策略转为倒退撤离，但尚未及时切断射线。',
            '终局失败：30秒才改为绕墙，运动执行器需要减速和转身。赛后记录人形29.58秒开始有效瞄准，30.62秒击中；新指令并未瞬间完成躲避。'],
        next_steps=[
            '下一轮先给技能增加基于视觉目标估计的短时避险/侧移接口，并把转向、加减速耗时显式提供给规划器。',
            '保持2秒高层周期，另做受击/新目标出现的事件触发中断对照；这会改变协议，需单独标记，不追改本轮。',
            '用独立新会话或独立模型只输入白名单观测，并做多种子、相同观测和技能接口的基线比较。'],
        limits=[
            '这是Python运动学比赛＋Isaac在线渲染，不是Isaac接触动力学，也不是真实照片。',
            '相机90°视场，云台±90°、最高120°/秒；机身射击总角60°，相机转动不会替机身自动瞄准。',
            '每0.5秒采样，不是连续视频感知。第一步只有t=0一帧；后续通常上一边界帧加4张新帧，终局不足2秒。',
            '决策时未读取对手当前坐标、朝向、HP、瞄准计时、内部状态或事件真值。渲染/裁判后端仍持有这些数据。',
            '这是同一开发会话中的接口隔离，保留了历史工程讨论与规则实现背景，不是全新上下文的严格盲测或OS权限隔离。',
            '公开自身冷却能间接反映自己命中过对方；保留该合法自身状态，不提供对方剩余HP。',
            '碰撞保护使用后台几何避免穿墙/相互穿越，不做隐藏敌情驱动的战术绕路；本轮保护触发0次。',
            '相机视点高0.60米、前移0.36米；射击采用既有代表点高度，能看见局部身体不等于射线判定有效。',
            '决策耗时从观测就绪到动作提交，包含看图、推理与工具开销，不是纯模型推理耗时。Token不可获取，记录null而非0。',
            '没有回滚、预演未来或失败后重跑筛选。工程预检另存在20261021-173441-086798；本轮为首次正式运行。'])
    if result['mode'] in ('qwen_semantic_trial','assistant_assessment_trial'):
        from assessment_review import review_for
        review=review_for(root,result,decisions,protocol)
    elif result['mode']=='deepseek_semantic_trial':
        api_times=[d['api']['api_wall_seconds'] for d in decisions]
        usage=read('api_usage.json')
        summary=dict(written_after_match=True,not_planner_input=True,model=protocol['planner'],reasoning_effort=protocol['reasoning_effort'],
            score=result['score'],simulation_seconds=result['simulation_seconds'],finish=result['finish'],
            decisions=len(decisions),frames=len(frames),api_mean_seconds=statistics.mean(api_times),
            api_median_seconds=statistics.median(api_times),api_min_seconds=min(api_times),api_max_seconds=max(api_times),
            decision_wait_mean_seconds=result['decision_wait_seconds']/len(decisions),
            skill_compile_mean_seconds=result['skill_compile_seconds']/len(decisions),
            selected_skills=dict(Counter(d['skill'] for d in decisions)),
            visual_annotations=sum('sighting' in d['params'] for d in decisions),usage=usage,
            automatic_retries=0,raw_decision_fields=protocol['intent_schema'],
            pure_thinking_seconds=None,timing_caveat='API elapsed includes network, queue, compute and output; not pure thinking time',
            comparison_caveat='Different planner context and patched skill layer versus earlier assistant; not model-only A/B')
        dump(root/'test_summary.json',summary)
        review=dict(written_after_match=True,not_planner_input=True,
            conclusion=f"DeepSeek官方Flash思考强度 {protocol['reasoning_effort']}，非全知机载多帧技能闭环。",
            findings=[f"本轮{len(decisions)}次决策、{len(frames)}帧，得分{result['score']:.2f}，比赛时长{result['simulation_seconds']:.2f}秒，结束原因：{result['finish']}。",
                f"平均端到端等待{result['decision_wait_seconds']/len(decisions):.2f}秒；API各阶段时间和实际usage见耗时CSV/api_calls。",
                f"API平均{summary['api_mean_seconds']:.2f}秒，中位数{summary['api_median_seconds']:.2f}秒，范围{min(api_times):.2f}～{max(api_times):.2f}秒；底层规划平均{summary['skill_compile_mean_seconds']:.3f}秒。",
                f"实际选择技能次数：{summary['selected_skills']}；视觉脚底标注{summary['visual_annotations']}次。提交占点意图不代表底层一直直线运动，受击撤掩体等保护由代码执行。",
                f"提供方记录总Token {usage['total_tokens']}，其中推理Token {usage['reasoning_tokens']}；短JSON不等于内部思考量很少。",
                ('模型输出决策及逐帧视觉事实、公开状态依据、选择摘要与不确定性，供人工对照截图；这些自述不是内部思维链，也不单独证明理解正确。' if protocol.get('assessment_output') else '模型仅输出三字段决策，不要求解释或计时。')+'外部记录API时间戳，不保存内部reasoning_content。'],
            next_steps=['检查逐帧识别、技能执行与占点效果，再在相同接口和多种子上比较模型。'],
            limits=['Python运动学＋Isaac在线RGB，没有接触动力学；每2秒决策，每0.5秒采样。',
                '仅初始敌方位置已知，之后只提供机载图、公开自状态及视觉记忆；标注脚底像素不是成熟检测器。',
                'DeepSeek接收当前多帧和最近4条动作摘要，不继承助手开发会话；与旧助手试验不是严格模型单变量对照。',
                f"使用修复后语义技能1.1；思考开启，档位 {protocol['reasoning_effort']}。API流式首输出可能只是推理，完整动作时间更重要。",
                'deepseek-flash是滚动别名；记录官方文档对应版本与实际响应model，不能保证服务端版本永远固定。',
                'Token仅采用提供方usage，缺失则null；不自动重试、不自动回退或修复策略。'])
    elif semantic:
        from semantic_review import review_for
        review=review_for(root,result,decisions,observations)
    # Optional post-match audit notes affect presentation only, never planner input.
    if (root/'review_addendum.json').exists():
        addendum=read('review_addendum.json')
        assert addendum.get('written_after_match') is True and addendum.get('not_planner_input') is True
        for key in ('findings','next_steps','limits'):
            items=addendum.get(key,[])
            assert isinstance(items,list) and all(isinstance(item,str) for item in items)
            review[key].extend(items)
    dump(root/'post_run_review.json',review)
    data=dict(result=result,protocol=protocol,decisions=decisions,observations=observations,frames=frames,review=review)
    if (root/'third_person_replay.json').exists():data['replay']=read('third_person_replay.json')
    payload=json.dumps(data,ensure_ascii=False).replace('<','\\u003c')
    template=Path(__file__).with_name('report_templates')/'vision_viewer.html'
    (root/'report.html').write_text(template.read_text().replace('__VISION_DATA__',payload))
    print(json.dumps(dict(report=str(root/'report.html'),gif=str(root/'onboard.gif'),audit='passed',wait_total_seconds=result['decision_wait_seconds'],wait_mean_seconds=result['decision_wait_seconds']/len(decisions)),ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);build(parser.parse_args().directory)
