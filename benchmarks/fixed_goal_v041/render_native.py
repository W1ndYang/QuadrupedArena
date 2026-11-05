"""Native Isaac spectator replays for saved fixed-goal matches; no policy rerun."""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

R=Path(__file__).resolve().parent
P=R.parents[1]
sys.path.insert(0,str(R/'render_helpers'))
from arena_paths import workspace_root
os.environ.setdefault('ARENA_WORKSPACE',str(P/'.arena-workspace'))
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();parser.add_argument('--data-dir',type=Path,default=R);parser.add_argument('--only-run',type=int,choices=[1,2,3])
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();args.enable_cameras=True
R=args.data_dir.expanduser().resolve()
app=AppLauncher(args).app
import carb
import omni.usd
import omni.replicator.core as rep
import numpy as np
from pxr import UsdGeom,UsdShade,Sdf,Gf
from PIL import Image,ImageDraw
import imageio.v2 as imageio
import imageio_ffmpeg
from arena_art import Courtyard,lighting
from demo_actors import Actor
from arena_fonts import load_font
from stun_visuals import StunStars

read=lambda p:json.loads(p.read_text())
def dump(p,d):p.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
WORK=workspace_root()
CASES=read(R/'cases.json')


def curve(stage,path,color,width=.03):
    c=UsdGeom.BasisCurves.Define(stage,path);c.CreateTypeAttr('linear');c.CreateCurveVertexCountsAttr([2]);c.CreatePointsAttr([(0,0,0),(.01,0,0)])
    c.CreateWidthsAttr([width]);c.SetWidthsInterpolation('constant');c.CreateDisplayColorAttr([Gf.Vec3f(*color)]);return c


def points(c,ps):c.GetPointsAttr().Set(ps);c.GetCurveVertexCountsAttr().Set([len(ps)])


def circle(c,center,r,z=.035):points(c,[(center[0]+r*math.cos(a),center[1]+r*math.sin(a),z) for a in np.linspace(0,2*math.pi,65)])


def cloud(stage):
    root=UsdGeom.Xform.Define(stage,'/World/Smoke');move=root.AddTranslateOp()
    mat=UsdShade.Material.Define(stage,'/World/SmokeMaterial');shader=UsdShade.Shader.Define(stage,'/World/SmokeMaterial/Surface')
    shader.CreateIdAttr('UsdPreviewSurface');shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.7,.74,.79))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(1.);shader.CreateInput('opacity',Sdf.ValueTypeNames.Float).Set(.24)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    # Spectator geometry inside the rule cylinder; not a fluid-density renderer.
    for i in range(9):
        a=i*2*math.pi/8;radius=0 if i==8 else 1.0
        s=UsdGeom.Sphere.Define(stage,f'/World/Smoke/Puff{i}');s.CreateRadiusAttr(.9)
        s.AddTranslateOp().Set(Gf.Vec3d(radius*math.cos(a),radius*math.sin(a),1.15))
        s.AddScaleOp().Set(Gf.Vec3f(1.,1.,1.1))
        UsdShade.MaterialBindingAPI.Apply(s.GetPrim()).Bind(mat)
    root.MakeInvisible();return root,move


def hud(raw,s,decisions,result,number,terminal):
    width,height=raw.size;out=Image.new('RGB',(width,height+142),'#102b37');out.paste(raw,(0,86));d=ImageDraw.Draw(out)
    f20,f16,f14=load_font(20),load_font(16),load_font(14)
    title=f'遮挡出生布局 {CASES[number-1]["layout"]} · DS LOW · v0.4.1'
    d.text((18,7),title,font=f20,fill='#e9f3f4')
    d.text((width-262,11),'Isaac 原生回放 / 运动学轨迹',font=f14,fill='#aec8d2')
    d.text((18,36),f"A 姿态：{s['posture_target']}   烟雾余量：{s['smoke_remaining']}   锁定：{s['aim'][0]:.2f}s",font=f16,fill='#6dd9f3')
    d.text((width-355,36),f"B 失能：{s['human_disabled_remaining']:.2f}s   锁定：{s['aim'][1]:.2f}s",font=f16,fill='#ffba83')
    index=max(0,s['llm_decision']-1);choice=decisions[min(index,len(decisions)-1)]
    target=choice['params'].get('target',choice['params'].get('return_to',''))
    d.text((18,61),f"{s['t']:05.2f} / 60s    决策 {index+1}/{len(decisions)}：{s.get('exec_skill',choice['skill'])} / {s.get('skill_phase','')}",font=f14,fill='#d2e2e7')
    y=height+94
    label=('四足到点获胜' if result['winner']=='attacker' else '人形防守成功' if result['winner']=='defender' else '同刻成功 · 平局') if terminal else '固定目标 G：进入圆内获胜；标记人形仅失能 1 秒'
    if number==0:label='两次探身与撤回完成' if terminal else f"第{s.get('peek_cycle',0)+1}次探身 · 暴露估计 {s.get('peek_exposure_estimate',0):.2f}s"
    d.text((18,y),label,font=f20,fill='#eaf6ee' if terminal else '#deebed')
    note=f"人形瞄准误差 {abs(s['aim_error_degrees'][1]):.1f}° / 5°；视距不限，射程8m。此镜头仅供审阅。"
    if result.get('requires_schema_recovery'):raise ValueError('This corrected protocol must remain unassisted')
    if result.get('completion_mode')=='metadata_assisted':note='编号修复续跑：原始测试17.18s中断，动作/路线未改。仅作为辅助结果。'
    if result.get('completion_mode')=='network_resumed':note='API响应中断后续跑；已接受决策原样保留，无人工改写动作。'
    d.text((18,y+27),note,font=f14,fill='#9ebbc6')
    return out


def main():
    base=read(R/'source_snapshot/configs/trial_a391_feedback_fix.json');cfg=copy.deepcopy(base)
    cfg['arena']['task_points']=[dict(name='G',center=[10.3,-1.],radius=1.)]
    cfg['collapse']['time']=1e9
    for c in cfg['arena']['covers']:
        if c['name']=='EastStore':c.get('architecture',{})['subtitle']='STATIC / FIXED GOAL'
    cfg['resolution']=[960,720];cfg['render_spp']=12
    scene=R/'native_scene';scene.mkdir(exist_ok=True)
    omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim())
    Courtyard(stage,WORK/'simulators/assets/quadruped-arena/pbr',scene/'labels').build(cfg);lighting(stage,12)
    for stem in ('Target','TargetLabel'):
        p=stage.GetPrimAtPath('/World/Art/'+stem+'0')
        if p.IsValid():UsdGeom.Imageable(p).MakeInvisible()
    actors=[Actor(stage,'/World/Actors/A','go2w',WORK),Actor(stage,'/World/Actors/B','g1',WORK)]
    stars=StunStars(stage)
    goal=curve(stage,'/World/Overlay/Goal',(.1,.9,.4),.06);circle(goal,[10.3,-1.],1.)
    rings=[curve(stage,'/World/Overlay/A',(.05,.65,1.)),curve(stage,'/World/Overlay/B',(1.,.35,.08))]
    beam=curve(stage,'/World/Overlay/Aim',(1.,.72,.15),.025)
    smoke_boundary=curve(stage,'/World/Overlay/SmokeBoundary',(.72,.8,.91),.035)
    smoke_root,smoke_move=cloud(stage)
    camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera.CreateFocalLengthAttr(42.);camera.CreateHorizontalApertureAttr(36.);camera.CreateVerticalApertureAttr(27.)
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(0,-10,34),Gf.Vec3d(0,0,.3),Gf.Vec3d(0,0,1)).GetInverse())
    product=rep.create.render_product(str(camera.GetPath()),tuple(cfg['resolution']));rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product]);rep.orchestrator.set_capture_on_play(False)
    stage.Export(str(scene/'arena.usda'))
    settings=carb.settings.get_settings()
    for number in ([args.only_run] if args.only_run else [1,2,3]):
        folder=R/'stages/01_peek/native_demo' if number==0 else R/f'run_{number}'
        if not (folder/'result.json').exists():raise FileNotFoundError(f'Completed run required: {folder}')
        while not (folder/'result.json').exists():
            recovering=(folder/'recovery_process.json').exists() and read(folder/'recovery_process.json')['status']=='running'
            if (folder/'process.json').exists() and read(folder/'process.json')['status']=='interrupted' and not recovering:
                print('SKIP_INTERRUPTED',number,flush=True);break
            time.sleep(2)
        if not (folder/'result.json').exists():continue
        if (folder/'render_delivery.json').exists() and read(folder/'render_delivery.json').get('status')=='passed':continue
        started=time.monotonic();states=read(folder/'timeline.json');events=read(folder/'events.json');result=read(folder/'result.json');decisions=read(folder/'decisions.json')
        trial=read(folder/'trial_config.json');circle(goal,trial['waypoints']['GOAL'],trial['rules']['goal_radius_m'])
        if number==0:
            center=states[0]['xy'][0];goal.MakeInvisible()
            camera.GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(center[0],center[1]-4,8),Gf.Vec3d(center[0],center[1],.3),Gf.Vec3d(0,0,1)).GetInverse())
        else:
            goal.MakeVisible();camera.GetOrderedXformOps()[0].Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(0,-10,34),Gf.Vec3d(0,0,.3),Gf.Vec3d(0,0,1)).GetInverse())
        hashes={name:sha(folder/name) for name in ('timeline.json','events.json','result.json','decisions.json','trial_config.json')}
        native=folder/'native';native.mkdir(exist_ok=True);(native/'steps').mkdir(exist_ok=True)
        fps=12;final=states[-1]['t'];times=[i/fps for i in range(math.floor(final*fps)+1)]
        if abs(times[-1]-final)>1e-8:times.append(final)
        video_times=list(times);video_keys={round(t,8) for t in times};times=sorted(set(times+[d['simulation_seconds'] for d in decisions]))
        writer=imageio.get_writer(str(native/'intermediate.mp4'),fps=fps,codec='libx264',quality=8,macro_block_size=1,ffmpeg_log_level='error')
        phases=[0.,0.];wheels=np.zeros(4);last=0;frames=[];step_files={}
        try:
            for k,t in enumerate(times):
                s=states[min(round(t/.02),len(states)-1)];delta=max(0,t-last)
                for i in (0,1):
                    phases[i]+=delta*abs(s['speed'][i])*2*math.pi/.65
                    if i==0:wheels+=s['speed'][0]*delta/.089
                    actors[i].update(s['xy'][i],s['yaw'][i],phases[i],abs(s['speed'][i]),0,wheels if i==0 else None,body_height=s['representative_heights'][0] if i==0 else None)
                    circle(rings[i],s['xy'][i],.4)
                stars.update(dict(t=s['t'],xy=s['xy'],hp=[1,1],stun_remaining=[0,s['human_disabled_remaining']]))
                active=[v for v in s['smoke'] if v['active_at']<=s['t']<v['expires_at']]
                smoke_root.MakeInvisible();smoke_boundary.MakeInvisible()
                if active:
                    smoke_root.MakeVisible();smoke_move.Set(Gf.Vec3d(*active[-1]['center'],0.));smoke_boundary.MakeVisible();circle(smoke_boundary,active[-1]['center'],active[-1]['radius'])
                beam.MakeInvisible()
                if any(a>0 and s['visible'][i] for i,a in enumerate(s['aim'])):
                    beam.MakeVisible();points(beam,[(*s['xy'][i],s['representative_heights'][i]) for i in (0,1)])
                for _ in range(12 if k==0 else 4):app.update()
                pixels=rgb.get_data()
                if pixels.size==0 or np.std(pixels[...,:3])<1:raise RuntimeError('Blank native frame')
                framed=hud(Image.fromarray(pixels[...,:3]).convert('RGB'),s,decisions,result,number,k==len(times)-1)
                if round(t,8) in video_keys:
                    writer.append_data(np.asarray(framed));frames.append(dict(time=t,source_state_time=s['t'],index=len(frames)))
                for d in decisions:
                    if abs(t-d['simulation_seconds'])<1e-8:
                        name=f"steps/{d['observation_id']:03d}.png";framed.save(native/name);step_files[str(d['observation_id'])]=dict(file=name,time=t)
                if k==0:framed.save(native/'preview.png')
                if k==len(times)-1:framed.save(native/'final.png')
                if k%30==0:print('NATIVE_FRAME',number,k+1,len(times),round(t,2),round(time.monotonic()-started,1),flush=True)
                last=t
        finally:writer.close()
        ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
        subprocess.run([ffmpeg,'-nostdin','-y','-v','error','-i',str(native/'intermediate.mp4'),'-filter_complex','fps=12,scale=800:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=sierra2_4a','-loop','0',str(folder/'demo.gif')],check=True)
        with Image.open(folder/'demo.gif') as gif:
            duration=0
            for j in range(gif.n_frames):gif.seek(j);duration+=gif.info.get('duration',0)
            gif_frames=gif.n_frames
        assert abs(duration/1000-len(video_times)/fps)<.3
        assert len(step_files)==len(decisions)
        assert all(sha(folder/name)==value for name,value in hashes.items())
        dump(native/'frame_index.json',dict(fps=fps,frames=frames));dump(native/'step_frames.json',step_files)
        dump(folder/'render_delivery.json',dict(status='passed',renderer=settings.get('/rtx/rendermode'),native_isaac=True,
             policy_rerun=False,full_match=True,fps=fps,frame_count=len(video_times),gif_frames=gif_frames,gif_duration_seconds=duration/1000,
             match_seconds=final,source_sha256=hashes,decision_screenshots=len(step_files),wall_seconds=time.monotonic()-started,
             smoke_visual='Native transparent cloud geometry and exact radius outline; spectator indication, not fluid simulation or model input.'))
        (native/'intermediate.mp4').unlink()
        print('NATIVE_COMPLETE',number,round(time.monotonic()-started,1),flush=True)
    rgb.detach([product.path]);product.destroy()


try:main()
finally:app.close(wait_for_replicator=False)
