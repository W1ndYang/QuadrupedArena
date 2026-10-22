"""Isaac native renderer for the CPU-refereed R0 trial. No physics stepping."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import traceback
import numpy as np
from isaaclab.app import AppLauncher

from arena_paths import PROJECT as project, workspace_root
workspace=workspace_root()
parser=argparse.ArgumentParser()
for name in ('demo','rules-v3','preview','adaptive','rotating-targets','industrial'):parser.add_argument('--'+name,action='store_true')
parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--replay-dir',type=Path)
parser.add_argument('--demo-config',type=Path)
AppLauncher.add_app_launcher_args(parser);args=parser.parse_args();args.enable_cameras=True
if args.rotating_targets or args.replay_dir:args.adaptive=True
if args.demo_config is None:args.demo_config=project/('configs/trial_rotating.json' if args.rotating_targets else 'configs/trial_adaptive.json' if args.adaptive else 'configs/trial_v3.json')
app=AppLauncher(args).app
import carb
import omni.usd
import omni.replicator.core as rep
import isaaclab.sim as sim_utils
from pxr import Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade, Gf
from PIL import Image
import imageio.v2 as imageio
import imageio_ffmpeg
from trial_motion import build
if args.adaptive:
    from adaptive_motion import build, load_config
from trial_rules import covers_at, aim_valid
from tactical_logic import block
from demo_actors import Actor
from beauty import prepare
from trial_visuals import panel, EYE, TARGET, FOCAL, APERTURE
if args.adaptive:
    # Wider framing keeps the full spawn positions and long flanking paths visible.
    # Mutating this shared camera vector also updates HUD projection coordinates.
    EYE[:]=[-10.5,9.5,14.]
if args.industrial:
    import trial_visuals
    EYE[:]=[-12.,-14.,20.]
    TARGET[:]=[0.,.8,.35]
    FOCAL=27.;trial_visuals.FOCAL=FOCAL


def box(path,size,position,color):
    cfg=sim_utils.CuboidCfg(size=tuple(size),collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(color)),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.7))
    cfg.func(path,cfg,translation=tuple(position))


def curve(stage,path,color,width=.025):
    item=UsdGeom.BasisCurves.Define(stage,path);item.CreateTypeAttr('linear')
    item.CreateCurveVertexCountsAttr([2]);item.CreatePointsAttr([(0,0,0),(.001,0,0)])
    item.CreateWidthsAttr([width]);item.SetWidthsInterpolation('constant')
    item.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    return item


def points(item,pts):
    item.GetPointsAttr().Set(pts);item.GetCurveVertexCountsAttr().Set([len(pts)])


def main():
    started=time.monotonic();cfg=load_config(args.demo_config) if args.adaptive else json.loads(args.demo_config.read_text())
    replay_manifest=None
    if args.replay_dir:
        from replay_io import load_replay
        cfg,states,events,validation,replay_manifest=load_replay(args.replay_dir)
    else:
        states,events,validation=build(cfg)
    for name,data in (('trial_config.json',cfg),('timeline.json',states),('events.json',events)):
        (args.run_dir/name).write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    if 'llm_decision' in states[-1]:cfg=dict(cfg,_presentation_decisions=states[-1]['llm_decision'])
    presentation_decisions=[]
    if args.replay_dir and (args.replay_dir/'protocol.json').exists():
        protocol=json.loads((args.replay_dir/'protocol.json').read_text())
        if protocol.get('mode') in ('deepseek_partial_observation','qwen_partial_observation','assistant_assessment_partial_observation','semantic_skills_short_intent','assistant_partial_observation'):
            cfg=dict(cfg,_presentation_partial_vision=True,_presentation_planner=protocol.get('planner','LLM'),_presentation_effort=protocol.get('reasoning_effort','low'))
            if (args.replay_dir/'decisions.json').exists():presentation_decisions=json.loads((args.replay_dir/'decisions.json').read_text())
    omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim())
    scene_dir=workspace/'simulators/scenes/quadruped-arena'/args.run_dir.name;scene_dir.mkdir(parents=True,exist_ok=True)
    if args.industrial:
        from arena_art import Courtyard, lighting
        Courtyard(stage,workspace/'simulators/assets/quadruped-arena/pbr',scene_dir/'labels').build(cfg)
        lighting(stage,cfg['render_spp'])
        for i in range(len(cfg['arena']['task_points'])):
            for stem in ('Target','TargetLabel'):
                UsdGeom.Imageable(stage.GetPrimAtPath('/World/Art/'+stem+str(i))).MakeInvisible()
    else:
        box('/World/Floor',cfg['arena']['floor_size'],(0,0,-.1),(.4,.42,.45))
        for c in cfg['arena']['covers']:
            box('/World/Covers/'+c['name'],c['size'],(*c['center'],c['size'][2]/2),(.3,.45,.6) if c['size'][2]>1 else (.8,.6,.3))
        light=UsdLux.DomeLight.Define(stage,'/World/Light');light.CreateIntensityAttr(650);light.CreateColorAttr((.85,.91,1))
        prepare(stage,scene_dir/'textures')
    settings=carb.settings.get_settings()
    settings.set('/rtx/pathtracing/totalSpp',cfg['render_spp']);settings.set('/rtx/pathtracing/clampSpp',cfg['render_spp'])
    wall=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Covers/'+cfg['collapse']['wall']))
    wall.ClearXformOpOrder();wall_op=wall.AddTransformOp()
    original=next(c for c in cfg['arena']['covers'] if c['name']==cfg['collapse']['wall'])
    pivot=np.asarray(cfg['collapse']['pivot']);center=np.array([*original['center'],original['size'][2]/2])
    def rotate_wall(u):
        theta=math.radians(cfg['collapse']['angle_degrees'])*u
        rotation=np.array([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
        matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=pivot+rotation@(center-pivot)
        wall_op.Set(Gf.Matrix4d(matrix.T.tolist()))
    def check_wall(u,expected):
        rotate_wall(u)
        if args.industrial:
            collider=stage.GetPrimAtPath(str(wall.GetPath())+'/Collider')
            matrix=UsdGeom.Xformable(collider).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            corners=np.array([matrix.Transform(Gf.Vec3d(x,y,z)) for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
            measured=dict(center=((corners.min(0)+corners.max(0))/2).tolist(),size=np.ptp(corners,axis=0).tolist())
        else:
            bbox=UsdGeom.BBoxCache(Usd.TimeCode.Default(),[UsdGeom.Tokens.default_]).ComputeWorldBound(wall.GetPrim()).ComputeAlignedRange()
            measured=dict(center=list(bbox.GetMidpoint()),size=list(bbox.GetSize()))
        assert np.allclose(measured['center'],[*expected['center'],expected['size'][2]/2],atol=1e-5),measured
        assert np.allclose(measured['size'],expected['size'],atol=1e-5),measured
        assert any(p.HasAPI(UsdPhysics.CollisionAPI) for p in Usd.PrimRange(wall.GetPrim()))
        return measured
    validation['usd_wall_before']=check_wall(0,original)
    stage.Export(str(scene_dir/'arena_before.usda'))
    validation['usd_wall_after']=check_wall(1,cfg['collapse']['debris'])
    stage.Export(str(scene_dir/'arena_after.usda'));rotate_wall(0)
    assert not stage.GetPrimAtPath('/World/Terrain').IsValid()
    actors=[Actor(stage,'/World/Actors/A','go2w',workspace),Actor(stage,'/World/Actors/B','g1',workspace)]
    colors=[(.04,.65,1.),(1.,.35,.06)]
    rings=[curve(stage,f'/World/Overlay/Ring{i}',c,.025) for i,c in enumerate(colors)]
    cones=[curve(stage,f'/World/Overlay/Fov{i}',c,.012) for i,c in enumerate(colors)]
    trails=[curve(stage,f'/World/Overlay/Trail{i}',c,.018) for i,c in enumerate(colors)]
    targets=[curve(stage,f'/World/Overlay/P{i}',(.1,.9,.4),.05) for i in range(2)]
    beam=curve(stage,'/World/Overlay/RefereeRay',(1,.7,.1),.022)
    if args.adaptive:
        memory_ring=curve(stage,'/World/Overlay/LastSeen',(1.,.6,.1),.025)
        planned_route=curve(stage,'/World/Overlay/PlannedRoute',(.05,.9,1.),.018)
        sensor_cone=curve(stage,'/World/Overlay/HumanSensor',(1.,.65,.3),.012)
    border=curve(stage,'/World/Overlay/Boundary',(.7,.75,.8),.025)
    points(border,[(-6,-4,.015),(6,-4,.015),(6,4,.015),(-6,4,.015),(-6,-4,.015)])
    camera=UsdGeom.Camera.Define(stage,'/World/Camera');camera.CreateFocalLengthAttr(FOCAL)
    camera.CreateHorizontalApertureAttr(APERTURE);camera.CreateVerticalApertureAttr(APERTURE*cfg['resolution'][1]/cfg['resolution'][0])
    camera.AddTransformOp().Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*EYE),Gf.Vec3d(*TARGET),Gf.Vec3d(0,0,1)).GetInverse())
    product=rep.create.render_product(str(camera.GetPath()),tuple(cfg['resolution']))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product]);rep.orchestrator.set_capture_on_play(False)
    final_t=states[-1]['t'];ct=cfg['collapse']['time'];fps=cfg['fps']
    if args.preview:
        times=sorted(set([0,3,12.4,16,ct-.1,ct,ct+1,min(ct+5,final_t),final_t]))
        if args.adaptive:
            times=sorted(set(times+[e['time'] for e in events if e['type'] in ('quadruped_decision','human_decision') and e.get('state') in ('EVADE','SEARCH_LAST_SEEN','WINDOW_ADVANCE')]))
    else:
        # Keep every integer 2s decision boundary exact even if elimination
        # ends a match between video ticks; append that final partial tick.
        times=[i/fps for i in range(math.floor(final_t*fps+1e-8)+1)]
        if abs(times[-1]-final_t)>1e-8:times.append(final_t)
    frames=[]
    for t in times:
        if t>final_t:continue
        if t>=ct and not any(f[1] is not None for f in frames):
            for u in ([0,.5,1] if args.preview else np.linspace(0,1,round(cfg['collapse']['display_seconds']*fps))):frames.append((ct,float(u)))
        frames.append((t,None))
    writer=imageio.get_writer(str(args.run_dir/'demo.mp4'),fps=fps,codec='libx264',quality=8,macro_block_size=1,ffmpeg_log_level='error')
    count=0;phase=[0,0];last_t=0;frame_index=[]
    try:
        for index,(t,fall) in enumerate(frames):
            s=states[min(round(t/cfg['dt']),len(states)-1)];rotate_wall(fall if fall is not None else float(s['collapsed']))
            for i,actor in enumerate(actors):
                xy=s['xy'][i];phase[i]+=max(0,t-last_t)*s['speed'][i]*2*math.pi/.65
                actor.update(xy,s['yaw'][i],phase[i],s['speed'][i],0,s['wheels'] if i==0 else None)
                if s['hp'][i]==0 and i==1:UsdGeom.Imageable(stage.GetPrimAtPath('/World/Actors/B')).MakeInvisible()
                points(rings[i],[(xy[0]+.4*math.cos(a),xy[1]+.4*math.sin(a),.025) for a in np.linspace(0,2*math.pi,49)])
                pts=[(*xy,.03)]+[(xy[0]+1.1*math.cos(a),xy[1]+1.1*math.sin(a),.03) for a in np.linspace(s['yaw'][i]-math.pi/6,s['yaw'][i]+math.pi/6,17)]+[(*xy,.03)]
                points(cones[i],pts)
                history=states[max(0,round(t/cfg['dt'])-150):round(t/cfg['dt'])+1:10]
                pts=[(*h['xy'][i],.02) for h in history]
                points(trails[i],pts if len(pts)>1 else pts*2)
                if i==1 and s['hp'][i]==0:
                    for item in (rings[i],cones[i],trails[i]):item.MakeInvisible()
            for i,p in enumerate(cfg['arena']['task_points']):
                if cfg.get('target_rotation',{}).get('enabled') and not s['eligible'][i]:
                    targets[i].MakeInvisible();continue
                targets[i].MakeVisible()
                points(targets[i],[(p['center'][0]+p['radius']*math.cos(a),p['center'][1]+p['radius']*math.sin(a),.025) for a in np.linspace(0,2*math.pi,65)])
                targets[i].GetDisplayColorAttr().Set([Gf.Vec3f(*((.9,.05,.05) if not s['eligible'][i] else (.08,.9,.3)))])
            start=(*s['xy'][1],cfg['representative_heights'][1]);end=(*s['xy'][0],cfg['representative_heights'][0])
            hit=block(start,end,covers_at(cfg,t));beam.MakeInvisible()
            if min(s['hp'])>0 and fall is None:
                if hit and not args.adaptive:
                    beam.MakeVisible();points(beam,[start,hit[2]]);beam.GetDisplayColorAttr().Set([Gf.Vec3f(1,.08,.05)])
                elif any(s['aim']):
                    if not hit:
                        beam.MakeVisible();points(beam,[start,end]);beam.GetDisplayColorAttr().Set([Gf.Vec3f(1,.75,.1)])
            if args.adaptive:
                memory_ring.MakeInvisible();planned_route.MakeInvisible()
                if s['human_memory'] is not None and not s['human_visible'] and s['hp'][1]>0:
                    mx,my=s['human_memory'];memory_ring.MakeVisible()
                    points(memory_ring,[(mx+.25*math.cos(a),my+.25*math.sin(a),.035) for a in np.linspace(0,2*math.pi,33)])
                if s['a_route'] and s['hp'][0]>0:
                    planned_route.MakeVisible();points(planned_route,[(*s['xy'][0],.035)]+[(*p,.035) for p in s['a_route']])
                xy=s['xy'][1];half=math.radians(cfg['policy']['human_sensor_half_angle_degrees'])
                points(sensor_cone,[(*xy,.022)]+[(xy[0]+1.6*math.cos(a),xy[1]+1.6*math.sin(a),.022) for a in np.linspace(s['yaw'][1]-half,s['yaw'][1]+half,25)]+[(*xy,.022)])
                if s['hp'][1]==0:sensor_cone.MakeInvisible()
            for _ in range(12 if index==0 else 4):app.update()
            pixels=rgb.get_data()
            if pixels.size==0 or np.std(pixels[...,:3])<1:raise RuntimeError(f'Blank frame {t}')
            display_state=s
            if presentation_decisions and s.get('llm_decision',0)>0:
                chosen=presentation_decisions[min(s['llm_decision']-1,len(presentation_decisions)-1)]
                display_state=dict(s,llm_skill=chosen.get('skill',s.get('llm_skill','')))
            raw=Image.fromarray(pixels[...,:3]).convert('RGB');framed=panel(raw,display_state,events,cfg,collapse=fall)
            if index==0:
                intro=panel(raw,display_state,events,cfg,intro=True);intro.save(args.run_dir/'intro.png')
                for _ in range(1 if args.preview else cfg['intro_seconds']*fps):writer.append_data(np.asarray(intro));count+=1
            if fall is None:frame_index.append(dict(simulation_time=t,video_frame_index=count))
            writer.append_data(np.asarray(framed));count+=1
            if index%40==0 or args.preview or fall is not None and abs(fall-.5)<.03:
                suffix=f'{index:04d}_{t:05.2f}'
                raw.save(args.run_dir/f'raw_{suffix}.png');framed.save(args.run_dir/f'frame_{suffix}.png')
                print(f'TRIAL_FRAME {index+1}/{len(frames)} t={t:.2f} elapsed={time.monotonic()-started:.1f}s',flush=True)
            if index==len(frames)-1:
                poster=panel(raw,display_state,events,cfg,outro=True);poster.save(args.run_dir/'poster.png')
                for _ in range(1 if args.preview else cfg['outro_seconds']*fps):writer.append_data(np.asarray(poster));count+=1
            last_t=t
    finally:
        writer.close();rgb.detach([product.path]);product.destroy()
    stage.Export(str(scene_dir/'demo_final_pose.usda'))
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe()
    subprocess.run([ffmpeg,'-nostdin','-y','-v','error','-i',str(args.run_dir/'demo.mp4'),'-filter_complex',
        'fps=10,scale=800:-2:flags=lanczos,split[s0][s1];[s0]palettegen=max_colors=192[p];[s1][p]paletteuse=dither=sierra2_4a','-loop','0',str(args.run_dir/'demo.gif')],check=True,timeout=180)
    reader=imageio.get_reader(str(args.run_dir/'demo.mp4'));actual=reader.count_frames();reader.close();assert actual==count
    with Image.open(args.run_dir/'demo.gif') as gif:
        milliseconds=0
        for i in range(gif.n_frames):gif.seek(i);milliseconds+=gif.info.get('duration',0)
    assert abs(milliseconds/1000-count/fps)<.3
    (args.run_dir/'frame_index.json').write_text(json.dumps(dict(fps=fps,frames=count,regular_frames=frame_index),indent=2)+'\n')
    result=dict(status='passed',kind='r0_rules_preview' if args.preview else 'r0_rules_trial',renderer=settings.get('/rtx/rendermode'),
        target_spp=cfg['render_spp'],training=False,dynamics_validated=False,frames=count,fps=fps,duration_seconds=count/fps,
        match_seconds=final_t,wall_seconds=time.monotonic()-started,validation=validation,scene_directory=str(scene_dir),
        outputs={n:str(args.run_dir/n) for n in ('demo.gif','demo.mp4','poster.png','events.json','timeline.json','trial_config.json')},
        limitations=['Kinematic actors, not validated locomotion/contact dynamics','Fixed-pose collapse with cinematic interpolation, not rigid-body fracture',
            'Known-map A* baseline, humanoid attack-priority tree; no learned tactical policy','Representative-point LOS against wall boxes, no mesh raycast',
            'Single fixed seed is not balance validation','HUD and referee helper geometry are presentation overlays'])
    if args.adaptive:
        result['variant']='asymmetric_visibility_risk_policy_2'
        result['limitations'][2]='A risk-aware heuristic; B only receives visible positions and last-seen memory; no training'
    if cfg.get('target_rotation',{}).get('enabled'):
        result['variant']='rotating_targets_fast_human_v04'
    if replay_manifest:
        result['replay']=replay_manifest
        result['variant']='llm_closed_loop_replay' if 'llm_decision' in states[-1] else 'saved_match_replay'
        result['limitations'][2]='Recorded external LLM skill choices; no policy or referee rerun during rendering' if 'llm_decision' in states[-1] else 'Saved trajectory replay; no policy rerun'
    result['industrial_art']=args.industrial
    (args.run_dir/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print('TRIAL_RESULT '+json.dumps(result,ensure_ascii=False),flush=True)


try:main()
except BaseException:
    error=traceback.format_exc();(args.run_dir/'error.txt').write_text(error)
    (args.run_dir/'result.json').write_text(json.dumps(dict(status='failed',error=error))+'\n')
    print(error,flush=True);raise
finally:app.close(wait_for_replicator=False)
