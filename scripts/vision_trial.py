"""Live kinematic referee + Isaac RGB sensor, externally paused decisions.

Private match state stays in memory until termination. Never calls Match.observe.
Isaac renders transforms only: no dynamics and no tactical overlays.
"""
import argparse
import datetime
import math
from pathlib import Path
import time
import traceback
import numpy as np
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser()
parser.add_argument('--vision-trial',action='store_true')
parser.add_argument('--vision-smoke',action='store_true')
parser.add_argument('--semantic-skills',action='store_true')
parser.add_argument('--deepseek',action='store_true',help='Official Flash API, configured effort; paid requests')
parser.add_argument('--qwen-vision',choices=['max','min'],help='Official Qwen3.6-27B with public RGB; paid requests')
parser.add_argument('--assessment',action='store_true',help='External assistant uses the same four-field assessment schema')
parser.add_argument('--smoke-skill',choices=['CAPTURE_POINT','OBSERVE'],default='CAPTURE_POINT')
parser.add_argument('--g1b1-visible-spawn',action='store_true',
                    help='G1-B.1 diagnostic only: start A at (4.8,-1.0), facing north; default match spawn is unchanged')
parser.add_argument('--geometry-diagnostic',action='store_true',
                    help='Capture two formal-scene geometry diagnostics and exit; no LLM, actions JSON or Skill commit')
parser.add_argument('--geometry-interface-probe',action='store_true',
                    help='Read-only formal-scene segmentation/depth interface probe; no Skill commit or asset writeback')
parser.add_argument('--diagnostic-output-dir',type=Path,
                    help='Diagnostic artifacts directory (default: this run/geometry_diagnostic)')
parser.add_argument('--run-dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.enable_cameras=True
from formal_geometry_diagnostic import diagnostic_mode_error
mode_error=diagnostic_mode_error(args)
if mode_error:parser.error(mode_error)
from geometry_interface_probe import interface_probe_mode_error
probe_mode_error=interface_probe_mode_error(args)
if probe_mode_error:parser.error(probe_mode_error)
if args.diagnostic_output_dir and not args.geometry_diagnostic:parser.error('--diagnostic-output-dir requires --geometry-diagnostic')
if (args.deepseek or args.qwen_vision or args.assessment) and (not args.semantic_skills or args.vision_smoke):parser.error('Planner modes require semantic skills and exclude fixed-command smoke')
if sum((bool(args.deepseek),bool(args.qwen_vision),args.assessment))>1:parser.error('Choose one planner mode')
if args.g1b1_visible_spawn and (not args.vision_trial or not args.semantic_skills or args.vision_smoke or args.deepseek or args.qwen_vision or args.assessment):
    parser.error('--g1b1-visible-spawn requires a manual semantic vision trial without smoke or planner modes')
app=AppLauncher(args).app
import omni.usd
import omni.replicator.core as rep
from pxr import UsdGeom,Gf
from PIL import Image
from adaptive_motion import load_config
from llm_trial import PROJECT,dump,export
from vision_core import VisionMatch,CameraRig,public_packet,validate_request
from arena_art import Courtyard,lighting
from demo_actors import Actor


def main():
    from arena_paths import workspace_root
    root=args.run_dir;workspace=workspace_root()
    if not (args.geometry_diagnostic or args.geometry_interface_probe):
        for name in ('frames','observations','actions','decisions'):(root/name).mkdir()
    cfg=load_config(PROJECT/'configs/trial_rotating.json');cfg['seed']=2026102107
    if args.vision_smoke:cfg['duration']=2.
    if args.g1b1_visible_spawn:
        # Only this explicit G1-B.1 diagnostic run moves A; keep B and the arena intact.
        cfg['start_a']=[4.8,-1.0]
    match=VisionMatch(cfg,initial_a_yaw=math.pi/2 if args.g1b1_visible_spawn else 0.);rig=CameraRig()
    engine=None;execution=[];api_planner=None
    if args.semantic_skills:
        from semantic_skills import SemanticSkills
        engine=SemanticSkills(cfg)
    if not (args.geometry_diagnostic or args.geometry_interface_probe):dump(root/'protocol.json',dict(mode='engineering_smoke' if args.vision_smoke else 'assistant_partial_observation',
        engine='Python kinematics/referee; Isaac online RGB only; no PhysX steps',
        decision_period_seconds=2,frame_interval_seconds=.5,horizontal_fov_degrees=90,
        camera_mount=dict(forward=.36,height=.60,pitch_degrees=-5),camera_pan_limits=[-90,90],camera_max_rate_degrees=120,
        timing='Observation ready to action accepted: includes model reasoning, reading images and tool overhead, not pure inference',
        tokens=None,token_status='Not exposed by assistant session; no estimate',
        camera_motion_consumes_simulation_time=True,initial_frame_only=True,
        hidden_state='Kept in renderer/referee memory; exported after match only',
        action_tracking='AIM and facing enemy rejected; collision safety guard only, no tactical rerouting',
        seed=cfg['seed'],known_map=cfg['arena'],known_collapse=cfg['collapse'],
        opponent_initial_xy=cfg['start_b']))
    if engine:
        import json
        protocol=json.loads((root/'protocol.json').read_text())
        protocol.update(mode='semantic_skills_engineering_smoke' if args.vision_smoke else 'semantic_skills_short_intent',
            skill_api_version='1.1_post_first_trial_fixes',
            intent_schema=['observation_id','skill','params','reason'],skills=['MOVE_TO','TAKE_COVER','AIM_AT_TRACK','CAPTURE_POINT','OBSERVE'],
            perception='LLM foot-pixel annotation of raw RGB -> calibrated flat-ground estimate. Not automatic tracking; uncertainty grows; memory expires at 4s; attack estimate expires at 2s.',
            execution='Known-map risk-weighted routes; local goal selection and turn/acceleration ETA. Never access current opponent truth.',
            differences_from_previous='Semantic skills + short output + visual-estimate-only attack gating + programmatic camera. Same match rules, seed, map, 2s decision / 0.5s RGB cadence. Not a pure model-speed A/B.',
            online_reason_limit_characters=100,api_model_change=False)
        dump(root/'protocol.json',protocol)
    if args.deepseek or args.qwen_vision:
        from deepseek_trial import motion_request
        if args.deepseek:
            from deepseek_trial import Planner as APIPlanner,settings
            from deepseek_client import load_key
            profile=settings()
        else:
            from qwen_vision import Planner as APIPlanner,settings,load_key
            profile=settings(args.qwen_vision)
        api_planner=APIPlanner(profile,cfg,load_key(),root)
        protocol.update(mode='qwen_partial_observation' if args.qwen_vision else 'deepseek_partial_observation',planner=profile['model'],provider=profile['provider'],
            intent_schema=profile['decision_fields'],online_reason_limit_characters=0,decision_only_output=not profile.get('explain_decisions',False),
            assessment_output=profile.get('explain_decisions',False),
            reasoning_effort=profile['reasoning_effort'],thinking=profile['thinking'],api_model_change=True,
            token_status='Provider usage when available; null otherwise',
            timing='Observation-ready-to-accepted plus API wall/first-output/first-content and skill compile separately',
            context='Public rules and initial location; current 1..5 RGB frames; last 4 compact actions; no historical developer conversation',
            max_calls=20,automatic_retries=0,automatic_strategy_fallback=False)
        protocol['json_transport_normalization']='remove one complete JSON code fence only; preserve raw response; no JSON/action repair' if profile.get('allow_single_json_fence') else 'strict JSON text'
        dump(root/'protocol.json',protocol)
    if args.assessment:
        from deepseek_trial import system_message,motion_request,validate_model_output
        protocol.update(mode='assistant_assessment_partial_observation',planner='current_assistant_session',
            provider='interactive_assistant',intent_schema=['observation_id','skill','params','assessment'],
            assessment_output=True,reasoning_effort='session_setting_not_api_controlled',
            context='Current assistant inherits development conversation; not a fresh blinded model context',
            online_reason_limit_characters=0)
        dump(root/'protocol.json',protocol)
        (root/'planner_context.txt').write_text(system_message(cfg,{'explain_decisions':True})['content'])
    def observe(window):
        packet=public_packet(len(match.decisions),match.states[-1],window,cfg['start_b'])
        if engine:packet['skills']=engine.public(match.t)
        return packet
    omni.usd.get_context().new_stage();stage=omni.usd.get_context().get_stage()
    UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    stage.SetDefaultPrim(UsdGeom.Xform.Define(stage,'/World').GetPrim())
    Courtyard(stage,workspace/'simulators/assets/quadruped-arena/pbr',root/'labels').build(cfg)
    lighting(stage,8)
    actors=[Actor(stage,'/World/Actors/A','go2w',workspace),Actor(stage,'/World/Actors/B','g1',workspace)]
    wall=UsdGeom.Xformable(stage.GetPrimAtPath('/World/Covers/'+cfg['collapse']['wall']))
    wall.ClearXformOpOrder();wallop=wall.AddTransformOp()
    original=next(c for c in cfg['arena']['covers'] if c['name']==cfg['collapse']['wall'])
    center=np.array([*original['center'],original['size'][2]/2]);pivot=np.asarray(cfg['collapse']['pivot'])
    camera=UsdGeom.Camera.Define(stage,'/World/OnboardCamera')
    camera.CreateFocalLengthAttr(12);camera.CreateHorizontalApertureAttr(24);camera.CreateVerticalApertureAttr(13.5)
    camera.CreateClippingRangeAttr(Gf.Vec2f(.04,100));camop=camera.AddTransformOp()
    product=rep.create.render_product(str(camera.GetPath()),(768,432))
    rgb=rep.AnnotatorRegistry.get_annotator('rgb');rgb.attach([product]);rep.orchestrator.set_capture_on_play(False)
    frames=[]
    def pose_scene(s):
        for i,actor in enumerate(actors):
            actor.update(s['xy'][i],s['yaw'][i],s['t']*s['speed'][i]*2*math.pi/.65,s['speed'][i],0,s['wheels'] if i==0 else None)
            if i==1 and s['hp'][1]==0:UsdGeom.Imageable(stage.GetPrimAtPath('/World/Actors/B')).MakeInvisible()
        for i in range(len(cfg['arena']['task_points'])):
            for stem in ('Target','TargetLabel'):
                item=UsdGeom.Imageable(stage.GetPrimAtPath('/World/Art/'+stem+str(i)))
                if i==s['active_target']:item.MakeVisible()
                else:item.MakeInvisible()
        theta=math.radians(cfg['collapse']['angle_degrees'])*float(s['collapsed'])
        rotation=np.array([[1,0,0],[0,math.cos(theta),-math.sin(theta)],[0,math.sin(theta),math.cos(theta)]])
        matrix=np.eye(4);matrix[:3,:3]=rotation;matrix[:3,3]=pivot+rotation@(center-pivot)
        wallop.Set(Gf.Matrix4d(matrix.T.tolist()))
        yaw=s['yaw'][0];heading=yaw+math.radians(rig.pan)
        eye=np.array([s['xy'][0][0]+.36*math.cos(yaw),s['xy'][0][1]+.36*math.sin(yaw),.60])
        direction=np.array([math.cos(heading)*math.cos(math.radians(5)),math.sin(heading)*math.cos(math.radians(5)),-math.sin(math.radians(5))])
        camop.Set(Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye),Gf.Vec3d(*(eye+direction)),Gf.Vec3d(0,0,1)).GetInverse())
        return yaw,heading,eye
    if args.geometry_diagnostic:
        from formal_geometry_diagnostic import run as run_geometry_diagnostic
        output=args.diagnostic_output_dir or root/'geometry_diagnostic'
        try:
            summary=run_geometry_diagnostic(output,match.states[-1],cfg,stage,camera,rgb,app,pose_scene)
        except Exception:
            print(f'Geometry diagnostic failed; partial artifacts: {output}',flush=True)
            raise
        dump(root/'result.json',dict(status='passed',mode='formal_geometry_diagnostic',
            diagnostic_status=summary['status'],runtime_verified=False,
            diagnostic_output_dir=str(Path(output).expanduser().resolve()),
            diagnostic_bundle=str(Path(output).expanduser().resolve()/'diagnostic_bundle.zip')))
        print(f'Geometry diagnostic output: {Path(output).expanduser().resolve()}',flush=True)
        print(f'Geometry diagnostic bundle: {Path(output).expanduser().resolve()/"diagnostic_bundle.zip"}',flush=True)
        print(f'Geometry diagnostic status: {summary["status"]}; runtime_verified=false',flush=True)
        return
    if args.geometry_interface_probe:
        from geometry_interface_probe import run as run_interface_probe
        output=root/'geometry_interface_probe'
        try:
            summary=run_interface_probe(output,match.states[-1],cfg,stage,camera,product,rgb,rep,app,pose_scene)
        except Exception:
            print(f'Interface probe failed; partial artifacts: {output}',flush=True)
            raise
        dump(root/'result.json',dict(status='passed',mode='geometry_interface_probe',
            probe_status=summary['probe_status'],runtime_verified=False,
            same_frame_verified=False,probe_output_dir=str(output.resolve()),
            diagnostic_bundle=str(output.resolve()/'diagnostic_bundle.zip')))
        print(f'Interface probe output: {output.resolve()}',flush=True)
        print(f'Interface probe bundle: {output.resolve()/"diagnostic_bundle.zip"}',flush=True)
        print(f'Interface probe status: {summary["probe_status"]}; runtime_verified=false',flush=True)
        return
    def capture():
        s=match.states[-1]
        yaw,heading,eye=pose_scene(s)
        for _ in range(12 if not frames else 6):app.update()
        pixels=np.asarray(rgb.get_data())
        if pixels.shape[:2]!=(432,768) or pixels[...,:3].std()<1:raise RuntimeError('Missing or blank onboard image')
        name=f'frames/{len(frames):03d}_{s["t"]:05.2f}.png';Image.fromarray(pixels[...,:3]).save(root/name)
        frame=dict(file=name,simulation_seconds=s['t'],own_xy=s['xy'][0],body_yaw_degrees=math.degrees(yaw),
            camera_pan_degrees=rig.pan,camera_world_yaw_degrees=math.degrees(heading),camera_eye=eye.tolist(),horizontal_fov_degrees=90)
        frames.append(frame);dump(root/'frame_index.json',frames)
        return frame
    window=[capture()]
    while not match.ref.finished:
        index=len(match.decisions);packet=observe(window)
        dump(root/'observations'/f'{index:03d}.json',packet)
        ready=time.monotonic();readyutc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        dump(root/'pending.json',dict(observation_id=index,simulation_seconds=match.t,ready_utc=readyutc))
        print(f'PUBLIC READY {index} t={match.t:.2f}',flush=True)
        path=root/'actions'/f'{index:03d}.json'
        api_timing=None
        if api_planner:
            try:request,api_timing=api_planner.decide(packet,engine.validate)
            except Exception:
                # Export only after the trial aborts; never fed back into planner.
                export(match,root)
                dump(root/'skill_execution.json',execution)
                dump(root/'interruption.json',dict(status='aborted_not_finished',simulation_seconds=match.t,observation_id=index,automatic_retry=False))
                raise
            dump(path,request)
        if args.vision_smoke:
            if engine:dump(path,dict(observation_id=index,skill=args.smoke_skill,params=dict(direction='north') if args.smoke_skill=='OBSERVE' else dict(target='active'),reason='固定指令的工程预检，不是正式LLM比赛。'))
            else:dump(path,dict(observation_id=index,action=dict(skill='HOLD',attack=False),camera=dict(mode='sweep',center_degrees=0,amplitude_degrees=60),
                evidence='Engineering camera test',choice_summary='Stationary scan',alternatives_summary='Not a policy evaluation',expected_result='Four timed camera frames'))
        while not path.exists():
            if time.monotonic()-ready>1200:raise TimeoutError('No external decision within 1200 seconds')
            time.sleep(.1)
        import json
        request=json.loads(path.read_text())
        if args.assessment:validate_model_output(request,packet,{'explain_decisions':True},engine.validate)
        motion=motion_request(request) if api_planner or args.assessment else request
        if engine:engine.validate(motion,packet)
        else:route=validate_request(request,match)
        elapsed=time.monotonic()-ready
        compiled=None;compile_started=time.monotonic()
        if engine:
            action,route,camera_command,compiled=engine.commit(motion,packet)
            match.compile(action)  # Schema/reachability check; execute the skill's risk-aware route.
            previous=np.asarray(packet['self_state']['xy'])
            for p in route:
                if not engine.base.segment(previous,p):raise ValueError('Unsafe compiled skill route')
                previous=p
        else:action=request['action'];camera_command=request['camera']
        decision=dict(**request,simulation_seconds=match.t,observation_file=f'observations/{index:03d}.json',
            observation_ready_utc=readyutc,decision_accepted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),decision_wait_seconds=elapsed,tokens=None)
        if api_timing:decision.update(tokens=api_timing['tokens'],api=api_timing)
        if engine:decision.update(compiled_action=action.copy(),compiled_skill=compiled,skill_compile_seconds=time.monotonic()-compile_started)
        match.decisions.append(decision);dump(root/'decisions'/f'{index:03d}.json',decision)
        match.apply(action,route);window=[frames[-1]]
        for step in range(1,101):
            if engine:
                own_packet=observe([]);interruption=engine.tick(own_packet)
                if interruption:match.apply(*interruption)
                else:
                    for key in ('attack','facing'):match.active[key]=engine.action[key]
                camera_command=engine.camera(own_packet)
            match.advance(cfg['dt']);rig.step(camera_command,step*cfg['dt'],cfg['dt'])
            if engine and abs(match.t-cfg['collapse']['time'])<1e-8 and not match.ref.finished:
                action,route,camera_command,compiled=engine.plan(observe([]));match.apply(action,route)
                execution.append(dict(time=match.t,event='known_map_replan',compiled=compiled))
            if step%25==0 or match.ref.finished:window.append(capture())
            if engine and (step%25==0 or match.ref.finished):execution.append(dict(time=match.t,**engine.public(match.t),camera_command=camera_command))
            if match.ref.finished:break
    dump(root/'observations'/f'{len(match.decisions):03d}.json',observe(window))
    # Ground truth is released only AFTER the final action and terminal capture.
    export(match,root)
    if engine:dump(root/'skill_execution.json',execution)
    dump(root/'pending.json',dict(finished=True,simulation_seconds=match.t))
    total_tokens=None
    if api_planner:
        from deepseek_client import usage_total
        usage=usage_total(api_planner.records);total_tokens=usage['total_tokens'];dump(root/'api_usage.json',usage)
    mode='qwen_semantic_trial' if args.qwen_vision else 'deepseek_semantic_trial' if api_planner else 'assistant_assessment_trial' if args.assessment else 'engineering_smoke' if args.vision_smoke else 'assistant_semantic_trial' if engine else 'assistant_visual_trial'
    dump(root/'result.json',dict(status='passed',mode=mode,
        simulation_seconds=match.t,frames=len(frames),decisions=len(match.decisions),score=match.ref.score,
        finish=match.ref.finished,final_hp=match.ref.hp,decision_wait_seconds=sum(d['decision_wait_seconds'] for d in match.decisions),
        tokens=total_tokens,isaac_dynamics=False,online_isaac_rgb=True,
        semantic_skills=bool(engine),skill_compile_seconds=sum(d.get('skill_compile_seconds',0) for d in match.decisions)))
    if api_planner or args.assessment:
        from vision_report import build
        build(root)


try:main()
except Exception:
    dump(args.run_dir/'result.json',dict(status='failed',error=traceback.format_exc()))
    traceback.print_exc()
    if args.geometry_diagnostic or args.geometry_interface_probe:raise SystemExit(1)
finally:app.close(wait_for_replicator=False,skip_cleanup=True)
