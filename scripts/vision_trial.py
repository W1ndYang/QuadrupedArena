"""Live kinematic referee + Isaac RGB sensor, externally paused decisions.

Private match state stays in memory until termination. Never calls Match.observe.
Isaac renders transforms only: no dynamics and no tactical overlays.
"""
import argparse
import copy
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
parser.add_argument('--synthetic-vision-mvp',action='store_true',
                    help='One-shot simulator-derived foot sighting, Skill validation and commit; no external action')
parser.add_argument('--synthetic-vision-mvp-validation',action='store_true',
                    help='Three finite current-scene sighting cycles: clear, High2 occluded, moved clear')
parser.add_argument('--synthetic-vision-skill-runtime-smoke',action='store_true',
                    help='Three native Skill actions in the normal match loop with fresh synthetic observations')
parser.add_argument('--synthetic-pixel-noise-std',type=float,default=1.0)
parser.add_argument('--synthetic-miss-probability',type=float,default=0.0)
parser.add_argument('--synthetic-seed',type=int,default=20261028)
parser.add_argument('--diagnostic-output-dir',type=Path,
                    help='Diagnostic artifacts directory (default: this run/geometry_diagnostic)')
parser.add_argument('--run-dir',type=Path,required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.enable_cameras=True
if (args.synthetic_vision_mvp or args.synthetic_vision_mvp_validation or args.synthetic_vision_skill_runtime_smoke) and (
        not args.vision_trial or not args.semantic_skills or
        args.geometry_diagnostic or args.geometry_interface_probe or args.vision_smoke or
        args.deepseek or args.qwen_vision or args.assessment or args.g1b1_visible_spawn or
        args.diagnostic_output_dir or
        sum((args.synthetic_vision_mvp, args.synthetic_vision_mvp_validation,
             args.synthetic_vision_skill_runtime_smoke)) != 1):
    parser.error('Synthetic vision modes require --vision-trial --semantic-skills and exclude other diagnostic/planner modes')
if args.synthetic_pixel_noise_std<0 or not math.isfinite(args.synthetic_pixel_noise_std) or not 0<=args.synthetic_miss_probability<=1 or not math.isfinite(args.synthetic_miss_probability):
    parser.error('Invalid synthetic sighting noise or miss probability')
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
    if args.g1b1_visible_spawn or args.synthetic_vision_mvp or args.synthetic_vision_mvp_validation:
        # Explicit sighting modes use the previously checked clear A viewpoint.
        cfg['start_a']=[4.8,-1.0]
    match=VisionMatch(cfg,initial_a_yaw=math.pi/2 if (args.g1b1_visible_spawn or args.synthetic_vision_mvp or args.synthetic_vision_mvp_validation) else 0.);rig=CameraRig()
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
        if args.synthetic_vision_mvp or args.synthetic_vision_mvp_validation or args.synthetic_vision_skill_runtime_smoke:
            protocol.update(mode='synthetic_vision_skill_runtime_smoke' if args.synthetic_vision_skill_runtime_smoke else
                                 'synthetic_vision_mvp_validation' if args.synthetic_vision_mvp_validation else 'simulator_derived_synthetic_vision_mvp',
                perception='Current USD camera/G1 mesh and PhysX cover visibility produce a noisy foot pixel; not RGB-only recognition.',
                sighting_source='simulator_geometry_truth',same_frame_verified=False,
                sync_integration_status='blocked',automatic_one_shot=bool(args.synthetic_vision_mvp),
                synthetic_seed=args.synthetic_seed,pixel_noise_std=args.synthetic_pixel_noise_std,
                miss_probability=args.synthetic_miss_probability)
            if args.synthetic_vision_mvp_validation:
                protocol.update(engine='Python kinematics/referee, Isaac RGB capture, current USD/PhysX geometry query; no PhysX stepping',
                                validation_case_count=3)
            if args.synthetic_vision_skill_runtime_smoke:
                protocol.update(engine='Normal Python match and Skill execution; Isaac RGB plus current USD/PhysX geometry query',
                                driver='smoke_test_driver',native_skill_actions=3,
                                diagnostic_pose_override=False)
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
    def observe(window,state=None,index=None):
        current=match.states[-1] if state is None else state
        packet=public_packet(len(match.decisions) if index is None else index,current,window,cfg['start_b'])
        if engine:packet['skills']=engine.public(current['t'])
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
        dump(root/'result.json',dict(status=summary['status'],mode='geometry_interface_probe',
            probe_status=summary['probe_status'],sync_integration_status=summary['sync_integration_status'],
            runtime_verified=False,
            same_frame_verified=False,probe_output_dir=str(output.resolve()),
            diagnostic_bundle=str(output.resolve()/'diagnostic_bundle.zip')))
        print(f'Interface probe output: {output.resolve()}',flush=True)
        print(f'Interface probe bundle: {output.resolve()/"diagnostic_bundle.zip"}',flush=True)
        print(f'Interface probe status: {summary["probe_status"]}; runtime_verified=false',flush=True)
        return
    def capture(state=None):
        s=match.states[-1] if state is None else state
        yaw,heading,eye=pose_scene(s)
        for _ in range(12 if not frames else 6):app.update()
        pixels=np.asarray(rgb.get_data())
        if pixels.shape[:2]!=(432,768) or pixels[...,:3].std()<1:raise RuntimeError('Missing or blank onboard image')
        name=f'frames/{len(frames):03d}_{s["t"]:05.2f}.png';Image.fromarray(pixels[...,:3]).save(root/name)
        frame=dict(file=name,simulation_seconds=s['t'],own_xy=s['xy'][0],body_yaw_degrees=math.degrees(yaw),
            camera_pan_degrees=rig.pan,camera_world_yaw_degrees=math.degrees(heading),camera_eye=eye.tolist(),horizontal_fov_degrees=90)
        frames.append(frame);dump(root/'frame_index.json',frames)
        return frame
    if args.synthetic_vision_mvp:
        from synthetic_vision_mvp import observe_current_scene,handoff_to_skill
        frame=capture()
        packet=observe([frame])
        packet['perception']='One-shot simulator-derived synthetic foot pixel; RGB saved for review, not used for detection.'
        dump(root/'observations/000.json',packet)
        with Image.open(root/frame['file']) as image:image_size=image.size
        synthetic=observe_current_scene(stage,camera,packet,frame,cfg,image_size,
            seed=args.synthetic_seed,pixel_noise_std=args.synthetic_pixel_noise_std,
            miss_probability=args.synthetic_miss_probability)
        synthetic=handoff_to_skill(synthetic,engine,packet)
        dump(root/'synthetic_vision_mvp.json',synthetic)
        accepted=synthetic['skill_handoff']['status']=='accepted_and_committed'
        dump(root/'result.json',dict(status='passed' if accepted else 'partial',
            mode='simulator_derived_synthetic_vision_mvp',sighting_source='simulator_geometry_truth',
            sync_integration_status='blocked',same_frame_verified=False,
            observation_file='observations/000.json',diagnostic_file='synthetic_vision_mvp.json',
            sighting_emitted=synthetic['status']=='emitted',
            skill_handoff_status=synthetic['skill_handoff']['status']))
        print(f'Synthetic vision MVP: {synthetic["status"]}; '
              f'Skill: {synthetic["skill_handoff"]["status"]}; same_frame_verified=false',flush=True)
        return
    if args.synthetic_vision_mvp_validation:
        from synthetic_vision_mvp import observe_current_scene, run_validation_cases, validation_pose
        match.apply(dict(skill='HOLD',facing='keep',attack=False),[])
        def sample_case(index,name):
            if index:
                match.advance(1.0)
            pose=validation_pose(index,match.states[-1],cfg)
            state=pose['state']
            frame=capture(state)
            packet=observe([frame],state,index)
            packet['perception']='Simulator-derived current scene foot pixel; RGB is review evidence, not the detector.'
            dump(root/'observations'/f'{index:03d}.json',packet)
            with Image.open(root/frame['file']) as image:image_size=image.size
            try:
                synthetic=observe_current_scene(stage,camera,packet,frame,cfg,image_size,
                    seed=args.synthetic_seed+index,pixel_noise_std=args.synthetic_pixel_noise_std,
                    miss_probability=args.synthetic_miss_probability)
            except Exception as exc:
                synthetic=dict(status='not_emitted',reason=f'adapter_failed: {type(exc).__name__}: {exc}',
                    sighting=None,candidates=[],source='simulator_geometry_truth',
                    simulation_time_seconds=frame['simulation_seconds'],
                    same_frame_verified=False,sync_integration_status='blocked')
            anchors={}
            cache=UsdGeom.XformCache()
            for label,actor in zip(('A','B'),actors):
                prim=stage.GetPrimAtPath(f'/World/Actors/{label}/{actor.root_link}')
                point=cache.GetLocalToWorldTransform(prim).Transform(Gf.Vec3d(0,0,0))
                anchors[label]=[float(point[k]) for k in range(3)]
            return dict(frame=frame,packet=packet,synthetic=synthetic,
                        authored_actor_xy=pose['authored_actor_xy'],
                        authored_pose_reason=pose['reason'],
                        stage_actor_root_link_world_m=anchors)
        def save_progress(records):
            dump(root/'synthetic_vision_mvp_validation.json',dict(status='running',
                mode='synthetic_vision_mvp_validation',source='simulator_geometry_truth',
                sync_integration_status='blocked',same_frame_verified=False,cases=records))
        summary=run_validation_cases(engine,sample_case,save_progress)
        dump(root/'synthetic_vision_mvp_validation.json',summary)
        dump(root/'result.json',dict(status=summary['status'],mode=summary['mode'],
            case_statuses={item['case']:item['status'] for item in summary['cases']},
            diagnostic_file='synthetic_vision_mvp_validation.json',
            sync_integration_status='blocked',same_frame_verified=False))
        print(f'Synthetic vision validation: {summary["status"]}; '
              f'cases={[item["status"] for item in summary["cases"]]}',flush=True)
        return
    smoke_mode=args.synthetic_vision_skill_runtime_smoke
    if smoke_mode:
        from synthetic_vision_mvp import smoke_observation, smoke_request, summarize_skill_runtime_smoke, observe_current_scene
        smoke_cycles=[]
        def stage_actor_state():
            cache=UsdGeom.XformCache()
            anchors={};forward_axes={}
            for label,actor in zip(('A','B'),actors):
                try:
                    prim=stage.GetPrimAtPath(f'/World/Actors/{label}/{actor.root_link}')
                    world=cache.GetLocalToWorldTransform(prim)
                    point=world.Transform(Gf.Vec3d(0,0,0))
                    forward=world.Transform(Gf.Vec3d(1,0,0))
                    anchors[label]=[float(point[k]) for k in range(3)]
                    forward_axes[label]=[float(forward[k]-point[k]) for k in range(3)]
                except Exception as exc:
                    anchors[label]={'status':'unavailable','reason':f'{type(exc).__name__}: {exc}'}
                    forward_axes[label]={'status':'unavailable','reason':f'{type(exc).__name__}: {exc}'}
            state=match.states[-1]
            return dict(actor_xy=copy.deepcopy(state['xy']),actor_yaw=copy.deepcopy(state['yaw']),
                        stage_root_link_world_m=anchors,stage_root_link_forward_world=forward_axes,
                        simulation_seconds=state['t'])
        def save_smoke_progress():
            dump(root/'synthetic_vision_skill_runtime_smoke.json',dict(status='running',
                mode='synthetic_vision_skill_runtime_smoke',driver='smoke_test_driver',
                source='simulator_geometry_truth',sync_integration_status='blocked',
                same_frame_verified=False,cycles=smoke_cycles))
    window=[capture()]
    while not match.ref.finished:
        index=len(match.decisions);packet=observe(window)
        smoke_record=None
        if smoke_mode:
            packet['perception']='Current simulator-derived synthetic foot pixel; RGB saved for review, not used for detection.'
            dump(root/'observations'/f'{index:03d}.json',packet)
            frame=window[-1]
            smoke_record=dict(cycle_index=index,start_status='started',end_status='running',
                observation_id=packet['observation_id'],simulation_seconds=packet['simulation_seconds'],
                frame=frame,actor_before=stage_actor_state(),
                memory_before=copy.deepcopy(engine.memory.track),
                sync_integration_status='blocked',same_frame_verified=False)
            smoke_cycles.append(smoke_record);save_smoke_progress()
            with Image.open(root/frame['file']) as image:image_size=image.size
            synthetic=smoke_observation(packet,frame,
                lambda current_packet,current_frame: observe_current_scene(
                    stage,camera,current_packet,current_frame,cfg,image_size,
                    seed=args.synthetic_seed+index,pixel_noise_std=args.synthetic_pixel_noise_std,
                    miss_probability=args.synthetic_miss_probability))
            smoke_record['synthetic']=synthetic
            smoke_record['no_sighting_reason']=synthetic.get('reason') if synthetic.get('sighting') is None else None
            if index>=3:
                smoke_record.update(end_status='observed_no_action',
                    action_status='not_requested_terminal_observation',
                    validator_status='not_attempted',skill_commit_status='not_attempted',
                    memory_after_commit=copy.deepcopy(engine.memory.track),
                    memory_check_passed=True)
                save_smoke_progress()
        if not smoke_mode:dump(root/'observations'/f'{index:03d}.json',packet)
        if smoke_mode and index>=3:break
        ready=time.monotonic();readyutc=datetime.datetime.now(datetime.timezone.utc).isoformat()
        api_timing=None
        if smoke_mode:
            request=smoke_request(index,packet,synthetic.get('sighting'))
            smoke_record['action_request']=request
            smoke_record['action_status']='requested'
            save_smoke_progress()
        else:
            dump(root/'pending.json',dict(observation_id=index,simulation_seconds=match.t,ready_utc=readyutc))
            print(f'PUBLIC READY {index} t={match.t:.2f}',flush=True)
            path=root/'actions'/f'{index:03d}.json'
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
        if engine:
            engine.validate(motion,packet)
            if smoke_mode:smoke_record['validator_status']='accepted'
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
            if smoke_mode:
                smoke_record['skill_commit_status']='accepted'
                smoke_record['memory_after_commit']=copy.deepcopy(engine.memory.track)
                sighting=synthetic.get('sighting')
                smoke_record['sighting_committed']=bool(sighting is not None and
                    engine.memory.track.get('seen_at')==packet['simulation_seconds'] and
                    engine.memory.track.get('feet')==sighting['feet'])
                smoke_record['memory_check_passed']=(
                    smoke_record['sighting_committed'] if sighting is not None else
                    engine.memory.track==smoke_record['memory_before'])
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
        if smoke_mode:
            before=smoke_record['actor_before'];after=stage_actor_state()
            a0=before['actor_xy'][0];a1=after['actor_xy'][0]
            match_changed=math.dist(a0,a1)>.02 or abs(after['actor_yaw'][0]-before['actor_yaw'][0])>.03
            p0=before['stage_root_link_world_m']['A'];p1=after['stage_root_link_world_m']['A']
            stage_changed=isinstance(p0,list) and isinstance(p1,list) and math.dist(p0,p1)>.02
            smoke_record.update(action_status='executed',end_status='completed',compiled_action=action,
                compiled_skill=compiled,route_count=len(route),camera_command=camera_command,
                actor_after=after,steps_executed=step,
                simulation_time_advanced=after['simulation_seconds']-before['simulation_seconds'],
                match_state_changed=match_changed,stage_geometry_changed=stage_changed,
                actual_state_change=bool(match_changed and stage_changed))
            save_smoke_progress()
    if smoke_mode:
        summary=summarize_skill_runtime_smoke(smoke_cycles)
        dump(root/'synthetic_vision_skill_runtime_smoke.json',summary)
        dump(root/'result.json',dict(status=summary['status'],mode=summary['mode'],
            checks=summary['checks'],diagnostic_file='synthetic_vision_skill_runtime_smoke.json',
            sync_integration_status='blocked',same_frame_verified=False))
        print(f'Synthetic Skill runtime smoke: {summary["status"]}; checks={summary["checks"]}',flush=True)
        return
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
    if (args.geometry_diagnostic or args.geometry_interface_probe or args.synthetic_vision_mvp or
            args.synthetic_vision_mvp_validation or args.synthetic_vision_skill_runtime_smoke):raise SystemExit(1)
finally:app.close(wait_for_replicator=False,skip_cleanup=True)
