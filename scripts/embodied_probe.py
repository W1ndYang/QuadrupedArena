"""Preflight only: free-base Go2-W, joint targets, PhysX stepping and body RGB.

No match, no opponent policy, no high-level LLM decisions. Only the initial reset
writes root/joint states. All subsequent motion comes from actuator targets.
"""
import argparse
import json
import math
from pathlib import Path
import time
import traceback
from isaaclab.app import AppLauncher

from arena_paths import PROJECT, workspace_root
WORKSPACE=workspace_root()
parser=argparse.ArgumentParser()
parser.add_argument('--embodied-probe',action='store_true')
parser.add_argument('--run-dir',type=Path,required=True)
parser.add_argument('--manual-test',action='store_true',help='Three externally selected 2s skills; static visual target, not a match')
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args();args.enable_cameras=True
app=AppLauncher(args).app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation,ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.sensors import Camera,CameraCfg
from isaaclab.utils.math import quat_apply
from PIL import Image
from pxr import UsdGeom,UsdPhysics
from adaptive_motion import load_config


def dump(name,value):
    (args.run_dir/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def box(path,size,position,color):
    cfg=sim_utils.CuboidCfg(size=tuple(size),collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=.8,dynamic_friction=.7))
    cfg.func(path,cfg,translation=tuple(position))


def main():
    start=time.monotonic();dt=1/120
    cfg=load_config(PROJECT/'configs/trial_rotating.json')
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=dt,device=args.device,enable_scene_query_support=True,use_fabric=False))
    box('/World/Floor',(12,8,.2),(0,0,-.1),(.32,.34,.36))
    for c in cfg['arena']['covers']:
        box('/World/Covers/'+c['name'],c['size'],(*c['center'],c['size'][2]/2),(.45,.52,.58))
    light=sim_utils.DomeLightCfg(intensity=2000)
    light.func('/World/Light',light)
    # A stationary colored calibration object, not a combat opponent or label.
    box('/World/CameraCheck',(.25,.25,1.2),(-1,-3.3,.6),(.85,.12,.06))
    asset=WORKSPACE/'simulators/assets/quadruped-arena/go2w/go2w_description.usd'
    robot=Articulation(ArticulationCfg(prim_path='/World/Robot',
        spawn=sim_utils.UsdFileCfg(usd_path=str(asset),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=False,max_depenetration_velocity=1.),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(enabled_self_collisions=False,
                solver_position_iteration_count=8,solver_velocity_iteration_count=4)),
        init_state=ArticulationCfg.InitialStateCfg(pos=(-5,-2.8,.43),
            joint_pos={'.*_hip_joint':0.,'.*_thigh_joint':.8,'.*_calf_joint':-1.5,'.*_foot_joint':0.},joint_vel={'.*':0.}),
        actuators={
            'legs':ImplicitActuatorCfg(joint_names_expr=['.*_hip_joint','.*_thigh_joint','.*_calf_joint'],
                effort_limit_sim=23.5,velocity_limit_sim=30.,stiffness=80.,damping=4.),
            'wheels':ImplicitActuatorCfg(joint_names_expr=['.*_foot_joint'],
                effort_limit_sim=23.5,velocity_limit_sim=30.,stiffness=0.,damping=10.)}))
    bases=[str(p.GetPath()) for p in sim.stage.Traverse() if p.GetName()=='base' and p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(bases)!=1:raise RuntimeError('Could not uniquely resolve physical base: '+str(bases))
    # Derive the rigid mount from measured PhysX base pose, never move the robot
    # to satisfy the camera. Pose buffers must opt into updates (default false).
    camera_path='/World/OnboardCamera'
    camera=Camera(CameraCfg(prim_path=camera_path,update_period=0.,height=360,width=640,update_latest_camera_pose=True,
        data_types=['rgb'],spawn=sim_utils.PinholeCameraCfg(focal_length=12.,horizontal_aperture=24.,clipping_range=(.05,100)),
        offset=CameraCfg.OffsetCfg(pos=(0,0,0),rot=(1,0,0,0),convention='world')))
    sim.set_camera_view(eye=(-8,-7,5),target=(-3,-1,0))
    sim.reset()
    root=robot.data.default_root_state.clone()
    robot.write_root_pose_to_sim(root[:,:7]);robot.write_root_velocity_to_sim(root[:,7:])
    robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),robot.data.default_joint_vel.clone())
    robot.reset();camera.reset()
    camera_offset=torch.tensor([[.36,0,.10]],device=sim.device)
    def mount_camera():
        pos=robot.data.root_pos_w+quat_apply(robot.data.root_quat_w,camera_offset)
        camera.set_world_poses(pos,robot.data.root_quat_w,convention='world')
        return pos
    mount_camera()
    wheel_ids,wheel_names=robot.find_joints('.*_foot_joint')
    leg_ids,_=robot.find_joints(['.*_hip_joint','.*_thigh_joint','.*_calf_joint'])
    dump('setup.json',dict(asset=str(asset),joint_names=robot.joint_names,body_names=robot.body_names,wheel_names=wheel_names,
        camera_path=camera_path,physical_base_path=bases[0],camera_fov_degrees=90,physics_dt=dt,
        camera_mount='measured PhysX base pose + rigid sensor extrinsic; no root actuation',
        initialization_root_writes=1,runtime_root_writes=0,control='leg position PD + wheel velocity drives',
        scope='engineering preflight; no competition; no trained gait policy; original match not modified'))
    phases=[('settle',3.,0.,0.),('forward',3.,.25,0.),('brake',2.,0.,0.),
            ('turn',3.,0.,.5),('brake_turn',2.,0.,0.),('reverse',2.,-.15,0.),('final_stop',2.,0.,0.)]
    timeline=[];boundaries=[];tick=0
    manual_action={};decisions=[]
    if args.manual_test:
        from partial_observation import observation,validate_action
        for folder in ('observations','actions','decisions'):(args.run_dir/folder).mkdir()
        dump('manual_protocol.json',dict(kind='onboard visual-control integration; not combat',
            target='static orange calibration block',target_initial_position=[-1,-3.3],
            known_map=cfg['arena'],period_seconds=2,decisions=3,
            only_own_pose_and_rgb=True,no_enemy_truth_interface=True,
            limitations=['No moving humanoid, shooting, scoring or collapse in this preflight',
                'Known static calibration target is not a hidden dynamic opponent',
                'Low speed engineering controller, not a validated locomotion policy']))
        def manual_phases():
            nonlocal manual_action
            yield ('settle',3.,0.,0.)
            for index in range(3):
                prior=boundaries[-1]
                packet=observation(index,tick*dt-3,prior,{},prior['image'],[-1,-3.3])
                dump(f'observations/{index:03}.json',packet)
                print('OBSERVATION_READY '+str(args.run_dir/'observations'/f'{index:03}.json'),flush=True)
                ready=time.monotonic();file=args.run_dir/'actions'/f'{index:03}.json'
                while not file.exists():
                    if time.monotonic()-ready>480:raise TimeoutError('Paused observation expired without an action')
                    time.sleep(.1)
                request=json.loads(file.read_text())
                if request.get('observation_id')!=index:raise ValueError('Wrong observation id')
                manual_action=validate_action(request['action'])
                # No navigation policy here: direct local motion only, with a
                # known-wall segment gate. The model chooses the destination.
                if manual_action['skill']=='MOVE':
                    from trial_motion import Planner
                    planner=Planner(cfg,cfg['arena']['covers'])
                    if not planner.segment(np.array(prior['position'][:2]),np.array(manual_action['goal'])):
                        raise ValueError('Direct local MOVE intersects map clearance')
                entry=dict(request,decision_wait_seconds=time.monotonic()-ready,tokens=None,
                    token_status='not exposed by assistant session',simulation_start=tick*dt-3)
                decisions.append(entry);dump(f'decisions/{index:03}.json',entry)
                yield (f'decision_{index:03}',2.,0.,0.)
                dump(f'decisions/{index:03}_outcome.json',dict(simulation_end=tick*dt-3,self_state=boundaries[-1]))
            packet=observation(3,tick*dt-3,boundaries[-1],{},boundaries[-1]['image'],[-1,-3.3])
            dump('observations/003.json',packet)
        phases=manual_phases()
    for name,duration,v,w in phases:
        steering_integral=0.
        for _ in range(round(duration/dt)):
            if not app.is_running():raise RuntimeError('App stopped before preflight completion')
            if args.manual_test and name!='settle':
                q=robot.data.root_quat_w[0].tolist()
                yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
                v=w=0.
                if manual_action['skill'] in ('MOVE','TURN'):
                    if manual_action['skill']=='MOVE':
                        delta=np.array(manual_action['goal'])-robot.data.root_pos_w[0,:2].cpu().numpy()
                        distance=float(np.linalg.norm(delta));target=math.atan2(delta[1],delta[0])
                    else:distance=0.;target=manual_action['yaw']
                    error=math.atan2(math.sin(target-yaw),math.cos(target-yaw))
                    w=float(np.clip(1.5*error,-.5,.5)) if abs(error)>.04 else 0.
                    if manual_action['skill']=='MOVE' and distance>.06 and abs(error)<.3:
                        v=min(.25,.8*distance)*math.cos(error)
            actual_w=float(robot.data.root_ang_vel_w[0,2])
            steering_integral=float(np.clip(steering_integral+(w-actual_w)*dt,-1.5,1.5)) if w else 0.
            steering=float(np.clip(w+2.*(w-actual_w)+2.*steering_integral,-4.,4.)) if w else 0.
            speeds=[(v+(-1 if n.startswith(('FL','RL')) else 1)*steering*.40/2)/.089 for n in wheel_names]
            robot.set_joint_position_target(robot.data.default_joint_pos[:,leg_ids],joint_ids=leg_ids)
            robot.set_joint_velocity_target(torch.tensor([speeds],device=sim.device),joint_ids=wheel_ids)
            robot.write_data_to_sim();sim.step(render=False);robot.update(dt)
            if tick%12==0:mount_camera();sim.render()
            camera.update(dt)
            tick+=1
            q=robot.data.root_quat_w[0].tolist();p=robot.data.root_pos_w[0].tolist()
            up=1-2*(q[1]**2+q[2]**2)
            yaw=math.atan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
            state=dict(t=round(tick*dt,6),phase=name,position=p,yaw=yaw,up_z=up,
                linear_velocity=robot.data.root_lin_vel_w[0].tolist(),angular_velocity=robot.data.root_ang_vel_w[0].tolist())
            if tick%12==0:timeline.append(state)
            if not np.isfinite(p+q).all() or p[2]<.16 or up<.65:
                dump('timeline.json',timeline);raise RuntimeError('Physical stability gate failed; no match started')
        expected_camera=mount_camera()[0].cpu().numpy()
        for _ in range(4):sim.render()
        camera.update(dt,force_recompute=True)
        rgb=camera.data.output['rgb'][0,:,:,:3].cpu().numpy()
        if rgb.std()<2:raise RuntimeError('Camera image is blank')
        Image.fromarray(rgb).save(args.run_dir/(name+'_rgb.png'))
        state=dict(state,image=name+'_rgb.png',camera_position=camera.data.pos_w[0].tolist(),
            camera_mount_error=float(np.linalg.norm(camera.data.pos_w[0].cpu().numpy()-expected_camera)),
            wheel_velocities=robot.data.joint_vel[0,wheel_ids].tolist(),wheel_targets=speeds,
            wheel_torque=robot.data.applied_torque[0,wheel_ids].tolist())
        boundaries.append(state);dump('timeline.json',timeline);dump('boundaries.json',boundaries)
        print('PROBE_PHASE '+json.dumps(state),flush=True)
    if args.manual_test:
        dump('decisions.json',decisions)
        dump('result.json',dict(status='passed',kind='manual_onboard_control_preflight',
            physics_motion=True,simulation_seconds=6,settling_seconds=3,decisions=len(decisions),
            runtime_root_pose_writes=0,match_started=False,opponent_is_static_calibration_target=True,
            camera_max_mount_error=max(s['camera_mount_error'] for s in boundaries),
            decision_wait_seconds=sum(e['decision_wait_seconds'] for e in decisions),tokens=None))
        return
    b={s['phase']:s for s in boundaries}
    distance=float(np.linalg.norm(np.asarray(b['forward']['position'][:2])-b['settle']['position'][:2]))
    turn=math.atan2(math.sin(b['turn']['yaw']-b['brake']['yaw']),math.cos(b['turn']['yaw']-b['brake']['yaw']))
    stopped=float(np.linalg.norm(b['final_stop']['linear_velocity'][:2]))
    checks=dict(stable=min(s['up_z'] for s in timeline)>.8,
        forward_motion=.4<distance<1.2,turn_response=.5<turn<2.3,stops=stopped<.06,
        camera_attached=all(.2<np.linalg.norm(np.asarray(s['camera_position'])-s['position'])<.6 for s in boundaries),
        camera_extrinsics=all(s['camera_mount_error']<1e-4 for s in boundaries),
        camera_moves_with_body=bool(np.linalg.norm(np.asarray(b['forward']['camera_position'])-b['settle']['camera_position'])>.4))
    result=dict(status='passed' if all(checks.values()) else 'failed',checks=checks,
        forward_distance_m=distance,turn_angle_radians=turn,final_planar_speed=stopped,
        simulation_seconds=tick*dt,wall_seconds=time.monotonic()-start,physics_motion=True,
        runtime_root_pose_writes=0,match_started=False,llm_decisions=0)
    dump('result.json',result);print('PROBE_RESULT '+json.dumps(result),flush=True)


try:main()
except BaseException:
    error=traceback.format_exc();(args.run_dir/'error.txt').write_text(error)
    dump('result.json',dict(status='failed',error=error,match_started=False));print(error,flush=True)
    raise
finally:app.close(wait_for_replicator=False,skip_cleanup=True)
